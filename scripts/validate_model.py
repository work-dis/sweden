#!/usr/bin/env python3
"""Validate habitat scores against spatially separated GBIF observations.

The model is fixed (it is not fitted on GBIF), so the spatial train partition is
reported for audit and only the held-out 10 km cells are used for metrics.
Pseudo-absence points are sampled reproducibly from mapped forest cells.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
from rasterio.warp import transform


SPECIES_KEYS = {
    "cib": 5249504,
    "tr": 2554536,
    "black": 2554662,
    "regalis": 5240248,
    "matsutake": 5241820,
}
SPECIES_NAMES = {
    "cib": "Лисичка обыкновенная",
    "tr": "Трубковидная лисичка",
    "black": "Чёрная лисичка",
    "regalis": "Королевский мухомор",
    "matsutake": "Мацутаке",
}
SWEDEN_BBOX = (10.40, 55.15, 24.30, 69.20)
GBIF_SEARCH_URL = "https://api.gbif.org/v1/occurrence/search"


@dataclass(frozen=True)
class Occurrence:
    lat: float
    lng: float
    uncertainty_m: float | None = None


@dataclass
class Metrics:
    taxon_key: int
    downloaded_records: int
    spatial_cells: int
    train_cells: int
    test_cells: int
    pseudo_absences: int
    auc_roc: float
    precision_at_k: float
    k: int


def habitat_score(lat: float, lng: float, score_img: np.ndarray) -> float | None:
    """Look up a 0..100 habitat score using the actual image dimensions."""
    west, south, east, north = SWEDEN_BBOX
    height, width = score_img.shape
    x = int((lng - west) / (east - west) * width)
    y = int((north - lat) / (north - south) * height)
    if x < 0 or x >= width or y < 0 or y >= height:
        return None
    return float(score_img[y, x]) / 255.0 * 100.0


def raster_index(lat: float, lng: float, shape: tuple[int, int]) -> int | None:
    west, south, east, north = SWEDEN_BBOX
    height, width = shape
    x = int((lng - west) / (east - west) * width)
    y = int((north - lat) / (north - south) * height)
    if x < 0 or x >= width or y < 0 or y >= height:
        return None
    return y * width + x


def grid_10km(lat: float, lng: float) -> tuple[int, int]:
    """Return a 10 km cell in Sweden's metric SWEREF 99 TM projection."""
    xs, ys = transform("EPSG:4326", "EPSG:3006", [lng], [lat])
    return int(xs[0] // 10_000), int(ys[0] // 10_000)


def deduplicate_spatially(records: list[Occurrence]) -> list[Occurrence]:
    """Keep the most precise record in each 10 km cell."""
    cells: dict[tuple[int, int], Occurrence] = {}
    xs, ys = transform(
        "EPSG:4326", "EPSG:3006",
        [record.lng for record in records], [record.lat for record in records],
    )
    for record, x, y in zip(records, xs, ys, strict=True):
        key = int(x // 10_000), int(y // 10_000)
        old = cells.get(key)
        old_u = float("inf") if old is None or old.uncertainty_m is None else old.uncertainty_m
        new_u = float("inf") if record.uncertainty_m is None else record.uncertainty_m
        if old is None or new_u < old_u:
            cells[key] = record
    return list(cells.values())


def spatial_split(
    records: list[Occurrence], test_fraction: float, seed: int
) -> tuple[list[Occurrence], list[Occurrence]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    test_size = max(1, int(round(len(records) * test_fraction)))
    test_ids = set(order[:test_size].tolist())
    train = [record for i, record in enumerate(records) if i not in test_ids]
    test = [record for i, record in enumerate(records) if i in test_ids]
    return train, test


def auc_roc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Mann–Whitney AUC with average ranks for tied scores."""
    values = np.concatenate((positive, negative)).astype(np.float64)
    labels = np.concatenate((np.ones(len(positive), dtype=bool), np.zeros(len(negative), dtype=bool)))
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    rank_sum = ranks[labels].sum()
    return float((rank_sum - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def precision_at_k(positive: np.ndarray, negative: np.ndarray) -> tuple[float, int]:
    labels = np.concatenate((np.ones(len(positive), dtype=np.uint8), np.zeros(len(negative), dtype=np.uint8)))
    scores = np.concatenate((positive, negative))
    k = len(positive)
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = scores > threshold
    tied = scores == threshold
    remaining = k - int(above.sum())
    expected_tied_positives = remaining * float(labels[tied].mean())
    precision = (float(labels[above].sum()) + expected_tied_positives) / k
    return precision, k


def _get_json(params: dict[str, object], attempts: int = 3) -> dict:
    request = Request(
        f"{GBIF_SEARCH_URL}?{urlencode(params)}",
        headers={"Accept": "application/json", "User-Agent": "sweden-mushroom-habitat/1.0"},
    )
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=45) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def fetch_occurrences(
    taxon_key: int, max_records: int | None = None, workers: int = 4
) -> list[Occurrence]:
    base_params = {
            "taxon_key": taxon_key,
            "country": "SE",
            "has_coordinate": "true",
            "has_geospatial_issue": "false",
            "occurrence_status": "present",
    }
    count_payload = _get_json({**base_params, "limit": 0})
    total = min(int(count_payload.get("count", 0)), 100_000)
    if max_records is not None:
        total = min(total, max_records)
    page_size = 300
    pages: dict[int, list[dict]] = {}

    def load_page(offset: int) -> tuple[int, list[dict]]:
        limit = min(page_size, total - offset)
        payload = _get_json({**base_params, "limit": limit, "offset": offset})
        return offset, payload.get("results", [])

    offsets = list(range(0, total, page_size))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(load_page, offset) for offset in offsets]
        for completed, future in enumerate(as_completed(futures), start=1):
            offset, items = future.result()
            pages[offset] = items
            if completed % 10 == 0 or completed == len(futures):
                print(f"  downloaded pages: {completed}/{len(futures)}", flush=True)

    records: list[Occurrence] = []
    for offset in sorted(pages):
        results = pages[offset]
        for item in results:
            lat, lng = item.get("decimalLatitude"), item.get("decimalLongitude")
            if lat is None or lng is None:
                continue
            uncertainty = item.get("coordinateUncertaintyInMeters")
            records.append(Occurrence(float(lat), float(lng), float(uncertainty) if uncertainty is not None else None))
    return records[:total]


def cache_path(cache_dir: Path, taxon_key: int, max_records: int | None) -> Path:
    suffix = "all" if max_records is None else str(max_records)
    return cache_dir / f"{taxon_key}-{suffix}.json"


def load_occurrences(
    cache_dir: Path, taxon_key: int, max_records: int | None, refresh: bool
) -> list[Occurrence]:
    path = cache_path(cache_dir, taxon_key, max_records)
    if path.exists() and not refresh:
        return [Occurrence(**item) for item in json.loads(path.read_text(encoding="utf-8"))]
    records = fetch_occurrences(taxon_key, max_records)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(record) for record in records]), encoding="utf-8")
    return records


def seed_for(species: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{species}:{base_seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def validate_species(
    species: str,
    taxon_key: int,
    records: list[Occurrence],
    score_img: np.ndarray,
    forest_indices: np.ndarray,
    negative_ratio: int,
    test_fraction: float,
    seed: int,
) -> Metrics:
    separated = deduplicate_spatially(records)
    if len(separated) < 10:
        raise ValueError(f"{species}: fewer than 10 occupied 10 km cells")
    train, test = spatial_split(separated, test_fraction, seed)
    scored_test = [(record, habitat_score(record.lat, record.lng, score_img)) for record in test]
    scored_test = [(record, score) for record, score in scored_test if score is not None]
    if not scored_test:
        raise ValueError(f"{species}: no test observations inside model bounds")
    positive = np.array([score for _, score in scored_test], dtype=np.float32)
    occupied = {
        index for record in separated
        if (index := raster_index(record.lat, record.lng, score_img.shape)) is not None
    }
    candidates = forest_indices[~np.isin(forest_indices, np.fromiter(occupied, dtype=np.int64))]
    negative_count = min(len(candidates), len(positive) * negative_ratio)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(candidates, size=negative_count, replace=False)
    negative = score_img.ravel()[sampled].astype(np.float32) / 255.0 * 100.0
    precision, k = precision_at_k(positive, negative)
    return Metrics(
        taxon_key=taxon_key,
        downloaded_records=len(records),
        spatial_cells=len(separated),
        train_cells=len(train),
        test_cells=len(positive),
        pseudo_absences=len(negative),
        auc_roc=round(auc_roc(positive, negative), 4),
        precision_at_k=round(precision, 4),
        k=k,
    )


def markdown_report(results: dict[str, Metrics], limited: bool) -> str:
    lines = [
        "# Валидация модели пригодности биотопов",
        "",
        f"Дата запуска: {date.today().isoformat()}. Источник наблюдений: GBIF Occurrence API.",
        "",
        "Метод: наблюдения GBIF дедуплицируются по 10-км ячейкам SWEREF 99 TM; "
        "20% ячеек откладываются в тест. Псевдоотсутствия воспроизводимо выбираются "
        "из лесных ячеек NMD в соотношении 10:1.",
        "",
    ]
    if limited:
        lines += ["> Это ограниченный проверочный запуск, а не итоговая валидация.", ""]
    lines += [
        "| Вид | Записей | Ячеек 10 км | Test | AUC-ROC | Precision@k |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for species, metrics in results.items():
        lines.append(
            f"| {SPECIES_NAMES[species]} | {metrics.downloaded_records} | "
            f"{metrics.spatial_cells} | {metrics.test_cells} | "
            f"{metrics.auc_roc:.3f} | {metrics.precision_at_k:.3f} (k={metrics.k}) |"
        )
    lines += [
        "",
        "AUC сравнивает оценки наблюдений и случайных лесных псевдоотсутствий. "
        "Precision@k рассчитана для k, равного числу тестовых наблюдений. "
        "GBIF отражает активность наблюдателей, а не истинное отсутствие вида.",
    ]
    return "\n".join(lines) + "\n"


def parse_taxon_ids(value: str) -> dict[str, int]:
    ids = [int(item) for item in value.split(",")]
    if len(ids) != len(SPECIES_KEYS):
        raise argparse.ArgumentTypeError("expected five comma-separated taxon ids")
    return dict(zip(SPECIES_KEYS, ids, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate habitat model against GBIF")
    parser.add_argument("--score-dir", type=Path, default=Path("assets/habitat-sweden"))
    parser.add_argument("--taxon-ids", type=parse_taxon_ids)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/gbif"))
    parser.add_argument("--max-records", type=int, help="limit per species for a smoke run")
    parser.add_argument("--negative-ratio", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    if args.negative_ratio < 1 or not 0 < args.test_fraction < 1:
        parser.error("negative-ratio must be >=1 and test-fraction must be between 0 and 1")

    taxa = args.taxon_ids or SPECIES_KEYS
    forest = np.asarray(Image.open(args.score_dir / "forest-mask.png").convert("L")) > 0
    forest_indices = np.flatnonzero(forest.ravel())
    results: dict[str, Metrics] = {}
    for species, taxon_key in taxa.items():
        print(f"{species}: loading GBIF taxon {taxon_key}...", flush=True)
        records = load_occurrences(args.cache_dir, taxon_key, args.max_records, args.refresh)
        score = np.asarray(Image.open(args.score_dir / f"{species}-score.png").convert("L"))
        if score.shape != forest.shape:
            raise ValueError(f"{species}: score and forest-mask dimensions differ")
        metrics = validate_species(
            species, taxon_key, records, score, forest_indices,
            args.negative_ratio, args.test_fraction, seed_for(species, args.seed),
        )
        results[species] = metrics
        print(f"  cells={metrics.spatial_cells}, test={metrics.test_cells}, "
              f"AUC={metrics.auc_roc:.3f}, precision@{metrics.k}={metrics.precision_at_k:.3f}")

    serializable = {species: asdict(metrics) for species, metrics in results.items()}
    if args.output_json:
        args.output_json.write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = markdown_report(results, limited=args.max_records is not None)
    if args.output_markdown:
        args.output_markdown.write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
