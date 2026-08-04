#!/usr/bin/env python3
"""Build nationwide mushroom forest-type suitability rasters from NMD2023.

The input is Naturvårdsverket's NMD2023 base layer v2.1. It classifies pine,
spruce, mixed conifer, mixed deciduous/conifer, common deciduous and noble
deciduous forest, separately on firm and wet ground, at 10 m.

Optional inputs for refined scoring:
  --soil-geojson      : SGU Jordarter GeoJSON for soil type nationwide
  --moisture-raster   : SLU Markfuktighet classified moisture GeoTIFF

The web output is intentionally aggregated. It identifies potentially suitable
host-forest types across Sweden; moisture, soil chemistry, stand continuity,
recent logging and fruiting weather still need separate checks.

Usage:
  .venv/bin/python3 scripts/build_sweden_habitat.py --base-raster <path> --output-dir assets/habitat-sweden
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

try:
    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import Resampling
except ImportError as e:
    print(f"Ошибка: {e}", file=sys.stderr)
    print("Запусти через .venv:", file=sys.stderr)
    print("  .venv/bin/python3 scripts/build_sweden_habitat.py ...", file=sys.stderr)
    print("  Или создай .venv: uv venv && uv pip install numpy rasterio Pillow", file=sys.stderr)
    sys.exit(1)


SWEDEN_BBOX_WGS84 = (10.40, 55.15, 24.30, 69.20)
WIDTH = 2200
HEIGHT = 6000
# Sample at twice the web resolution and reduce 2 x 2 cells afterwards. Reading
# one dominant class per web cell removes narrow and fragmented forests whenever
# another land-cover class dominates a ~200-260 m output cell.
PRESERVATION_FACTOR = 2

FOREST_CLASSES = (23, 43, *range(111, 118), *range(121, 128))
TEMPORARY_FOREST_CLASSES = (118, 128)

# Absolute scores communicate how strongly a mapped forest class matches each
# species. Codes 111-117 are firm-ground forest; 121-127 are wet-ground forest.
CLASS_SCORES = {
    "cib": {
        23: 45, 43: 50,
        111: 58, 112: 78, 113: 72, 114: 78, 115: 65, 116: 82, 117: 76,
        121: 45, 122: 62, 123: 58, 124: 62, 125: 45, 126: 55, 127: 52,
    },
    "tr": {
        23: 50, 43: 35,
        111: 35, 112: 78, 113: 65, 114: 58, 115: 25, 116: 20, 117: 28,
        121: 40, 122: 88, 123: 78, 124: 72, 125: 35, 126: 25, 127: 38,
    },
    "black": {
        23: 5, 43: 8,
        111: 8, 112: 12, 113: 12, 114: 35, 115: 45, 116: 90, 117: 72,
        121: 5, 122: 8, 123: 8, 124: 25, 125: 30, 126: 68, 127: 55,
    },
    "regalis": {
        23: 48, 43: 58,
        111: 70, 112: 82, 113: 80, 114: 75, 115: 50, 116: 25, 117: 40,
        121: 55, 122: 68, 123: 65, 124: 62, 125: 40, 126: 20, 127: 32,
    },
    "matsutake": {
        23: 18, 43: 35,
        111: 92, 112: 15, 113: 65, 114: 35, 115: 5, 116: 5, 117: 8,
        121: 20, 122: 5, 123: 15, 124: 8, 125: 2, 126: 2, 127: 3,
    },
}

DISPLAY_BANDS = (
    (40, 60, (235, 190, 58, 150)),
    (60, 75, (72, 132, 79, 190)),
    (75, 101, (24, 85, 50, 225)),
)

SOIL_CATEGORIES = {
    0: "unknown",
    1: "sand-gravel",
    2: "moraine",
    3: "clay-silt",
    4: "peat-wetland",
    5: "rock-thin-soil",
    255: "water",
}

# Soil preference per species: [unknown, sand, moraine, clay, peat, rock]
# Used when --soil-geojson is provided
SOIL_PREFERENCE = {
    "cib":      [0.55, 0.72, 1.00, 0.82, 0.18, 0.58],
    "tr":       [0.50, 0.32, 0.90, 0.72, 0.62, 0.42],
    "black":    [0.35, 0.18, 0.58, 1.00, 0.08, 0.38],
    "regalis":  [0.52, 0.62, 1.00, 0.62, 0.22, 0.62],
    "matsutake":[0.28, 1.00, 0.42, 0.04, 0.00, 0.78],
}

# Moisture preference per species: [dry-fresh, fresh-moist, moist-wet]
# Used when --moisture-raster is provided
MOISTURE_PREFERENCE = {
    "cib":      [0.72, 1.00, 0.42],
    "tr":       [0.38, 1.00, 0.72],
    "black":    [0.62, 1.00, 0.38],
    "regalis":  [0.72, 1.00, 0.55],
    "matsutake":[1.00, 0.30, 0.04],
}


def read_grid(raster: Path, *, categorical: bool = False) -> np.ndarray:
    width = WIDTH * PRESERVATION_FACTOR
    height = HEIGHT * PRESERVATION_FACTOR
    with rasterio.open(raster) as src:
        with WarpedVRT(
            src,
            crs="EPSG:4326",
            transform=from_bounds(*SWEDEN_BBOX_WGS84, width, height),
            width=width,
            height=height,
            src_nodata=src.nodata,
            nodata=0 if categorical else np.nan,
            dtype="uint16" if categorical else "float32",
            resampling=Resampling.nearest if categorical else Resampling.bilinear,
        ) as vrt:
            return vrt.read(1)


def class_score(classes: np.ndarray, values: dict[int, int]) -> np.ndarray:
    score = np.zeros(classes.shape, dtype=np.uint8)
    for code, value in values.items():
        score[classes == code] = value
    return score


def reduce_max(values: np.ndarray) -> np.ndarray:
    return values.reshape(
        HEIGHT, PRESERVATION_FACTOR, WIDTH, PRESERVATION_FACTOR
    ).max(axis=(1, 3))


def reduce_coverage(mask: np.ndarray) -> np.ndarray:
    return mask.reshape(
        HEIGHT, PRESERVATION_FACTOR, WIDTH, PRESERVATION_FACTOR
    ).sum(axis=(1, 3))


def reduce_mean(values: np.ndarray) -> np.ndarray:
    return values.reshape(
        HEIGHT, PRESERVATION_FACTOR, WIDTH, PRESERVATION_FACTOR
    ).mean(axis=(1, 3))


def reduce_mode(values: np.ndarray, categories: tuple[int, ...]) -> np.ndarray:
    """Reduce the fine grid by the most frequent categorical value per block."""
    blocks = values.reshape(
        HEIGHT, PRESERVATION_FACTOR, WIDTH, PRESERVATION_FACTOR
    )
    result = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    best_count = (blocks == 0).sum(axis=(1, 3))
    for category in categories:
        count = (blocks == category).sum(axis=(1, 3))
        replace = count > best_count
        result[replace] = category
        best_count[replace] = count[replace]
    return result


def apply_categorical_preference(
    score: np.ndarray,
    classes: np.ndarray,
    preferences: tuple[float, ...] | list[float],
    weight: float,
) -> np.ndarray:
    """Blend a 0..100 score with a categorical preference without overflow."""
    factor = np.full(classes.shape, preferences[0], dtype=np.float32)
    for category, value in enumerate(preferences[1:], start=1):
        factor[classes == category] = value
    factor[classes == 255] = 0.0
    refined = score.astype(np.float32) * ((1.0 - weight) + weight * factor)
    return np.rint(np.clip(refined, 0, 100)).astype(np.uint8)


def make_forest_reference(
    coverage: np.ndarray, temporary_coverage: np.ndarray
) -> np.ndarray:
    rgba = np.zeros((*coverage.shape, 4), dtype=np.uint8)
    temporary = (coverage == 0) & (temporary_coverage > 0)
    rgba[temporary, :3] = (184, 126, 48)
    rgba[temporary, 3] = np.take(
        np.array([0, 68, 82, 96, 108], dtype=np.uint8),
        temporary_coverage[temporary],
    )
    present = coverage > 0
    rgba[present, :3] = (75, 88, 72)
    rgba[present, 3] = np.take(
        np.array([0, 75, 92, 108, 120], dtype=np.uint8), coverage[present]
    )
    return rgba


def make_display(
    score: np.ndarray, forest_mask: np.ndarray, coverage: np.ndarray, minimum: int
) -> np.ndarray:
    rgba = np.zeros((*score.shape, 4), dtype=np.uint8)
    for low, high, colour in DISPLAY_BANDS:
        effective_low = max(low, minimum)
        selected = forest_mask & (score >= effective_low) & (score < high)
        rgba[selected] = colour
        # A forest found in only one of the four preservation subcells remains
        # visible, but is drawn more softly than a cell filled with forest.
        rgba[selected, 3] = np.minimum(
            rgba[selected, 3],
            np.take(
                np.array([0, 110, 145, 180, 225], dtype=np.uint8),
                coverage[selected],
            ),
        )
    return rgba


def soil_category(name: str) -> int:
    value = name.casefold()
    if "vatten" in value:
        return 255
    if "torv" in value or "kärr" in value or "mosse" in value:
        return 4
    if any(term in value for term in ("lera", "silt", "gyttja", "svämsediment")):
        return 3
    if "morän" in value:
        return 2
    if any(term in value for term in ("sand", "grus", "isälvssediment", "klapper")):
        return 1
    if any(term in value for term in ("urberg", "berg", "häll")):
        return 5
    return 0


def read_soil_grid(paths: list[Path]) -> np.ndarray | None:
    fine_w, fine_h = WIDTH * PRESERVATION_FACTOR, HEIGHT * PRESERVATION_FACTOR
    shapes: list[tuple[dict, int]] = []
    for path in paths:
        collection = json.loads(path.read_text(encoding="utf-8"))
        for feature in collection.get("features", []):
            category = soil_category(str(feature.get("properties", {}).get("jg2_tx", "")))
            if category:
                shapes.append((feature["geometry"], category))
    if not shapes:
        return None
    return rasterize(
        shapes,
        out_shape=(fine_h, fine_w),
        transform=from_bounds(*SWEDEN_BBOX_WGS84, fine_w, fine_h),
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build nationwide mushroom forest-type suitability rasters."
    )
    parser.add_argument("--base-raster", required=True, type=Path,
                        help="NMD2023 base layer GeoTIFF (basskikt v2.1)")
    parser.add_argument("--output-dir", default=Path("assets/habitat-sweden"), type=Path)
    parser.add_argument("--soil-geojson", nargs="*", default=[], type=Path,
                        help="SGU Jordarter GeoJSON files for soil type nationwide")
    parser.add_argument("--moisture-raster", type=Path, default=None,
                        help="SLU Markfuktighet classified moisture GeoTIFF")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Read base NMD2023 raster
    classes = read_grid(args.base_raster, categorical=True)
    fine_forest_mask = np.isin(classes, FOREST_CLASSES)
    fine_temporary_mask = np.isin(classes, TEMPORARY_FOREST_CLASSES)
    forest_coverage = reduce_coverage(fine_forest_mask)
    temporary_coverage = reduce_coverage(fine_temporary_mask)
    forest_mask = forest_coverage > 0
    temporary_mask = (~forest_mask) & (temporary_coverage > 0)

    soil_classes = read_soil_grid(args.soil_geojson) if args.soil_geojson else None
    if soil_classes is not None:
        print("  SGU soil data loaded, refining scores with soil preference")
        soil_web = reduce_mode(soil_classes, (1, 2, 3, 4, 5, 255))
    else:
        soil_web = None

    if args.moisture_raster and args.moisture_raster.exists():
        print(f"  Reading moisture raster: {args.moisture_raster.name}")
        moisture_web = reduce_mode(
            read_grid(args.moisture_raster, categorical=True), (1, 2, 3)
        )
    else:
        moisture_web = None

    # Write forest mask outputs
    Image.fromarray((forest_mask * 255).astype(np.uint8)).save(
        args.output_dir / "forest-mask.png", compress_level=6
    )
    Image.fromarray(
        np.rint(forest_coverage / (PRESERVATION_FACTOR**2) * 255).astype(np.uint8)
    ).save(args.output_dir / "forest-coverage.png", compress_level=6)
    forest_state = np.zeros(forest_mask.shape, dtype=np.uint8)
    forest_state[forest_mask] = 1
    forest_state[temporary_mask] = 2
    Image.fromarray(forest_state).save(
        args.output_dir / "forest-state.png", compress_level=6
    )
    Image.fromarray(
        make_forest_reference(forest_coverage, temporary_coverage)
    ).save(
        args.output_dir / "forest-reference.png", compress_level=6
    )

    # Score per species
    stats: dict[str, dict[str, float]] = {}
    for species, values in CLASS_SCORES.items():
        score = reduce_max(class_score(classes, values))

        # If soil data is available, refine the score (Phase 1.3)
        if soil_web is not None and species in SOIL_PREFERENCE:
            score = apply_categorical_preference(
                score, soil_web, SOIL_PREFERENCE[species], 0.20
            )

        # If moisture data is available, refine the score (Phase 1.2)
        if moisture_web is not None and species in MOISTURE_PREFERENCE:
            score = apply_categorical_preference(
                score, moisture_web, [1.0, *MOISTURE_PREFERENCE[species]], 0.15
            )

        score_u8 = np.rint(score * 2.55).astype(np.uint8)
        Image.fromarray(score_u8).save(
            args.output_dir / f"{species}-score.png", compress_level=6
        )
        for minimum in (40, 60, 75):
            Image.fromarray(
                make_display(score, forest_mask, forest_coverage, minimum)
            ).save(
                args.output_dir / f"{species}-overlay-{minimum}.png",
                compress_level=6,
            )
        forest_scores = score[forest_mask]
        stats[species] = {
            "p50": round(float(np.percentile(forest_scores, 50)), 1),
            "p90": round(float(np.percentile(forest_scores, 90)), 1),
            "max": round(float(np.max(forest_scores)), 1),
        }

    metadata = {
        "model": "sweden-forest-class-v2",
        "generated": date.today().isoformat(),
        "bboxWgs84": list(SWEDEN_BBOX_WGS84),
        "width": WIDTH,
        "height": HEIGHT,
        "sourceCellMetres": 10,
        "preservationFactor": PRESERVATION_FACTOR,
        "aggregation": (
            "nearest samples at 2x web resolution, then maximum suitability "
            "and forest presence across each 2x2 group"
        ),
        "webRasterMeaning": "forest-type suitability, not full habitat or occurrence probability",
        "input": args.base_raster.name,
        "forestClasses": list(FOREST_CLASSES),
        "temporaryForestClasses": list(TEMPORARY_FOREST_CLASSES),
        "classScores": CLASS_SCORES,
        "displayBands": {
            "hidden": "<40",
            "candidate": "40-59",
            "good": "60-74",
            "best": "75-100",
        },
        "limitations": [
            "stand continuity and forest age are not included",
            "soil and moisture are included only when their optional inputs are supplied",
            "web raster is aggregated from the 10 m source",
            "small forest fragments are preserved but represented by a full web cell",
            "recent logging and fruiting weather are separate checks",
        ],
        "stats": stats,
        "inputs": {
            "baseRaster": args.base_raster.name,
            "soilGeoJson": [p.name for p in args.soil_geojson] if args.soil_geojson else [],
            "moistureRaster": args.moisture_raster.name if args.moisture_raster else None,
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
