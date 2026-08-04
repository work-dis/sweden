#!/usr/bin/env python3
"""Build the Värmland forest-habitat overlays used by index.html.

Input files are the county 17 GeoTIFF downloads from Skogsstyrelsen:
SLU Skogskarta volumes (pine, spruce, birch, oak and beech), mean diameter,
and the classified SLU soil-moisture map. The output is a display PNG and a
grayscale score PNG for each mushroom species.

This is a habitat-suitability model, not a probability of fruiting or finding
a mushroom. It deliberately does not use GBIF observations in the score.
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
    print("Запусти через .venv: .venv/bin/python3 scripts/build_varmland_habitat.py ...", file=sys.stderr)
    sys.exit(1)


BBOX_WGS84 = (12.70, 59.25, 14.25, 59.52)
WIDTH = 1800
HEIGHT = 720

COLORS = {
    "cib": (232, 163, 23),
    "tr": (139, 107, 61),
    "black": (78, 62, 91),
    "regalis": (167, 67, 48),
    "matsutake": (42, 110, 79),
}

FILE_HINTS = {
    "pine": ("TallVolym",),
    "spruce": ("GranVolym",),
    "birch": ("birch", "BjörkVolym", "BjorkVolym"),
    "oak": ("EkVolym",),
    "beech": ("BokVolym",),
    "diameter": ("Medeldiameter",),
    "moisture": ("Markfuktighet",),
}

SOIL_CATEGORIES = {
    0: "unknown",
    1: "sand-gravel",
    2: "moraine",
    3: "clay-silt",
    4: "peat-wetland",
    5: "rock-thin-soil",
    255: "water",
}


def find_raster(folder: Path, hints: tuple[str, ...]) -> Path:
    candidates = list(folder.rglob("*.tif")) + list(folder.rglob("*.tiff"))
    for hint in hints:
        for path in candidates:
            if hint.lower() in path.name.lower():
                return path
    raise FileNotFoundError(f"Raster not found for any of: {', '.join(hints)}")


def read_wgs84_grid(path: Path, *, categorical: bool = False) -> np.ndarray:
    with rasterio.open(path) as src:
        with WarpedVRT(
            src,
            crs="EPSG:4326",
            transform=from_bounds(*BBOX_WGS84, WIDTH, HEIGHT),
            width=WIDTH,
            height=HEIGHT,
            src_nodata=src.nodata,
            nodata=np.nan,
            dtype="float32",
            resampling=Resampling.nearest if categorical else Resampling.bilinear,
        ) as vrt:
            return vrt.read(1)


def moisture_preference(classes: np.ndarray, values: tuple[float, float, float]) -> np.ndarray:
    result = np.zeros_like(classes, dtype=np.float32)
    for category, value in enumerate(values, start=1):
        result[classes == category] = value
    return result


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


def read_soil_grid(paths: list[Path]) -> np.ndarray:
    shapes: list[tuple[dict, int]] = []
    for path in paths:
        collection = json.loads(path.read_text(encoding="utf-8"))
        for feature in collection.get("features", []):
            category = soil_category(str(feature.get("properties", {}).get("jg2_tx", "")))
            if category:
                shapes.append((feature["geometry"], category))
    if not shapes:
        return np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    return rasterize(
        shapes,
        out_shape=(HEIGHT, WIDTH),
        transform=from_bounds(*BBOX_WGS84, WIDTH, HEIGHT),
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )


def make_soil_display(classes: np.ndarray) -> np.ndarray:
    palette = {
        1: (232, 194, 86, 125),
        2: (151, 116, 80, 105),
        3: (151, 92, 155, 125),
        4: (67, 139, 181, 125),
        5: (125, 130, 133, 105),
    }
    rgba = np.zeros((*classes.shape, 4), dtype=np.uint8)
    for category, color in palette.items():
        rgba[classes == category] = color
    return rgba


def suitability(
    host: np.ndarray,
    moisture: np.ndarray,
    maturity: np.ndarray,
    forest_structure: np.ndarray,
    forest_mask: np.ndarray,
    soil: np.ndarray,
) -> np.ndarray:
    # Structure adjusts the score inside forest, but never creates suitability
    # outside the independently calculated forest mask.
    structure_weight = 0.70 + 0.30 * forest_structure
    score = structure_weight * (
        0.52 * host + 0.23 * moisture + 0.17 * maturity + 0.08 * soil
    )
    score[~forest_mask] = 0
    return np.clip(np.nan_to_num(score) * 100.0, 0, 100)


def make_display(score: np.ndarray, forest_mask: np.ndarray) -> np.ndarray:
    """Render only actionable candidates, using fixed semantic score colours."""
    rgba = np.zeros((*score.shape, 4), dtype=np.uint8)
    bands = (
        (40, 60, (235, 190, 58, 150)),
        (60, 75, (72, 132, 79, 190)),
        (75, 101, (24, 85, 50, 225)),
    )
    for low, high, colour in bands:
        rgba[forest_mask & (score >= low) & (score < high)] = colour
    return rgba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("assets/habitat"), type=Path)
    parser.add_argument("--soil-geojson", nargs="*", default=[], type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = {name: find_raster(args.input_dir, hints) for name, hints in FILE_HINTS.items()}
    data = {
        name: read_wgs84_grid(path, categorical=name == "moisture")
        for name, path in paths.items()
    }

    pine = np.maximum(data["pine"], 0)
    spruce = np.maximum(data["spruce"], 0)
    birch = np.maximum(data["birch"], 0)
    oak = np.maximum(data["oak"], 0)
    beech = np.maximum(data["beech"], 0)
    total = pine + spruce + birch + oak + beech
    safe_total = np.maximum(total, 1)
    shares = [item / safe_total for item in (pine, spruce, birch, oak, beech)]
    pine_s, spruce_s, birch_s, oak_s, beech_s = shares

    # A hard mask prevents fields, roads and built-up pixels from receiving a
    # habitat colour. The volume threshold is deliberately conservative.
    # Mean diameter is a proxy for stand maturity, not a measured stand age.
    forest_mask = (
        np.isfinite(total)
        & np.isfinite(data["diameter"])
        & (total >= 20)
        & np.isin(data["moisture"], (1, 2, 3))
    )
    forest_structure = np.clip((total - 20) / 70, 0, 1)
    maturity = np.clip((np.maximum(data["diameter"], 0) - 8) / 22, 0, 1)
    moist = data["moisture"]
    Image.fromarray((forest_mask * 255).astype(np.uint8), "L").save(
        args.output_dir / "forest-mask.png", optimize=True
    )
    soil_classes = read_soil_grid(args.soil_geojson)
    Image.fromarray(soil_classes, "L").save(args.output_dir / "soil-category.png", optimize=True)
    Image.fromarray(make_soil_display(soil_classes), "RGBA").save(
        args.output_dir / "soil-overlay.png", optimize=True
    )

    host = {
        "cib": np.clip(0.58 + 0.28 * spruce_s + 0.18 * (oak_s + beech_s) + 0.08 * birch_s, 0, 1),
        "tr": np.clip(0.10 + 0.85 * spruce_s + 0.22 * pine_s + 0.12 * birch_s, 0, 1),
        "black": np.clip(0.03 + 1.60 * (oak_s + beech_s) + 0.22 * birch_s, 0, 1),
        "regalis": np.clip(0.12 + 0.62 * spruce_s + 0.38 * pine_s + 0.32 * birch_s, 0, 1),
        "matsutake": np.clip(0.03 + 0.97 * pine_s, 0, 1),
    }
    moisture = {
        "cib": moisture_preference(moist, (0.72, 1.00, 0.42)),
        "tr": moisture_preference(moist, (0.38, 1.00, 0.72)),
        "black": moisture_preference(moist, (0.62, 1.00, 0.38)),
        "regalis": moisture_preference(moist, (0.72, 1.00, 0.55)),
        "matsutake": moisture_preference(moist, (1.00, 0.30, 0.04)),
    }
    soil = {
        "cib": np.choose(np.minimum(soil_classes, 5), (0.55, 0.72, 1.00, 0.82, 0.18, 0.58)),
        "tr": np.choose(np.minimum(soil_classes, 5), (0.50, 0.32, 0.90, 0.72, 0.62, 0.42)),
        "black": np.choose(np.minimum(soil_classes, 5), (0.35, 0.18, 0.58, 1.00, 0.08, 0.38)),
        "regalis": np.choose(np.minimum(soil_classes, 5), (0.52, 0.62, 1.00, 0.62, 0.22, 0.62)),
        "matsutake": np.choose(np.minimum(soil_classes, 5), (0.28, 1.00, 0.42, 0.04, 0.00, 0.78)),
    }
    for values in soil.values():
        values[soil_classes == 255] = 0

    stats: dict[str, dict[str, float]] = {}
    for species in COLORS:
        score = suitability(
            host[species],
            moisture[species],
            maturity,
            forest_structure,
            forest_mask,
            soil[species],
        )
        score[soil_classes == 255] = 0
        Image.fromarray(np.rint(score * 2.55).astype(np.uint8), "L").save(
            args.output_dir / f"{species}-score.png", optimize=True
        )
        Image.fromarray(make_display(score, forest_mask), "RGBA").save(
            args.output_dir / f"{species}-overlay.png", optimize=True
        )
        forest_scores = score[forest_mask]
        stats[species] = {
            "p50": round(float(np.percentile(forest_scores, 50)), 1),
            "p90": round(float(np.percentile(forest_scores, 90)), 1),
            "max": round(float(np.max(forest_scores)), 1),
        }

    metadata = {
        "model": "varmland-habitat-v3",
        "generated": date.today().isoformat(),
        "bboxWgs84": list(BBOX_WGS84),
        "width": WIDTH,
        "height": HEIGHT,
        "cellSourceMetres": 12.5,
        "scoreMeaning": "habitat suitability, not mushroom occurrence probability",
        "forestMask": "tree volume >= 20, valid mean diameter and SLU moisture class",
        "displayBands": {"hidden": "<40", "candidate": "40-59", "good": "60-74", "best": "75-100"},
        "inputs": {name: path.name for name, path in paths.items()},
        "soilInput": [path.name for path in args.soil_geojson],
        "soilCategories": {str(key): value for key, value in SOIL_CATEGORIES.items()},
        "moistureClasses": {"1": "dry-fresh", "2": "fresh-moist", "3": "moist-wet"},
        "stats": stats,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
