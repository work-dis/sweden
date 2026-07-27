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
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling


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


def suitability(
    host: np.ndarray,
    moisture: np.ndarray,
    maturity: np.ndarray,
    forest_structure: np.ndarray,
) -> np.ndarray:
    score = forest_structure * (0.55 * host + 0.25 * moisture + 0.20 * maturity)
    return np.clip(np.nan_to_num(score) * 100.0, 0, 100)


def make_display(score: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    rgba = np.zeros((*score.shape, 4), dtype=np.uint8)
    strength = np.clip(score / 100.0, 0, 1)
    low = np.array((250, 225, 60), dtype=np.float32)
    high = np.array(color, dtype=np.float32)
    rgb = low[None, None, :] * (1 - strength[..., None]) + high[None, None, :] * strength[..., None]
    rgba[..., :3] = rgb.astype(np.uint8)
    rgba[..., 3] = np.where(score >= 20, 45 + strength * 175, 0).astype(np.uint8)
    return rgba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("assets/habitat"), type=Path)
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

    # Total estimated volume is used only as a soft forest-structure mask.
    # Mean diameter is a proxy for stand maturity, not a measured stand age.
    forest_structure = np.clip((total - 8) / 80, 0, 1)
    maturity = np.clip((np.maximum(data["diameter"], 0) - 8) / 22, 0, 1)
    moist = data["moisture"]

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

    stats: dict[str, dict[str, float]] = {}
    for species in COLORS:
        score = suitability(host[species], moisture[species], maturity, forest_structure)
        Image.fromarray(np.rint(score * 2.55).astype(np.uint8), "L").save(
            args.output_dir / f"{species}-score.png", optimize=True
        )
        Image.fromarray(make_display(score, COLORS[species]), "RGBA").save(
            args.output_dir / f"{species}-overlay.png", optimize=True
        )
        forest_scores = score[forest_structure > 0.15]
        stats[species] = {
            "p50": round(float(np.percentile(forest_scores, 50)), 1),
            "p90": round(float(np.percentile(forest_scores, 90)), 1),
            "max": round(float(np.max(forest_scores)), 1),
        }

    metadata = {
        "model": "varmland-habitat-v1",
        "generated": "2026-07-27",
        "bboxWgs84": list(BBOX_WGS84),
        "width": WIDTH,
        "height": HEIGHT,
        "cellSourceMetres": 12.5,
        "scoreMeaning": "habitat suitability, not mushroom occurrence probability",
        "inputs": {name: path.name for name, path in paths.items()},
        "moistureClasses": {"1": "dry-fresh", "2": "fresh-moist", "3": "moist-wet"},
        "stats": stats,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
