#!/usr/bin/env python3
"""Build nationwide mushroom forest-type suitability rasters from NMD2023.

The input is Naturvårdsverket's NMD2023 base layer v2.1. It classifies pine,
spruce, mixed conifer, mixed deciduous/conifer, common deciduous and noble
deciduous forest, separately on firm and wet ground, at 10 m.

The web output is intentionally aggregated. It identifies potentially suitable
host-forest types across Sweden; moisture, soil chemistry, stand continuity,
recent logging and fruiting weather still need separate checks.
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


SWEDEN_BBOX_WGS84 = (10.40, 55.15, 24.30, 69.20)
WIDTH = 2200
HEIGHT = 6000

FOREST_CLASSES = (23, 43, *range(111, 118), *range(121, 128))

# Absolute scores communicate how strongly a mapped forest class matches each
# species. Codes 111–117 are firm-ground forest; 121–127 are wet-ground forest.
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


def read_grid(raster: Path) -> np.ndarray:
    with rasterio.open(raster) as src:
        with WarpedVRT(
            src,
            crs="EPSG:4326",
            transform=from_bounds(*SWEDEN_BBOX_WGS84, WIDTH, HEIGHT),
            width=WIDTH,
            height=HEIGHT,
            src_nodata=65535,
            nodata=0,
            dtype="uint16",
            resampling=Resampling.mode,
        ) as vrt:
            return vrt.read(1)


def class_score(classes: np.ndarray, values: dict[int, int]) -> np.ndarray:
    score = np.zeros(classes.shape, dtype=np.float32)
    for code, value in values.items():
        score[classes == code] = value
    return score


def make_display(score: np.ndarray, forest_mask: np.ndarray, minimum: int) -> np.ndarray:
    rgba = np.zeros((*score.shape, 4), dtype=np.uint8)
    for low, high, colour in DISPLAY_BANDS:
        effective_low = max(low, minimum)
        rgba[forest_mask & (score >= effective_low) & (score < high)] = colour
    return rgba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-raster", required=True, type=Path)
    parser.add_argument(
        "--output-dir", default=Path("assets/habitat-sweden"), type=Path
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    classes = read_grid(args.base_raster)
    forest_mask = np.isin(classes, FOREST_CLASSES)

    Image.fromarray((forest_mask * 255).astype(np.uint8)).save(
        args.output_dir / "forest-mask.png", compress_level=6
    )

    stats: dict[str, dict[str, float]] = {}
    for species, values in CLASS_SCORES.items():
        score = class_score(classes, values)
        score_u8 = np.rint(score * 2.55).astype(np.uint8)
        Image.fromarray(score_u8).save(
            args.output_dir / f"{species}-score.png", compress_level=6
        )
        for minimum in (40, 60, 75):
            Image.fromarray(make_display(score, forest_mask, minimum)).save(
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
        "model": "sweden-forest-class-v1",
        "generated": "2026-07-28",
        "bboxWgs84": list(SWEDEN_BBOX_WGS84),
        "width": WIDTH,
        "height": HEIGHT,
        "sourceCellMetres": 10,
        "webRasterMeaning": "forest-type suitability, not full habitat or occurrence probability",
        "input": args.base_raster.name,
        "forestClasses": list(FOREST_CLASSES),
        "classScores": CLASS_SCORES,
        "displayBands": {
            "hidden": "<40",
            "candidate": "40-59",
            "good": "60-74",
            "best": "75-100",
        },
        "limitations": [
            "national layer does not include soil, moisture or stand continuity",
            "web raster is aggregated from the 10 m source",
            "recent logging and fruiting weather are separate checks",
        ],
        "stats": stats,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
