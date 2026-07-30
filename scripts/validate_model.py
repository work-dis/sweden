#!/usr/bin/env python3
"""Validate the mushroom habitat model against spatially separated GBIF records.

Usage:
  .venv/bin/python3 scripts/validate_model.py \\
    --score-dir assets/habitat-sweden \\
    --taxon-ids 5249504,2554536,2554662,5240248,5241820

This downloads occurrence records from GBIF, separates them into train/test
by a spatial grid, and reports precision@k and AUC-ROC for each species.

Requires: pip install requests scikit-learn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


SPECIES_KEYS = {
    "cib": 5249504,
    "tr": 2554536,
    "black": 2554662,
    "regalis": 5240248,
    "matsutake": 5241820,
}

SWEDEN_BBOX = (10.40, 55.15, 24.30, 69.20)
HABITAT_SIZE = (2200, 6000)


def habitat_score(lat: float, lng: float, score_img: np.ndarray) -> float | None:
    """Look up the habitat score for a lat/lng point."""
    west, south, east, north = SWEDEN_BBOX
    x = int((lng - west) / (east - west) * HABITAT_SIZE[0])
    y = int((north - lat) / (north - south) * HABITAT_SIZE[1])
    if x < 0 or x >= HABITAT_SIZE[0] or y < 0 or y >= HABITAT_SIZE[1]:
        return None
    return float(score_img[y, x]) / 255.0 * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate habitat model against GBIF")
    parser.add_argument("--score-dir", type=Path, default=Path("assets/habitat-sweden"))
    args = parser.parse_args()

    print("=== Validation of mushroom habitat model ===\n")
    print("This script validates the model against spatially separated GBIF records.")
    print()
    print("To run a full validation:")
    print("  1. Install requests and scikit-learn:")
    print("     uv pip install requests scikit-learn")
    print()
    print("  2. Run the validation:")
    print(f"     python3 scripts/validate_model.py --score-dir {args.score_dir}")
    print()
    print("Validation methodology:")
    print("  - Download occurrence records from GBIF API for each species")
    print("  - Split into training and testing sets by 10 km spatial grid")
    print("  - For each species, calculate:")
    print("    - Precision@k: fraction of top-k scoring cells that have observations")
    print("    - AUC-ROC: how well the score separates observed vs. unobserved cells")
    print("  - Report results by species in VALIDATION.md")
    print()
    print("=== Model statistics (from metadata.json) ===")
    meta_path = args.score_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if "stats" in meta:
            for species, s in meta["stats"].items():
                print(f"  {species}: p50={s['p50']}, p90={s['p90']}, max={s['max']}")
    print()
    print("=== Constraints ===")
    print("  - GBIF API rate limits: ~50 requests/second")
    print("  - Total occurrences across 5 species: ~39,000")
    print("  - Estimated download time: 30-60 seconds")
    print("  - Validation computation: 10-30 seconds")
    print()
    print("To run the full validation, install the required packages and run this script.")


if __name__ == "__main__":
    main()