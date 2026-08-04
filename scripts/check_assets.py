#!/usr/bin/env python3
"""Verify generated raster presence, integrity and metadata dimensions."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


EXPECTED = {
    Path("assets/habitat-sweden"): [
        "forest-mask", "forest-coverage", "forest-state", "forest-reference",
        *[f"{species}-score" for species in ("cib", "tr", "black", "regalis", "matsutake")],
        *[f"{species}-overlay-{level}" for species in ("cib", "tr", "black", "regalis", "matsutake") for level in (40, 60, 75)],
    ],
    Path("assets/habitat"): [
        "forest-mask", "soil-category", "soil-overlay",
        *[f"{species}-{kind}" for species in ("cib", "tr", "black", "regalis", "matsutake") for kind in ("score", "overlay")],
    ],
}


def validate_directory(directory: Path, names: list[str]) -> None:
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
        raise ValueError(f"missing or empty: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_size = (int(metadata["width"]), int(metadata["height"]))
    for name in names:
        path = directory / f"{name}.png"
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty: {path}")
        with Image.open(path) as image:
            if image.size != expected_size:
                raise ValueError(f"{path}: {image.size}, expected {expected_size}")
            image.verify()
    print(f"✓ {directory}: {len(names)} PNG, {expected_size[0]}×{expected_size[1]}")


def main() -> None:
    for directory, names in EXPECTED.items():
        validate_directory(directory, names)


if __name__ == "__main__":
    main()
