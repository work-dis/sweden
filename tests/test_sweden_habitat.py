
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Import the module (will fail if deps missing, which is fine for CI)
try:
    import sys
    sys.path.insert(0, "scripts")
except:
    pass

class TestClassScores:
    """Test the class_score function logic."""
    
    def test_class_score_assigns_values(self):
        """Test that class_score maps class codes to values."""
        from build_sweden_habitat import class_score
        classes = np.array([[23, 111], [112, 999]], dtype=np.uint16)
        values = {23: 45, 111: 58, 112: 62}
        result = class_score(classes, values)
        assert result[0, 0] == 45  # 23 -> 45
        assert result[0, 1] == 58  # 111 -> 58
        assert result[1, 0] == 62  # 112 -> 62
        assert result[1, 1] == 0   # 999 -> 0 (no mapping)

    def test_reduce_max_preserves_max(self):
        """Test that reduce_max keeps the maximum value in 2x2 blocks."""
        from build_sweden_habitat import reduce_max, PRESERVATION_FACTOR, HEIGHT, WIDTH
        # Create a 2x2 block with values 10, 20, 30, 40
        values = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        # Need to shape it properly: (HEIGHT*2, WIDTH*2)
        # Actually let's just test the logic directly
        values_wide = np.zeros((HEIGHT * PRESERVATION_FACTOR, WIDTH * PRESERVATION_FACTOR), dtype=np.uint8)
        values_wide[0, 0] = 10
        values_wide[0, 1] = 20
        values_wide[1, 0] = 30
        values_wide[1, 1] = 40
        result = reduce_max(values_wide)
        assert result[0, 0] == 40  # Max of 10,20,30,40

    def test_reduce_coverage_counts_forest(self):
        """Test that reduce_coverage counts non-zero subcells."""
        from build_sweden_habitat import reduce_coverage, PRESERVATION_FACTOR, HEIGHT, WIDTH
        mask = np.zeros((HEIGHT * PRESERVATION_FACTOR, WIDTH * PRESERVATION_FACTOR), dtype=bool)
        mask[0, 0] = True  # 1 of 4 subcells is forest
        mask[0, 1] = True  # 2 of 4
        result = reduce_coverage(mask)
        assert result[0, 0] == 2  # 2 of 4 subcells

    def test_water_dominant_cell_is_not_displayed_as_forest(self):
        from build_sweden_habitat import classify_surface_coverage

        forest, temporary = classify_surface_coverage(
            np.array([[1, 2, 1]], dtype=np.uint8),
            np.array([[0, 0, 0]], dtype=np.uint8),
            np.array([[3, 2, 0]], dtype=np.uint8),
        )

        # Water majority and a water/forest tie are both hidden. An inland
        # fragment is preserved when no sampled water competes with it.
        assert forest.tolist() == [[False, False, True]]
        assert not temporary.any()

    def test_water_dominant_cell_is_not_temporary_forest(self):
        from build_sweden_habitat import classify_surface_coverage

        forest, temporary = classify_surface_coverage(
            np.zeros((1, 3), dtype=np.uint8),
            np.array([[1, 2, 1]], dtype=np.uint8),
            np.array([[3, 2, 0]], dtype=np.uint8),
        )

        assert not forest.any()
        assert temporary.tolist() == [[False, False, True]]

    def test_reduce_mode_preserves_category(self):
        from build_sweden_habitat import reduce_mode, PRESERVATION_FACTOR, HEIGHT, WIDTH
        values = np.zeros((HEIGHT * PRESERVATION_FACTOR, WIDTH * PRESERVATION_FACTOR), dtype=np.uint8)
        values[0, 0] = 3
        values[0, 1] = 3
        values[1, 0] = 3
        values[1, 1] = 1
        result = reduce_mode(values, (1, 2, 3, 4, 5, 255))
        assert result[0, 0] == 3

    def test_categorical_preference_does_not_overflow(self):
        from build_sweden_habitat import apply_categorical_preference
        score = np.array([[80, 80, 80]], dtype=np.uint8)
        classes = np.array([[1, 2, 255]], dtype=np.uint8)
        result = apply_categorical_preference(score, classes, [1.0, 1.0, 0.5], 0.15)
        assert result.tolist() == [[80, 74, 68]]
        assert result.max() <= 100

    def test_forest_classes_defined(self):
        """Test that forest classes are properly defined."""
        from build_sweden_habitat import FOREST_CLASSES, TEMPORARY_FOREST_CLASSES
        assert len(FOREST_CLASSES) > 0
        assert len(TEMPORARY_FOREST_CLASSES) > 0
        assert all(c in (23, 43, *range(111, 118), *range(121, 128)) for c in FOREST_CLASSES)

    def test_class_scores_defined_for_all_species(self):
        """Test that all 5 species have score tables."""
        from build_sweden_habitat import CLASS_SCORES
        assert set(CLASS_SCORES.keys()) == {"cib", "tr", "black", "regalis", "matsutake"}
        for species, scores in CLASS_SCORES.items():
            assert len(scores) > 0
            # All scores should be between 0 and 100
            for v in scores.values():
                assert 0 <= v <= 100, f"{species}: score {v} out of range"

    def test_display_bands_ordered(self):
        """Test display bands are ordered low->high."""
        from build_sweden_habitat import DISPLAY_BANDS
        for i in range(len(DISPLAY_BANDS) - 1):
            assert DISPLAY_BANDS[i][1] <= DISPLAY_BANDS[i+1][0]

    def test_output_grid_uses_leaflet_web_mercator(self):
        from build_sweden_habitat import OUTPUT_BOUNDS, OUTPUT_CRS, SWEDEN_BBOX_WGS84
        from rasterio.warp import transform

        assert OUTPUT_CRS == "EPSG:3857"
        west, south, east, north = SWEDEN_BBOX_WGS84
        xs, ys = transform("EPSG:4326", OUTPUT_CRS, [west, east], [south, north])
        assert OUTPUT_BOUNDS == pytest.approx((xs[0], ys[0], xs[1], ys[1]))

    def test_web_mercator_grid_does_not_treat_latitude_as_linear(self):
        from build_sweden_habitat import OUTPUT_BOUNDS, OUTPUT_CRS, SWEDEN_BBOX_WGS84
        from rasterio.warp import transform

        latitude = 58.65
        _, ys = transform("EPSG:4326", OUTPUT_CRS, [13.25], [latitude])
        projected_ratio = (OUTPUT_BOUNDS[3] - ys[0]) / (OUTPUT_BOUNDS[3] - OUTPUT_BOUNDS[1])
        linear_ratio = (SWEDEN_BBOX_WGS84[3] - latitude) / (
            SWEDEN_BBOX_WGS84[3] - SWEDEN_BBOX_WGS84[1]
        )
        # Across Sweden the old latitude-linear placement was off by hundreds
        # of source rows around Vanern; the projected ratios must not coincide.
        assert abs(projected_ratio - linear_ratio) > 0.03

    def test_generated_vanern_pixels_are_water(self):
        import json
        from pathlib import Path
        from PIL import Image
        from rasterio.warp import transform

        directory = Path("assets/habitat-sweden")
        metadata = json.loads((directory / "metadata.json").read_text())
        west, south, east, north = metadata["outputBounds"]
        width, height = metadata["width"], metadata["height"]
        state = Image.open(directory / "forest-state.png")
        reference = Image.open(directory / "forest-reference.png").convert("RGBA")

        for lat, lng in ((58.65, 13.25), (58.85, 13.20), (58.95, 13.45)):
            xs, ys = transform("EPSG:4326", metadata["outputCrs"], [lng], [lat])
            x = int((xs[0] - west) / (east - west) * width)
            y = int((north - ys[0]) / (north - south) * height)
            assert state.getpixel((x, y)) == 0
            assert reference.getpixel((x, y))[3] == 0
