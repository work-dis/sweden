
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
