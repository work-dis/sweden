
import pytest
import numpy as np

class TestVarmlandLogic:
    """Test the Värmland habitat builder logic."""
    
    def test_soil_category_sand(self):
        from build_varmland_habitat import soil_category
        assert soil_category("Sand") == 1
        assert soil_category("isälvssediment") == 1
        assert soil_category("grus") == 1
        
    def test_soil_category_clay(self):
        from build_varmland_habitat import soil_category
        assert soil_category("Lera") == 3
        assert soil_category("silt") == 3
        assert soil_category("gyttja") == 3
        
    def test_soil_category_peat(self):
        from build_varmland_habitat import soil_category
        assert soil_category("Torv") == 4
        assert soil_category("kärr") == 4
        
    def test_soil_category_rock(self):
        from build_varmland_habitat import soil_category
        assert soil_category("Urberg") == 5
        assert soil_category("berg") == 5
        
    def test_soil_category_water(self):
        from build_varmland_habitat import soil_category
        assert soil_category("Vatten") == 255
        
    def test_moisture_preference(self):
        from build_varmland_habitat import moisture_preference
        classes = np.array([[1, 2], [3, 0]], dtype=np.uint8)
        result = moisture_preference(classes, (0.5, 1.0, 0.3))
        assert result[0, 0] == 0.5  # Class 1 -> 0.5
        assert result[0, 1] == 1.0  # Class 2 -> 1.0
        assert result[1, 0] == 0.3  # Class 3 -> 0.3
        assert result[1, 1] == 0.0  # Class 0 -> 0.0

    def test_soil_categories_defined(self):
        from build_varmland_habitat import SOIL_CATEGORIES
        assert 0 in SOIL_CATEGORIES
        assert 255 in SOIL_CATEGORIES
        assert len(SOIL_CATEGORIES) >= 6

    def test_file_hints_defined(self):
        from build_varmland_habitat import FILE_HINTS
        expected = {"pine", "spruce", "birch", "oak", "beech", "diameter", "moisture"}
        assert set(FILE_HINTS.keys()) == expected

    def test_output_grid_uses_leaflet_web_mercator(self):
        from build_varmland_habitat import OUTPUT_BOUNDS, OUTPUT_CRS, BBOX_WGS84
        from rasterio.warp import transform

        assert OUTPUT_CRS == "EPSG:3857"
        west, south, east, north = BBOX_WGS84
        xs, ys = transform("EPSG:4326", OUTPUT_CRS, [west, east], [south, north])
        assert OUTPUT_BOUNDS == pytest.approx((xs[0], ys[0], xs[1], ys[1]))
