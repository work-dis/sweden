.PHONY: all sweden varmland validate validate_model test clean rebuild deploy

SWEDEN_SCRIPT = scripts/build_sweden_habitat.py
VARMLAND_SCRIPT = scripts/build_varmland_habitat.py
SWEDEN_RASTER = data/raw/sweden/NMD2023bas_v2_1.tif
VARMLAND_INPUT = data/raw/varmland
SOIL_JSON = data/raw/varmland/soil/sgu-soil.json data/raw/varmland/soil/sgu-page-10000.json data/raw/varmland/soil/sgu-page-20000.json
SWEDEN_OUT = assets/habitat-sweden
VARMLAND_OUT = assets/habitat

PYTHON = .venv/bin/python3

# ---------- сборка ----------

all: sweden varmland
	$(MAKE) validate

sweden: $(SWEDEN_OUT)/metadata.json

$(SWEDEN_OUT)/metadata.json: $(SWEDEN_SCRIPT) $(SWEDEN_RASTER)
	$(PYTHON) $(SWEDEN_SCRIPT) --base-raster $(SWEDEN_RASTER) --output-dir $(SWEDEN_OUT)

varmland: $(VARMLAND_OUT)/metadata.json

$(VARMLAND_OUT)/metadata.json: $(VARMLAND_SCRIPT) $(wildcard $(VARMLAND_INPUT)/*.tif) $(SOIL_JSON)
	$(PYTHON) $(VARMLAND_SCRIPT) --input-dir $(VARMLAND_INPUT) --output-dir $(VARMLAND_OUT) --soil-geojson $(SOIL_JSON)

# ---------- валидация ----------

validate:
	$(PYTHON) scripts/check_assets.py

# ---------- тесты ----------

test:
	$(PYTHON) -m pytest tests/ -q

# ---------- валидация модели ----------

validate_model:
	$(PYTHON) scripts/validate_model.py --score-dir $(SWEDEN_OUT) \
		--output-json validation-results.json --output-markdown VALIDATION.md $(VALIDATION_ARGS)

# ---------- очистка ----------

clean:
	rm -f $(SWEDEN_OUT)/*.png $(SWEDEN_OUT)/metadata.json
	rm -f $(VARMLAND_OUT)/*.png $(VARMLAND_OUT)/metadata.json
	@echo "  ✓ Сгенерированные растры удалены"

# ---------- пересборка ----------

rebuild: clean
	$(MAKE) all
	$(MAKE) test

# ---------- хостинг (заглушка) ----------

deploy:
	@test -n "$(DEST)" || (echo "Укажи целевую директорию: make deploy DEST=/path/to/www" && exit 1)
	@mkdir -p "$(DEST)"
	cp -R assets index.html manifest.json sw.js "$(DEST)/"
	@echo "  ✓ Deployed to $(DEST) (исходные data/raw не копировались)"
