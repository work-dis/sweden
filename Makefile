.PHONY: all sweden varmland validate test clean rebuild

SWEDEN_SCRIPT = scripts/build_sweden_habitat.py
VARMLAND_SCRIPT = scripts/build_varmland_habitat.py
SWEDEN_RASTER = data/raw/sweden/NMD2023bas_v2_1.tif
VARMLAND_INPUT = data/raw/varmland
SOIL_JSON = data/raw/varmland/soil/sgu-soil.json data/raw/varmland/soil/sgu-page-10000.json data/raw/varmland/soil/sgu-page-20000.json
SWEDEN_OUT = assets/habitat-sweden
VARMLAND_OUT = assets/habitat

PYTHON = .venv/bin/python3

# ---------- сборка ----------

all: sweden varmland validate

sweden: $(SWEDEN_OUT)/metadata.json

$(SWEDEN_OUT)/metadata.json: $(SWEDEN_SCRIPT) $(SWEDEN_RASTER)
	$(PYTHON) $(SWEDEN_SCRIPT) --base-raster $(SWEDEN_RASTER) --output-dir $(SWEDEN_OUT)

varmland: $(VARMLAND_OUT)/metadata.json

$(VARMLAND_OUT)/metadata.json: $(VARMLAND_SCRIPT) $(wildcard $(VARMLAND_INPUT)/*.tif) $(SOIL_JSON)
	$(PYTHON) $(VARMLAND_SCRIPT) --input-dir $(VARMLAND_INPUT) --output-dir $(VARMLAND_OUT) --soil-geojson $(SOIL_JSON)

# ---------- валидация ----------

validate:
	@echo "=== Проверка assets/habitat-sweden/ ==="
	@test -f $(SWEDEN_OUT)/metadata.json || (echo "  MISSING: metadata.json" && exit 1)
	@for f in forest-mask forest-coverage forest-state forest-reference cib-score tr-score black-score regalis-score matsutake-score; do \
		test -f $(SWEDEN_OUT)/$$f.png || (echo "  MISSING: $$f.png" && exit 1); \
	done
	@for s in cib tr black regalis matsutake; do \
		for t in 40 60 75; do \
			test -f $(SWEDEN_OUT)/$$s-overlay-$$t.png || (echo "  MISSING: $$s-overlay-$$t.png" && exit 1); \
		done; \
	done
	@echo "  ✓ Все файлы Sweden присутствуют"
	@echo "=== Проверка assets/habitat/ ==="
	@test -f $(VARMLAND_OUT)/metadata.json || (echo "  MISSING: metadata.json" && exit 1)
	@for f in forest-mask soil-category soil-overlay; do \
		test -f $(VARMLAND_OUT)/$$f.png || (echo "  MISSING: $$f.png" && exit 1); \
	done
	@for s in cib tr black regalis matsutake; do \
		test -f $(VARMLAND_OUT)/$$s-score.png || (echo "  MISSING: $$s-score.png" && exit 1); \
		test -f $(VARMLAND_OUT)/$$s-overlay.png || (echo "  MISSING: $$s-overlay.png" && exit 1); \
	done
	@echo "  ✓ Все файлы Värmland присутствуют"
	@echo "=== OK ==="

# ---------- тесты ----------

test:
	$(PYTHON) -m pytest tests/ -q

# ---------- валидация модели ----------

validate_model:
	@echo "=== Валидация модели по GBIF ==="
	@echo "Установи зависимости: uv pip install requests scikit-learn"
	@echo "Затем запусти:"
	@echo "  $(PYTHON) scripts/validate_model.py --score-dir assets/habitat-sweden"

# ---------- очистка ----------

clean:
	rm -f $(SWEDEN_OUT)/*.png $(SWEDEN_OUT)/metadata.json
	rm -f $(VARMLAND_OUT)/*.png $(VARMLAND_OUT)/metadata.json
	@echo "  ✓ Сгенерированные растры удалены"

# ---------- пересборка ----------

rebuild: clean all

# ---------- хостинг (заглушка) ----------

deploy:
	@echo "Укажи целевую директорию: make deploy DEST=/path/to/www"
	@test -n "$(DEST)" && cp -r assets index.html data/ $(DEST) && echo "  ✓ Deployed to $(DEST)"