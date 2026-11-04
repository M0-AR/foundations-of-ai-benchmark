PYTHON ?= python
COMPOSE ?= docker compose

.PHONY: install data all labs test lint report preview demo check docker-build docker-bench docker-report clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) scripts/fetch_data.py

all:
	$(PYTHON) experiments/run_all.py

labs:
	$(PYTHON) labs/lab1_falsification.py
	$(PYTHON) labs/lab2_locality.py
	$(PYTHON) labs/lab3_reproducibility.py
	$(PYTHON) labs/lab4_calibration.py
	$(PYTHON) labs/lab5_null_market.py
	$(PYTHON) labs/build_summary.py

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check .

report:
	$(PYTHON) scripts/make_report.py

preview:
	$(PYTHON) scripts/build_preview.py

demo:
	$(PYTHON) scripts/make_demo.py

check:
	$(PYTHON) scripts/check_results.py

docker-build:
	$(COMPOSE) build

docker-bench:
	$(COMPOSE) run --rm bench

docker-report:
	$(COMPOSE) run --rm report

clean:
	rm -rf .pytest_cache .ruff_cache paper/figures/*.png docs/assets/demo.gif docs/assets/demo.mp4
