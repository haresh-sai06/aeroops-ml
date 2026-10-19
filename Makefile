PY ?= python
export PYTHONPATH := src

.PHONY: data pipeline features kpis train-delay train-rul explain train serve bench dashboard test lint docker

data:            ## download BTS Jan-Mar 2024 + NASA C-MAPSS
	$(PY) scripts/download_bts.py --year 2024 --months 1 2 3
	bash scripts/download_cmapss.sh
pipeline:        ## raw CSV -> bronze -> silver (+DQ) -> gold parquet
	$(PY) -m aeroops.flights.pipeline
features:        ## Spark feature pipeline + online feature snapshot
	$(PY) -m aeroops.flights.features
kpis:            ## DuckDB SQL KPI layer -> reports/kpis.md
	$(PY) -m aeroops.kpis
train-delay:
	$(PY) -m aeroops.flights.train_delay
train-rul:
	$(PY) -m aeroops.rul.train
explain:
	$(PY) -m aeroops.rul.explain
train: pipeline features kpis train-delay train-rul explain
serve:
	uvicorn aeroops.serving.api:app --port 8000
bench:           ## requires `make serve` in another shell
	$(PY) scripts/benchmark_api.py --requests 2000 --concurrency 1 16
dashboard:
	streamlit run dashboard/app.py
test:
	$(PY) -m pytest -q
lint:
	ruff check src tests scripts dashboard
docker:
	docker build -t aeroops-api .
