.PHONY: install demo test lint stack down
install:
	python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
	python -m pip install -e '.[train,dev]'
demo:
	python scripts/make_demo_data.py
	python -m paywatch.train --data data/demo.csv --output models/demo --dataset-kind synthetic --epochs 8
	python scripts/plot_results.py --models models/demo
	python scripts/benchmark.py --models models/demo
test:
	python -m pytest -q
lint:
	python -m ruff check .
stack:
	docker compose up --build --wait
down:
	docker compose down
