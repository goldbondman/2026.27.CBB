.PHONY: install test lint backtest
install:
	python -m pip install -r requirements.lock
	python -m pip install --no-deps -e .
test:
	pytest
lint:
	ruff check src tests
backtest:
	python -m cbb.backtest.run --config configs/production/r1_r2.yaml
