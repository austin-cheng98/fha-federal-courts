.PHONY: install test synthetic figures clean

install:
	pip install -e .

test:
	PYTHONPATH=src pytest -q

synthetic:
	python scripts/run_pipeline.py --source synthetic

figures:
	python scripts/make_paper_assets.py

clean:
	rm -rf outputs/__pycache__ src/fha/__pycache__ tests/__pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
