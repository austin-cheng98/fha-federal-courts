.PHONY: test reproduce validation clean

test:
	PYTHONPATH=src pytest -q

reproduce:
	python scripts/run_pipeline.py

validation:
	python scripts/score_goldset.py
	python scripts/validation_robustness.py

clean:
	rm -rf outputs data/processed/case_features.csv data/processed/analysis_panel.csv \
	  data/processed/feii_panel.csv data/processed/doctrine_map.csv \
	  data/processed/doctrine_divergence.csv data/processed/doctrine_transitions.csv \
	  data/raw/synthetic_corpus.jsonl data/external/synthetic_housing_panel.csv \
	  data/processed/_synth_truth.json .pytest_cache
