.PHONY: test pipeline validation llm schelling reproduce clean

test:
	PYTHONPATH=src python3 -m pytest -q

pipeline:
	python3 scripts/run_pipeline.py

validation:
	python3 scripts/score_goldset.py
	python3 scripts/validation_robustness.py
	python3 scripts/draw_random_sample.py

llm:
	python3 scripts/score_llm_baseline.py
	python3 scripts/analyze_prevalence.py

schelling: pipeline
	python3 scripts/run_schelling.py

reproduce: pipeline validation llm schelling

clean:
	rm -rf outputs data/processed/case_features.csv data/processed/analysis_panel.csv \
	  data/processed/feii_panel.csv data/processed/doctrine_map.csv \
	  data/processed/doctrine_divergence.csv data/processed/doctrine_transitions.csv \
	  data/raw/synthetic_corpus.jsonl data/external/synthetic_housing_panel.csv \
	  data/processed/_synth_truth.json .pytest_cache
