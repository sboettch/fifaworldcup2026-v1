PYTHON ?= python

.PHONY: help install collect live features models rubric report pipeline

help:
	@echo "Targets:"
	@echo "  install   Install Python dependencies"
	@echo "  collect   Refresh raw source data from collectors"
	@echo "  live      Collect and harmonize the latest 2026 live snapshot"
	@echo "  features  Rebuild processed dimensional and matchup tables"
	@echo "  models    Rebuild clustering, supervised models, and 2026 validation"
	@echo "  rubric    Build rubric-facing CSV and figure artifacts"
	@echo "  report    Rebuild the rubric-ready DOCX report"
	@echo "  pipeline  Rebuild features, models, rubric artifacts, and report"

install:
	$(PYTHON) -m pip install -r requirements.txt

collect:
	$(PYTHON) -m src.data_collection.collect_all

live:
	$(PYTHON) -m src.data_collection.collect_2026_live
	$(PYTHON) -m src.features.harmonize_2026_live

features:
	$(PYTHON) -m src.features.build_dimensions
	$(PYTHON) -m src.features.build_backbone
	$(PYTHON) -m src.features.build_players
	$(PYTHON) -m src.features.build_elo
	$(PYTHON) -m src.features.build_matchup_features
	$(PYTHON) -m src.features.build_archetypes

models:
	$(PYTHON) -m src.models.cluster_archetypes
	$(PYTHON) -m src.models.predict_outcomes
	$(PYTHON) -m src.models.augment_and_improve
	$(PYTHON) -m src.models.validate_2026

rubric:
	$(PYTHON) -m src.models.build_rubric_artifacts

report:
	$(PYTHON) tools/build_rubric_ready_report_docx.py

pipeline: features models rubric report
