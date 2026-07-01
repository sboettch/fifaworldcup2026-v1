PYTHON ?= python3

.PHONY: help install collect live features models validate pipeline

help:
	@echo "FIFA World Cup 2026 — Matchup Archetype Framework"
	@echo ""
	@echo "Targets:"
	@echo "  install   Install Python dependencies"
	@echo "  collect   Fetch raw data (matches, squads, Elo inputs)"
	@echo "  live      Collect and harmonize latest 2026 live snapshot"
	@echo "  features  Build processed tables (dimensions, Elo, matchup features, archetypes)"
	@echo "  models    Run clustering, supervised models, augmentation experiments"
	@echo "  validate  Run 2026 out-of-sample validation"
	@echo "  pipeline  Full pipeline: features → models → validate"
	@echo ""
	@echo "Quick start:"
	@echo "  make install && make collect && make pipeline"

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
	$(PYTHON) -m src.features.contextual_features

models:
	$(PYTHON) -m src.models.cluster_archetypes
	$(PYTHON) -m src.models.predict_outcomes

validate:
	$(PYTHON) -m src.models.validate_2026

pipeline: features models validate
