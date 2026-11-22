# Convenience targets. Run `make help`.
SHELL := /bin/bash
COMPOSE := docker compose
RUN := $(COMPOSE) run --rm app
CONFIG ?= config/default.yaml
PY ?= .venv/bin/python
export LPPL_GIT_SHA := $(shell git rev-parse --verify -q HEAD 2>/dev/null)

.PHONY: help up down clean-db build lock ingest ingest-snapshot scan scan-quick case-studies \
        report notebooks jupyter test test-integration test-docker lint reproduce verify \
        tune-build tune-up tune-down tune-benchmark tune-ga tune-ga-ray tune-indicator \
        tune-indicator-ray tune-lock validate validate-quick bubbles bubbles-ingest bubbles-scan bubbles-short bubbles-validate combine combine-pipeline pipeline-full pipeline pipeline-ingest pipeline-scan-base \
        pipeline-tune pipeline-adopt pipeline-scan-tuned pipeline-validate pipeline-compare

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n",$$1,$$2}'

up:  ## start PostgreSQL (waits until healthy)
	$(COMPOSE) up -d --wait db

down:  ## stop containers (database volume kept)
	$(COMPOSE) down

clean-db:  ## stop containers AND delete the database volume
	$(COMPOSE) down -v

build:  ## build the app image
	$(COMPOSE) build app

lock:  ## regenerate requirements.lock inside python:3.12 (linux) for reproducible builds
	docker run --rm -v "$$PWD":/src -w /src python:3.12.8-slim-bookworm bash -c \
	  "pip install -q '.[notebooks,dev]' && pip freeze --exclude lppl > requirements.lock"

ingest: up  ## download Yahoo data -> Postgres (+ data/snapshots/*.csv)
	$(RUN) ingest -c $(CONFIG)

ingest-snapshot: up  ## rebuild Postgres from data/snapshots (offline, bit-exact)
	$(RUN) ingest -c $(CONFIG) --from-snapshot

scan: up  ## full scan with CONFIG (default: analysis dates 2019 -> today)
	$(RUN) scan -c $(CONFIG) --label default

scan-quick: up  ## quick scan with config/quick.yaml (minutes)
	$(RUN) scan -c config/quick.yaml --label quick

case-studies: up  ## historical episodes used in docs/model_presentation.md
	$(RUN) scan -t ^IXIC --start 1999-01-01 --end 2000-12-31 --step 5 --label case:dotcom
	$(RUN) scan -t ^GSPC --start 2007-01-01 --end 2008-12-31 --step 5 --label case:gfc
	$(RUN) scan -t TSLA --start 2020-06-01 --end 2021-12-31 --step 5 --label case:tsla2021
	$(RUN) scan -t NVDA --start 2023-06-01 --end 2024-12-31 --step 5 --label case:nvda2024

report: up  ## latest indicator per ticker (also reports/latest.csv)
	$(RUN) report --csv reports/latest.csv

notebooks: up  ## execute all notebooks in place against the database
	$(COMPOSE) run --rm --entrypoint "" app bash -c \
	  'for nb in notebooks/0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$$nb" || exit 1; done'

jupyter: up  ## JupyterLab on http://localhost:8888 (token: JUPYTER_TOKEN in .env)
	$(COMPOSE) --profile notebooks up -d jupyter

test:  ## unit tests on the host venv (no DB / network needed)
	$(PY) -m pytest --cov=lppl --cov-report=term-missing

test-integration: up  ## integration tests against the compose database
	$(PY) -m pytest -m integration

test-docker: up  ## unit + integration tests inside the container
	$(COMPOSE) run --rm --entrypoint "" app python -m pytest -m "not network"

lint:  ## ruff
	$(PY) -m ruff check src tests

verify: up  ## compare the latest results with reports/reference.csv
	$(RUN) verify --reference reports/reference.csv

reproduce: build ingest-snapshot scan case-studies report notebooks verify  ## full rebuild from snapshots

# --------------------------------------------------------------------------- tuning
TUNE := $(COMPOSE) --profile tune run --rm tune
TRIALS ?= 40
JOBS ?= 2

tune-build: build  ## build the tuning image (app image + Ray Tune, Optuna, MLflow)
	$(COMPOSE) --profile tune build mlflow

tune-lock:  ## regenerate requirements-tune.lock (packages on top of requirements.lock)
	docker run --rm -v "$$PWD":/src -w /src python:3.12.8-slim-bookworm bash -c \
	  "pip install -q -r requirements.lock 'ray[tune]==2.60.0' 'optuna==5.1.0' 'mlflow==3.18.0' \
	   'optuna-dashboard==0.21.0' && pip freeze --exclude lppl | sort > /tmp/all.txt && \
	   sort requirements.lock > /tmp/base.txt && comm -23 /tmp/all.txt /tmp/base.txt > requirements-tune.lock"

tune-up: up  ## start MLflow (http://localhost:5050) and optuna-dashboard (http://localhost:8081)
	$(COMPOSE) --profile tune up -d --wait mlflow optuna-dashboard

tune-down:  ## stop the tuning services (MLflow data stays in Postgres + volume)
	$(COMPOSE) --profile tune stop mlflow optuna-dashboard

tune-benchmark: tune-up  ## build the GA benchmark + oracle references (cached in data/tuning)
	$(TUNE) build-benchmark

tune-ga: tune-benchmark  ## GA study with Optuna (TRIALS, JOBS)
	$(TUNE) run ga --backend optuna --trials $(TRIALS) --jobs $(JOBS)

tune-ga-ray: tune-benchmark  ## GA study with Ray Tune + ASHA (TRIALS, JOBS)
	$(TUNE) run ga --backend ray --trials $(TRIALS) --jobs $(JOBS)

tune-indicator: tune-up  ## detector study with Optuna on the stored default scan
	$(TUNE) run indicator --backend optuna --trials $(or $(ITRIALS),300) --jobs 1

tune-indicator-ray: tune-up  ## detector study with Ray Tune
	$(TUNE) run indicator --backend ray --trials $(or $(ITRIALS),300) --jobs $(JOBS)

# --------------------------------------------------------------------------- validation
VALIDATE := $(COMPOSE) --profile tune run --rm --entrypoint lppl tune validate

validate: tune-up  ## full validation of the latest `default` scan (null model, walk-forward, MLflow)
	$(VALIDATE) --label $(or $(LABEL),default)

validate-quick: up  ## validation without null-model rescans (seconds; app image, no MLflow)
	$(RUN) validate --label $(or $(LABEL),default) --null-paths 0 --no-mlflow

# --------------------------------------------------------------------------- full pipeline
# ingest -> baseline scan -> tune GA (Ray Tune) + detector (Optuna) -> adopt only validated
# improvements -> tuned scan -> validate both -> compare. ~3 h on 2 CPUs. Steps can be re-run
# individually; pass the same PIPE_ID to steps run on another day.
PIPE_ID ?= $(shell date +%Y%m%d)
P_CFG := config/pipeline.yaml
P_TCFG := config/pipeline_tuning.yaml
P_TUNED := config/pipeline_tuned.yaml
TUNE_LPPL := $(COMPOSE) --profile tune run --rm --entrypoint lppl tune
P_GA := ga-pipeline-$(PIPE_ID)
P_IND := indicator-pipeline-$(PIPE_ID)

pipeline: tune-build tune-up pipeline-ingest pipeline-scan-base pipeline-tune pipeline-adopt \
          pipeline-scan-tuned pipeline-validate pipeline-compare  ## full pipeline with tuning + validation

pipeline-ingest:
	$(RUN) ingest -c $(P_CFG)

pipeline-scan-base:
	$(RUN) scan -c $(P_CFG) --label pipeline-base

pipeline-tune:
	$(TUNE) build-benchmark --force -c $(P_CFG) --tuning-config $(P_TCFG)
	$(TUNE) run ga --backend ray --trials $(TRIALS) --jobs $(JOBS) --name $(P_GA) -c $(P_CFG) --tuning-config $(P_TCFG)
	$(TUNE) run indicator --backend optuna --trials $(or $(ITRIALS),300) --jobs 1 --name $(P_IND) -c $(P_CFG) --tuning-config $(P_TCFG)

pipeline-adopt:
	$(TUNE) adopt --ga-study $(P_GA) --indicator-study $(P_IND) -c $(P_CFG) --tuning-config $(P_TCFG) --out $(P_TUNED)

pipeline-scan-tuned:
	$(RUN) scan -c $(P_TUNED) --label pipeline-tuned

pipeline-validate:
	$(TUNE_LPPL) validate --label pipeline-base --tuning-config $(P_TCFG)
	$(TUNE_LPPL) validate --label pipeline-tuned --tuning-config $(P_TCFG)

pipeline-compare:
	$(TUNE_LPPL) compare-reports $$(ls -td reports/validation/pipeline-base_* | head -1) \
	  $$(ls -td reports/validation/pipeline-tuned_* | head -1) --out reports/pipeline_$(PIPE_ID).md

# --------------------------------------------------------------------------- burst-bubble universe
# 16 assets with documented bubbles, history from 1985 (docs/validation.md §6). ~2 h scan + ~25 min validation.
bubbles: tune-build tune-up bubbles-ingest bubbles-scan bubbles-validate bubbles-short combine  ## burst-bubble universe: ingest, scans, validate, combined model

bubbles-ingest:
	$(RUN) ingest -c config/bubbles.yaml

bubbles-scan:
	$(RUN) scan -c config/bubbles.yaml --label bubbles

bubbles-validate:
	$(TUNE_LPPL) validate --label bubbles --tuning-config config/pipeline_tuning.yaml

bubbles-short:  ## short-scale scan (8 windows, 40..180 days) of the burst-bubble universe, ~40 min
	$(RUN) scan -c config/bubbles_short.yaml --label bubbles-short

# --------------------------------------------------------------------------- combined model
combine: up  ## refined LPPLS signals + walk-forward combined model (docs/combined_model.md)
	$(RUN) combine --label bubbles --short-label bubbles-short

combine-pipeline: up  ## the same on the 12-stock universe (needs a `pipeline-short` scan)
	$(RUN) scan -c config/pipeline_short.yaml --label pipeline-short
	$(RUN) combine --label pipeline-base --short-label pipeline-short --first-test-year 2021

# --------------------------------------------------------------------------- full-period pipeline
# The 12-instrument universe over its entire history (scan from 1987): ingest -> base scan -> tuning
# -> validated adoption -> tuned scan -> validation of both -> compare -> short scan -> combined model.
# Scans/validation run on the host (.venv, all cores); tuning in the lppl-tune image. ~6 h on 4 cores.
# Resume from a step with: PIPE_ID=<id> bash scripts/full_pipeline.sh <step>
pipeline-full: tune-build tune-up  ## the whole pipeline over the entire period (scripts/full_pipeline.sh)
	PIPE_ID=$(PIPE_ID) TRIALS=$(TRIALS) JOBS=$(JOBS) bash scripts/full_pipeline.sh
