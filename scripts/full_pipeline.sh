#!/usr/bin/env bash
# FULL-PERIOD pipeline: the 12-instrument universe over its entire history (scan from 1987).
#
#   ingest -> baseline scan -> tune GA (Ray Tune) + detector (Optuna) -> adopt only validated
#   improvements -> tuned scan -> validate both -> compare -> short-scale scan -> combined model
#
# Heavy steps (scans, validation, combined model) run on the HOST with all cores (`lppl` from
# .venv); the tuning studies and adoption run in the `lppl-tune` image (Ray Tune, MLflow).
# Usage:  make pipeline-full   (or: PIPE_ID=20261122 bash scripts/full_pipeline.sh [from-step])
# Steps:  ingest base tune adopt tuned validate compare short combine
set -euo pipefail
cd "$(dirname "$0")/.."

PIPE_ID=${PIPE_ID:-$(date +%Y%m%d)}
TRIALS=${TRIALS:-40}
JOBS=${JOBS:-2}
ITRIALS=${ITRIALS:-300}
CFG=config/full.yaml
TCFG=config/full_tuning.yaml
VCFG=config/full_validation.yaml
TUNED=config/full_tuned.yaml
SHORT=config/full_short.yaml
GA=ga-full-$PIPE_ID
IND=indicator-full-$PIPE_ID
TUNE="docker compose --profile tune run --rm tune"
TUNE_LPPL="docker compose --profile tune run --rm --entrypoint lppl tune"
LPPL=${LPPL:-.venv/bin/lppl}

set -a; [ -f .env ] && . ./.env; set +a
export LPPL_GIT_SHA=$(git rev-parse --verify -q HEAD || true)

STEPS=(ingest base tune adopt tuned validate compare short combine)
FROM=${1:-ingest}
run=0
stamp() { echo "=== [$(date '+%F %T')] $*"; }

for step in "${STEPS[@]}"; do
  [ "$step" = "$FROM" ] && run=1
  [ $run -eq 1 ] || continue
  stamp "step $step"
  case $step in
    ingest)   $LPPL ingest -c $CFG ;;
    base)     $LPPL scan -c $CFG --label full-base ;;
    tune)     $TUNE build-benchmark --force -c $CFG --tuning-config $TCFG
              $TUNE run ga --backend ray --trials "$TRIALS" --jobs "$JOBS" --name "$GA" -c $CFG --tuning-config $TCFG
              $TUNE run indicator --backend optuna --trials "$ITRIALS" --jobs 1 --name "$IND" -c $CFG --tuning-config $TCFG ;;
    adopt)    $TUNE adopt --ga-study "$GA" --indicator-study "$IND" -c $CFG --tuning-config $TCFG --out $TUNED ;;
    tuned)    $LPPL scan -c $TUNED --label full-tuned ;;
    validate) $LPPL validate --label full-base --validation-config $VCFG --tuning-config $TCFG --no-mlflow
              $LPPL validate --label full-tuned --validation-config $VCFG --tuning-config $TCFG --no-mlflow ;;
    compare)  $LPPL compare-reports "$(ls -td reports/validation/full-base_* | head -1)" \
                "$(ls -td reports/validation/full-tuned_* | head -1)" --out "reports/full_$PIPE_ID.md" ;;
    short)    .venv/bin/python - "$TUNED" "$SHORT" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg["windows"] = {"min_len": 40, "max_len": 180, "step": 20}
open(sys.argv[2], "w").write("# SHORT-SCALE scan of the full-period pipeline: config/full_tuned.yaml with 8 windows of\n"
                             "# 40..180 days (written by scripts/full_pipeline.sh).\n" + yaml.safe_dump(cfg, sort_keys=False))
EOF
              $LPPL scan -c $SHORT --label full-short ;;
    combine)  $LPPL combine --label full-tuned --short-label full-short --first-test-year 1995 ;;
  esac
done
stamp "done (pipeline id $PIPE_ID)"
