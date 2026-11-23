#!/usr/bin/env bash
# ROBUST VARIANT experiment (docs/validation.md §8): scan config/full_robust.yaml, validate it with
# the full-period settings, compare with the latest full-tuned report, run the combined model and
# the pre-registered paired comparison. Runs on the host venv (all cores). Steps can be resumed:
#   bash scripts/robust_variant.sh [scan|validate|compare|combine|analyse]
set -euo pipefail
cd "$(dirname "$0")/.."
LPPL=${LPPL:-.venv/bin/lppl}
set -a; [ -f .env ] && . ./.env; set +a
export LPPL_GIT_SHA=$(git rev-parse --verify -q HEAD || true)
STEPS=(scan validate compare combine analyse)
FROM=${1:-scan}; run=0
stamp() { echo "=== [$(date '+%F %T')] $*"; }
for step in "${STEPS[@]}"; do
  [ "$step" = "$FROM" ] && run=1
  [ $run -eq 1 ] || continue
  stamp "step $step"
  case $step in
    scan)     $LPPL scan -c config/full_robust.yaml --label full-robust ;;
    validate) $LPPL validate --label full-robust --validation-config config/full_validation.yaml \
                --tuning-config config/full_tuning.yaml --no-mlflow ;;
    compare)  $LPPL compare-reports "$(ls -td reports/validation/full-tuned_* | head -1)" \
                "$(ls -td reports/validation/full-robust_* | head -1)" --out reports/robust_variant.md ;;
    combine)  $LPPL combine --label full-robust --short-label full-short --first-test-year 1995 ;;
    analyse)  .venv/bin/python scripts/robust_compare.py --base full-tuned --variant full-robust ;;
  esac
done
stamp "done"
