# syntax=docker/dockerfile:1
# Tuning image: the app image plus Ray Tune, Optuna, MLflow and optuna-dashboard.
# Build the base first (`make build`); `make tune-build` does both.
ARG BASE_IMAGE=lppl-app:0.1.0
FROM ${BASE_IMAGE}

COPY requirements.lock requirements-tune.lock ./
# requirements-tune.lock pins only the packages added on top of requirements.lock
RUN pip install -r requirements.lock -r requirements-tune.lock

# Refresh the package source so code changes are picked up without rebuilding the base
COPY src ./src
COPY config ./config
COPY tests ./tests

ENV RAY_USAGE_STATS_ENABLED=0 \
    RAY_DEDUP_LOGS=1 \
    MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT=true \
    MLFLOW_TRACKING_URI=http://mlflow:5000

ENTRYPOINT ["lppl", "tune"]
CMD ["--help"]
