# syntax=docker/dockerfile:1
FROM python:3.12.8-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get clean

WORKDIR /app

# 1) Pinned third-party dependencies (layer cached unless the lock changes)
COPY requirements.lock ./
RUN pip install -r requirements.lock

# 2) The package itself, without re-resolving dependencies
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
RUN pip install --no-deps -e . \
    && python -m ipykernel install --sys-prefix --name lppl --display-name "Python (lppl)"

COPY tests ./tests
COPY notebooks ./notebooks

ENTRYPOINT ["lppl"]
CMD ["--help"]
