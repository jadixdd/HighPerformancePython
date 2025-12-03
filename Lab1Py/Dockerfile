FROM python:3.12-slim-bookworm AS clone

ARG REPO_URL=https://github.com/jadixdd/HighPerformancePython.git
ARG BRANCH=main

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && \
    CLEAN_URL=$(echo "$REPO_URL" | tr -d '[:space:]') && \
    git clone --depth 1 --branch "$BRANCH" "$CLEAN_URL" /repo && \
    rm -rf /repo/.git && \
    apt-get purge -y --auto-remove git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

FROM python:3.12-slim-bookworm AS builder

COPY --from=clone /repo/Lab1Py /app
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libatlas-base-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1 \
    PYTHONUNBUFFERED=1

RUN pip install "poetry==$POETRY_VERSION" --no-cache-dir

RUN poetry install --no-root --only main --no-cache

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libatlas-base-dev \
    libxcb1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "main.py"]