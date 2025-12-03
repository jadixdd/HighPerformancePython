FROM python:3.12-slim-bookworm AS clone

ARG REPO_URL=https://github.com/jadixdd/HighPerformancePython.git
ARG BRANCH=main

RUN echo "Клонируем репозиторий:" && \
    echo "URL: '${REPO_URL}'" && \
    echo "Branch: '${BRANCH}'" && \
    echo "Длина URL: $(echo -n "$REPO_URL" | wc -c)"

RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates && \
    CLEAN_URL=$(echo "$REPO_URL" | tr -d '[:space:]') && \
    echo "Очищенный URL: '$CLEAN_URL'" && \
    git ls-remote --exit-code "$CLEAN_URL" "$BRANCH" && \
    git clone --depth 1 --branch "$BRANCH" "$CLEAN_URL" /repo && \
    ls -la /repo && \
    rm -rf /repo/.git && \
    apt-get purge -y --auto-remove git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

FROM python:3.12-slim-bookworm AS builder

COPY --from=clone /repo/Lab2Py /app
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv .venv && \
    . .venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

RUN . .venv/bin/activate && \
    pip install --no-cache-dir ./cpp_solver

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libatlas-base-dev \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "main.py"]