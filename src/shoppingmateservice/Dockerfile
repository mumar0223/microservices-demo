FROM python:3.12.8-slim AS base

# ----------------------
# Builder stage
# ----------------------
FROM base AS builder

# Install build-time dependencies
RUN apt-get -qq update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip and install Python packages
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ----------------------
# Final stage
# ----------------------
FROM python:3.12.8-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime PostgreSQL client libraries
RUN apt-get -qq update \
    && apt-get install -y --no-install-recommends libpq-dev libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/ /usr/local/lib/python3.12/

# Copy app source code
COPY . .

EXPOSE 8080

ENTRYPOINT ["python", "main.py"]
