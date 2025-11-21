# Multi-stage Dockerfile für Crypto Trading Bot

# ====== BUILD STAGE ======
FROM python:3.10-slim as builder

# Installiere Build-Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Erstelle Virtual Environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Installiere Python Dependencies
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-cloud.txt

# ====== RUNTIME STAGE ======
FROM python:3.10-slim as runtime

# Umgebungsvariablen
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Kopiere Virtual Environment
COPY --from=builder /opt/venv /opt/venv

# Erstelle Non-Root User
RUN groupadd -r tradingbot && useradd -r -g tradingbot tradingbot

# Arbeitsverzeichnis
WORKDIR /app

# Kopiere Anwendungscode
COPY --chown=tradingbot:tradingbot . .

# Erstelle notwendige Verzeichnisse
RUN mkdir -p /app/data /app/logs /app/models && \
    chown -R tradingbot:tradingbot /app

# Wechsle zu Non-Root User
USER tradingbot

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default Command
CMD ["python", "paper_trade_cloud.py"]

# ====== LABELS ======
LABEL maintainer="Trading Bot Team" \
      version="1.0" \
      description="AI-powered Crypto Trading Bot"
