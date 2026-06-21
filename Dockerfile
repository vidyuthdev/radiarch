# ---- Stage 1: Builder ----
FROM python:3.12-slim AS builder

WORKDIR /build


COPY src/pyproject.toml src/
RUN pip install --no-cache-dir --prefix=/install -e ./src[dev] 2>/dev/null || \
    pip install --no-cache-dir --prefix=/install ./src

# ---- Stage 2: Runtime ----
FROM python:3.12-slim

WORKDIR /app

# Runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./

# Ensure the vendored MCsquare Monte Carlo binaries are executable (COPY can
# drop the +x bit depending on the build context). The Celery worker invokes
# MCsquare_linux* for real proton dose; without +x it fails with EACCES.
RUN chmod +x /app/opentps/core/processing/doseCalculation/protons/MCsquare/MCsquare* \
    2>/dev/null || true

# Create data directories
RUN mkdir -p /data/artifacts /data/sessions

# Default environment (vendored OpenTPS Core available)
ENV RADIARCH_ENVIRONMENT=production \
    RADIARCH_FORCE_SYNTHETIC=false \
    RADIARCH_DATABASE_URL=postgresql+psycopg://radiarch:radiarch@postgres:5432/radiarch \
    RADIARCH_BROKER_URL=redis://redis:6379/0 \
    RADIARCH_RESULT_BACKEND=redis://redis:6379/1 \
    RADIARCH_ARTIFACT_DIR=/data/artifacts \
    RADIARCH_ORTHANC_BASE_URL=http://orthanc:8042 \
    RADIARCH_ORTHANC_USE_MOCK=false \
    RADIARCH_DICOMWEB_URL=http://orthanc:8042/dicom-web

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/info || exit 1

# Default: run the API server
CMD ["python", "-m", "uvicorn", "radiarch.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
