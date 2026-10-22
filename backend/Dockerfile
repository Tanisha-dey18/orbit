# PROJECT ORBIT backend (FastAPI + PyTorch CPU)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/backend
COPY backend/requirements.txt backend/requirements-geo.txt ./
# CPU-only PyTorch keeps the image ~1 GB smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt -r requirements-geo.txt

COPY backend/ ./
COPY data/samples/ /srv/data/samples/

# Writable runtime state lives under /srv/artifacts - mount a volume there for persistence.
ENV ORBIT_ARTIFACT_DIR=/srv/artifacts \
    ORBIT_DATABASE_URL=sqlite:////srv/artifacts/orbit.db \
    ORBIT_MODEL_PATH=/srv/data/checkpoints/orbit_fc_siam_diff.pt \
    ORBIT_SAMPLES_DIR=/srv/data/samples
RUN useradd --create-home orbit && mkdir -p /srv/artifacts /srv/data/checkpoints && chown -R orbit /srv
USER orbit

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
