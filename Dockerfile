# TwinSync — one command to a running twin.
#
#     docker compose up      →  http://localhost:8000
#
# The v0.1 prototype deliberately has no database or broker to stand up: the world is
# baked into GeoJSON artifacts, runtime state is in-process, and both model artifacts
# are committed. That is what makes this image a single stage with no services behind
# it — and it is why the container needs no network at run time, only at build.

FROM python:3.12-slim

# libgomp is LightGBM's OpenMP runtime; onnxruntime wants libstdc++. Both come from
# the slim image's own repos, and nothing else is needed — there is no GDAL here,
# because rasterio is a build-time dependency for baking the DEM and the NDVI scene,
# not a runtime one.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing source does not re-resolve the wheel set.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY twinsync/ twinsync/
COPY edge/ edge/
COPY data/ data/
COPY models/ models/
COPY web/ web/

# Coverage is ray-cast once and cached against a fingerprint of the scene
# (twinsync/coverage.py). data/coverage_cache.json is committed and copied in above, so
# the container starts in seconds rather than spending half a minute on startup.
ENV PYTHONUNBUFFERED=1 \
    TWINSYNC_DATA=/app/data

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/models > /dev/null || exit 1

CMD ["python", "-m", "uvicorn", "twinsync.server:app", \
     "--host", "0.0.0.0", "--port", "8000"]
