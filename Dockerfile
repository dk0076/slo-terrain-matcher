# Build:  docker build -t terrain-matcher .
# Run:    docker run -p 8000:8000 -v terrain-cache:/data terrain-matcher
#
# The volume matters. Every new search area downloads a DEM clip and an
# Overpass extract; without a persistent cache a restart re-fetches all of it,
# which is slow for you and rude to OpenStreetMap's volunteer servers.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TERRAIN_WORK_DIR=/data/web-work

WORKDIR /app

# rasterio and pyproj ship their own GDAL/PROJ in the wheels, so no system
# GDAL is needed. curl is only here for the health check.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

# Caches and uploads live on the volume, not in the image layer.
RUN mkdir -p /data/web-work /data/cache \
 && useradd --create-home --uid 10001 runner \
 && chown -R runner:runner /app /data
USER runner

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS http://127.0.0.1:8000/ > /dev/null || exit 1

# Two workers, two threads each. Route building is CPU-bound and holds a DEM
# in memory, so more workers mostly buys memory pressure.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "2", \
     "--timeout", "600", "--access-logfile", "-", "webapp:app"]
