"""Data acquisition: USGS 3DEP elevation and the OpenStreetMap road network.

Both downloads are cached under the configured cache directory and keyed by
bounding box, so reruns do not refetch. Delete the cache directory to force a
refresh, or pass ``--refresh``.

Run directly to fetch (and report on) both datasets::

    python fetch_data.py
    python fetch_data.py --check     # only probe that the sources respond
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from config import Config, configure_logging, load_config

LOG = logging.getLogger("fetch")

USER_AGENT = "slo-terrain-matcher/1.0 (local training-route analysis)"

# GDAL needs these set before rasterio opens a remote GeoTIFF, otherwise it
# tries to list the whole S3 prefix on every open.
_GDAL_HTTP_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "100000000",
}


@dataclass(frozen=True)
class DemTile:
    """One 3DEP product returned by the TNM Access API."""

    title: str
    url: str
    size_bytes: int
    published: str

    @property
    def cell(self) -> str:
        """The one-degree cell name, e.g. ``n36w121``, parsed from the title."""
        for token in self.title.replace("_", " ").split():
            lowered = token.lower()
            if (
                len(lowered) == 7
                and lowered[0] in "ns"
                and lowered[3] in "ew"
                and lowered[1:3].isdigit()
                and lowered[4:7].isdigit()
            ):
                return lowered
        return self.title


def _bbox_key(bbox: tuple[float, float, float, float]) -> str:
    """A short stable cache key for a bounding box."""
    text = ",".join(f"{value:.5f}" for value in bbox)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Elevation
# ---------------------------------------------------------------------------


def query_dem_tiles(config: Config) -> list[DemTile]:
    """Ask the TNM Access API which 3DEP tiles cover the configured bbox.

    Args:
        config: Loaded configuration.

    Returns:
        One tile per one-degree cell, keeping the most recently published
        product for each cell.

    Raises:
        RuntimeError: If the API responds but lists no usable GeoTIFF.
        requests.HTTPError: If the API call itself fails.
    """
    west, south, east, north = config.bbox
    params = {
        "datasets": config.dem["tnm_dataset"],
        "bbox": f"{west},{south},{east},{north}",
        "prodFormats": "GeoTIFF",
        "max": 100,
    }
    LOG.info("querying TNM Access API for 3DEP tiles over %s", config.bbox)
    response = requests.get(
        config.dem["tnm_api"],
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()

    tiles: list[DemTile] = []
    for item in payload.get("items", []):
        url = item.get("downloadURL") or ""
        if not url.lower().endswith(".tif"):
            continue
        tiles.append(
            DemTile(
                title=item.get("title", url.rsplit("/", 1)[-1]),
                url=url,
                size_bytes=int(item.get("sizeInBytes") or 0),
                published=str(
                    item.get("publicationDate") or item.get("dateCreated") or ""
                ),
            )
        )
    if not tiles:
        raise RuntimeError(
            "TNM Access API returned no GeoTIFF products for this bbox; "
            f"total reported={payload.get('total')!r}"
        )

    # One product per one-degree cell: prefer the newest publication date,
    # falling back to the date embedded in the filename.
    best: dict[str, DemTile] = {}
    for tile in tiles:
        key = tile.cell
        current = best.get(key)
        if current is None or _tile_sort_key(tile) > _tile_sort_key(current):
            best[key] = tile

    chosen = sorted(best.values(), key=lambda tile: tile.cell)
    for tile in chosen:
        LOG.info(
            "  cell %s -> %s (%.0f MB)",
            tile.cell,
            tile.title,
            tile.size_bytes / 1e6,
        )
    return chosen


def _tile_sort_key(tile: DemTile) -> str:
    """Sortable recency key for a DEM product."""
    if tile.published:
        return tile.published
    digits = "".join(ch for ch in tile.url.rsplit("/", 1)[-1] if ch.isdigit())
    return digits[-8:] if len(digits) >= 8 else digits


def fetch_dem(config: Config, refresh: bool = False) -> Path:
    """Build a single DEM GeoTIFF clipped to the configured bounding box.

    By default this reads only the bounding box out of the remote tiles using
    HTTP range requests, which takes seconds instead of downloading several
    hundred megabytes per one-degree tile. Set ``dem.windowed_download: false``
    in the config to download whole tiles instead.

    Args:
        config: Loaded configuration.
        refresh: Re-download even if a cached clip exists.

    Returns:
        Path to the cached clipped GeoTIFF.

    Raises:
        RuntimeError: If no tile overlaps the bounding box.
    """
    import rasterio
    from rasterio.merge import merge

    cache_dir = config.cache_dir / "dem"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"dem_{_bbox_key(config.bbox)}.tif"
    if out_path.exists() and not refresh:
        LOG.info("using cached DEM clip %s", out_path.name)
        return out_path

    tiles = query_dem_tiles(config)
    windowed = bool(config.dem.get("windowed_download", True))

    sources: list[str] = []
    if windowed:
        sources = [f"/vsicurl/{tile.url}" for tile in tiles]
    else:
        for tile in tiles:
            sources.append(str(_download_file(tile.url, cache_dir, refresh)))

    west, south, east, north = config.bbox
    started = time.time()
    LOG.info(
        "reading DEM window from %d tile(s)%s",
        len(sources),
        " over HTTP" if windowed else "",
    )

    with rasterio.Env(**_GDAL_HTTP_ENV):
        handles = []
        try:
            for source in sources:
                handles.append(rasterio.open(source))
            mosaic, transform = merge(
                handles, bounds=(west, south, east, north)
            )
            profile = handles[0].profile.copy()
            crs = handles[0].crs
        finally:
            for handle in handles:
                handle.close()

    if mosaic.size == 0:
        raise RuntimeError("DEM mosaic is empty; does the bbox overlap a tile?")

    profile.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        count=1,
        transform=transform,
        crs=crs,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    tmp_path = out_path.with_suffix(".tif.part")
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(mosaic[0], 1)
    tmp_path.replace(out_path)

    LOG.info(
        "DEM clip written: %s  %dx%d px, CRS %s, %.1f MB, %.1fs",
        out_path.name,
        mosaic.shape[2],
        mosaic.shape[1],
        crs,
        out_path.stat().st_size / 1e6,
        time.time() - started,
    )
    return out_path


def _download_file(url: str, directory: Path, refresh: bool) -> Path:
    """Stream a URL to a file in ``directory``, with progress logging."""
    target = directory / url.rsplit("/", 1)[-1]
    if target.exists() and not refresh:
        LOG.info("using cached download %s", target.name)
        return target

    LOG.info("downloading %s", url)
    tmp_path = target.with_suffix(target.suffix + ".part")
    with requests.get(
        url, stream=True, headers={"User-Agent": USER_AGENT}, timeout=600
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length") or 0)
        written = 0
        next_report = 10
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
                written += len(chunk)
                if total and written * 100 // total >= next_report:
                    LOG.info("  %s: %d%%", target.name, next_report)
                    next_report += 10
    tmp_path.replace(target)
    LOG.info("  saved %s (%.0f MB)", target.name, target.stat().st_size / 1e6)
    return target


# ---------------------------------------------------------------------------
# Road network
# ---------------------------------------------------------------------------


def build_overpass_query(config: Config) -> str:
    """Compose the Overpass QL query for the configured area and tags."""
    west, south, east, north = config.bbox
    tags = "|".join(config.osm["highway_tags"])
    timeout = int(config.osm.get("timeout_s", 180))
    return (
        f"[out:json][timeout:{timeout}];\n"
        f'way["highway"~"^({tags})$"]'
        f"({south},{west},{north},{east});\n"
        "out body geom;\n"
    )


def fetch_osm(config: Config, refresh: bool = False) -> dict[str, Any]:
    """Fetch the road and path network for the configured bbox.

    Tries each configured Overpass mirror in turn with exponential backoff.
    Public Overpass instances routinely return 429/504 under load, so this is
    expected to retry. If every mirror fails, falls back to ``osmnx``, which
    maintains its own endpoint list.

    Args:
        config: Loaded configuration.
        refresh: Re-download even if a cached response exists.

    Returns:
        The Overpass JSON payload, with ``elements`` holding the ways. Each way
        carries ``tags``, ``nodes`` (node ids) and ``geometry``
        (``{"lat", "lon"}`` per node).

    Raises:
        RuntimeError: If every mirror and the fallback fail.
    """
    cache_dir = config.cache_dir / "osm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"ways_{_bbox_key(config.bbox)}.json.gz"

    if cache_path.exists() and not refresh:
        LOG.info("using cached OSM extract %s", cache_path.name)
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    query = build_overpass_query(config)
    endpoints = list(config.osm["overpass_endpoints"])
    max_retries = int(config.osm.get("max_retries", 4))
    timeout = int(config.osm.get("timeout_s", 180)) + 60

    last_error: str | None = None
    for endpoint in endpoints:
        for attempt in range(max_retries):
            delay = min(60.0, 4.0 * (2**attempt))
            try:
                LOG.info(
                    "Overpass request to %s (attempt %d/%d)",
                    endpoint,
                    attempt + 1,
                    max_retries,
                )
                started = time.time()
                response = requests.post(
                    endpoint,
                    data={"data": query},
                    headers={"User-Agent": USER_AGENT},
                    timeout=timeout,
                )
                if response.status_code == 200:
                    payload = response.json()
                    count = len(payload.get("elements", []))
                    LOG.info(
                        "  got %d ways in %.1fs", count, time.time() - started
                    )
                    if count == 0:
                        last_error = "Overpass returned zero ways"
                        break
                    _write_json_gz(cache_path, payload)
                    return payload
                last_error = f"HTTP {response.status_code} from {endpoint}"
                LOG.warning("  %s; retrying in %.0fs", last_error, delay)
            except (requests.RequestException, ValueError) as error:
                last_error = f"{type(error).__name__}: {error}"
                LOG.warning("  %s; retrying in %.0fs", last_error, delay)
            time.sleep(delay)

    LOG.warning("all Overpass mirrors failed (%s); trying osmnx", last_error)
    payload = _fetch_osm_via_osmnx(config)
    _write_json_gz(cache_path, payload)
    return payload


def fetch_waterways(
    config: Config, refresh: bool = False
) -> dict[str, Any] | None:
    """Fetch mapped waterways so creek crossings can be flagged.

    The DEM is bare earth, so a road crossing a creek on a culvert or a small
    unmapped bridge can read as a notch in the profile. Knowing where the
    creeks are lets the analysis mark those segments as unreliable.

    Args:
        config: Loaded configuration.
        refresh: Re-download even if a cached response exists.

    Returns:
        The Overpass payload, or ``None`` if every mirror failed. A failure
        here is not fatal; it only costs the waterway-crossing flag.
    """
    cache_dir = config.cache_dir / "osm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"waterways_{_bbox_key(config.bbox)}.json.gz"

    if cache_path.exists() and not refresh:
        LOG.info("using cached waterways %s", cache_path.name)
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    west, south, east, north = config.bbox
    timeout = int(config.osm.get("timeout_s", 180))
    query = (
        f"[out:json][timeout:{timeout}];\n"
        'way["waterway"~"^(river|stream|canal|ditch|drain)$"]'
        f"({south},{west},{north},{east});\n"
        "out geom;\n"
    )

    for endpoint in config.osm["overpass_endpoints"]:
        try:
            LOG.info("fetching waterways from %s", endpoint)
            response = requests.post(
                endpoint,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
                timeout=timeout + 60,
            )
            if response.status_code == 200:
                payload = response.json()
                LOG.info(
                    "  got %d waterway ways", len(payload.get("elements", []))
                )
                _write_json_gz(cache_path, payload)
                return payload
            LOG.warning("  HTTP %d", response.status_code)
        except (requests.RequestException, ValueError) as error:
            LOG.warning("  %s: %s", type(error).__name__, error)
        time.sleep(4.0)

    LOG.warning("waterway fetch failed; crossings will not be flagged")
    return None


def _write_json_gz(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON gzipped, atomically."""
    tmp_path = path.with_suffix(path.suffix + ".part")
    with gzip.open(tmp_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    tmp_path.replace(path)
    LOG.info("cached OSM extract to %s (%.1f MB)", path.name, path.stat().st_size / 1e6)


def _fetch_osm_via_osmnx(config: Config) -> dict[str, Any]:
    """Fallback network fetch through ``osmnx``, shaped like Overpass JSON.

    Raises:
        RuntimeError: If osmnx is unavailable or returns nothing.
    """
    try:
        import osmnx as ox
    except ImportError as error:  # pragma: no cover - depends on install
        raise RuntimeError(
            "Overpass mirrors failed and osmnx is not installed"
        ) from error

    west, south, east, north = config.bbox
    tags = "|".join(config.osm["highway_tags"])
    ox.settings.use_cache = True
    ox.settings.requests_timeout = int(config.osm.get("timeout_s", 180))
    try:
        graph = ox.graph_from_bbox(
            bbox=(west, south, east, north),
            custom_filter=f'["highway"~"^({tags})$"]',
            simplify=False,
            retain_all=True,
            truncate_by_edge=True,
        )
    except Exception as error:  # pragma: no cover - network dependent
        raise RuntimeError(f"osmnx fallback failed: {error}") from error

    nodes = graph.nodes(data=True)
    ways: dict[int, dict[str, Any]] = {}
    for u, v, data in graph.edges(data=True):
        osmid = data.get("osmid")
        if isinstance(osmid, list):
            osmid = osmid[0]
        way = ways.setdefault(
            int(osmid),
            {
                "type": "way",
                "id": int(osmid),
                "tags": {
                    key: value
                    for key, value in data.items()
                    if key
                    in {
                        "highway",
                        "surface",
                        "name",
                        "bridge",
                        "tunnel",
                        "access",
                        "oneway",
                    }
                    and isinstance(value, str)
                },
                "nodes": [],
                "geometry": [],
            },
        )
        for node in (u, v):
            if way["nodes"] and way["nodes"][-1] == node:
                continue
            way["nodes"].append(int(node))
            attrs = nodes[node]
            way["geometry"].append(
                {"lat": float(attrs["y"]), "lon": float(attrs["x"])}
            )

    if not ways:
        raise RuntimeError("osmnx fallback produced no ways")
    LOG.info("osmnx fallback produced %d ways", len(ways))
    return {"elements": list(ways.values()), "generator": "osmnx-fallback"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def check_sources(config: Config) -> bool:
    """Probe both data sources without downloading bulk data.

    Returns:
        True when both the TNM Access API and an Overpass mirror respond.
    """
    ok = True
    try:
        tiles = query_dem_tiles(config)
        LOG.info("TNM Access API OK: %d tile(s) selected", len(tiles))
    except Exception as error:
        LOG.error("TNM Access API unreachable: %s", error)
        ok = False

    probe = (
        "[out:json][timeout:60];"
        f"way[highway]({config.bbox[1]},{config.bbox[0]},"
        f"{config.bbox[1] + 0.01},{config.bbox[0] + 0.01});out count;"
    )
    reachable = False
    for endpoint in config.osm["overpass_endpoints"]:
        try:
            response = requests.post(
                endpoint,
                data={"data": probe},
                headers={"User-Agent": USER_AGENT},
                timeout=120,
            )
            if response.status_code == 200:
                LOG.info("Overpass OK: %s", endpoint)
                reachable = True
                break
            LOG.warning("Overpass %s -> HTTP %d", endpoint, response.status_code)
        except requests.RequestException as error:
            LOG.warning("Overpass %s -> %s", endpoint, error)
    if not reachable:
        LOG.error("no Overpass mirror responded")
        ok = False
    return ok


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for the data acquisition step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--refresh", action="store_true", help="ignore cached downloads"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only probe that the data sources respond, then exit",
    )
    args = parser.parse_args(argv)

    configure_logging()
    config = load_config(args.config)

    if args.check:
        return 0 if check_sources(config) else 1

    dem_path = fetch_dem(config, refresh=args.refresh)
    payload = fetch_osm(config, refresh=args.refresh)
    waterways = fetch_waterways(config, refresh=args.refresh)
    LOG.info(
        "ready: DEM %s, %d OSM ways, %d waterways",
        dem_path.name,
        len(payload.get("elements", [])),
        len((waterways or {}).get("elements", [])),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
