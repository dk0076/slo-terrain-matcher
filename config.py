"""Configuration loading for the SLO terrain matcher.

Everything tunable lives in ``config.yaml``; this module turns it into typed
objects so the rest of the pipeline does not pass raw dictionaries around.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

LOG = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

#: Metres per statute mile.
METRES_PER_MILE = 1609.344


@dataclass(frozen=True)
class Target:
    """One terrain feature from the race course that we want to reproduce.

    Attributes:
        key: Short machine-readable identifier, used in filenames and columns.
        label: Human-readable name for reports.
        race_location: Where the feature occurs on the race course.
        spec: The original written specification, echoed into the report.
        length_mi: ``(minimum, maximum)`` acceptable segment length in miles.
        grade_pct: ``(minimum, maximum)`` acceptable mean grade in percent.
            Negative values describe descents.
        ideal_grade_pct: The grade a perfect match would have.
        require_paved: Drop candidates that are not known to be paved.
        prefer_paved: Award the paved bonus in the quality score.
        want_loopable: Run the loopability check and reward segments that can
            be repeated without retracing the same road.
        color: Hex colour used for this feature on the Folium map.
    """

    key: str
    label: str
    race_location: str
    spec: str
    length_mi: tuple[float, float]
    grade_pct: tuple[float, float]
    ideal_grade_pct: float
    require_paved: bool = False
    prefer_paved: bool = True
    want_loopable: bool = False
    color: str = "#555555"

    @property
    def length_m(self) -> tuple[float, float]:
        """The acceptable length range in metres."""
        return (
            self.length_mi[0] * METRES_PER_MILE,
            self.length_mi[1] * METRES_PER_MILE,
        )

    @property
    def is_descent(self) -> bool:
        """True when this target describes a net downhill."""
        return self.grade_pct[1] <= 0.0

    def grade_matches(self, grade_pct: float) -> bool:
        """Whether a mean grade falls inside this target's grade band."""
        return self.grade_pct[0] <= grade_pct <= self.grade_pct[1]

    def length_matches(self, length_m: float) -> bool:
        """Whether a segment length falls inside this target's length band."""
        lo, hi = self.length_m
        return lo <= length_m <= hi


@dataclass(frozen=True)
class Config:
    """The whole configuration file, plus a few derived conveniences."""

    area: dict[str, Any]
    dem: dict[str, Any]
    osm: dict[str, Any]
    analysis: dict[str, Any]
    scoring: dict[str, Any]
    output: dict[str, Any]
    targets: list[Target]
    routes: list[dict[str, Any]] = field(default_factory=list)
    course_gpx: str | None = None
    root: Path = field(default_factory=lambda: DEFAULT_CONFIG_PATH.parent)

    # -- area ------------------------------------------------------------
    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Bounding box as ``(west, south, east, north)`` in EPSG:4326."""
        west, south, east, north = self.area["bbox"]
        return float(west), float(south), float(east), float(north)

    @property
    def reference_point(self) -> tuple[float, float]:
        """The ``(latitude, longitude)`` the proximity bonus measures from."""
        lat, lon = self.area["reference_point"]
        return float(lat), float(lon)

    # -- paths -----------------------------------------------------------
    @property
    def output_dir(self) -> Path:
        """Directory for generated reports, created on demand."""
        path = self._resolve(self.output.get("dir", "output"))
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cache_dir(self) -> Path:
        """Directory for cached downloads, created on demand."""
        path = self._resolve(self.output.get("cache_dir", "data"))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    # -- targets ---------------------------------------------------------
    def with_area(
        self,
        bbox: tuple[float, float, float, float],
        center: tuple[float, float] | None = None,
        name: str | None = None,
    ) -> "Config":
        """A copy of this config pointed at a different search area.

        Lets the command-line entry points retarget the pipeline without
        anyone hand-editing ``config.yaml``.

        Args:
            bbox: ``(west, south, east, north)`` in EPSG:4326.
            center: Optional ``(latitude, longitude)`` for the proximity
                bonus and as the route start point.
            name: Optional human-readable area name for reports.

        Returns:
            A new :class:`Config`; the original is unchanged.
        """
        area = dict(self.area)
        area["bbox"] = list(bbox)
        if center is not None:
            area["reference_point"] = [center[0], center[1]]
        if name is not None:
            area["name"] = name
        return replace(self, area=area)

    def with_course(self, course_gpx: str | Path) -> "Config":
        """A copy of this config pointed at a different race course GPX."""
        return replace(self, course_gpx=str(course_gpx))

    def with_routes(self, routes: list[dict[str, Any]]) -> "Config":
        """A copy of this config with a different set of route specs."""
        return replace(self, routes=list(routes))

    def target(self, key: str) -> Target:
        """Look up a single target by key.

        Raises:
            KeyError: If no target with that key is configured.
        """
        for target in self.targets:
            if target.key == key:
                return target
        raise KeyError(f"no target configured with key {key!r}")


def load_config(path: str | Path | None = None) -> Config:
    """Read and validate ``config.yaml``.

    Args:
        path: Path to the YAML file. Defaults to ``config.yaml`` beside this
            module.

    Returns:
        The parsed configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a required section or target field is missing, or a
            target's ranges are inverted.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    for section in ("area", "dem", "osm", "analysis", "scoring", "output"):
        if section not in raw:
            raise ValueError(f"config is missing the {section!r} section")
    if not raw.get("targets"):
        raise ValueError("config must define at least one target")

    targets = [_parse_target(entry) for entry in raw["targets"]]
    keys = [target.key for target in targets]
    duplicates = {key for key in keys if keys.count(key) > 1}
    if duplicates:
        raise ValueError(f"duplicate target keys: {sorted(duplicates)}")

    bbox = raw["area"].get("bbox")
    if not bbox or len(bbox) != 4:
        raise ValueError("area.bbox must be [west, south, east, north]")
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError(f"area.bbox is inverted or empty: {bbox}")

    return Config(
        area=raw["area"],
        dem=raw["dem"],
        osm=raw["osm"],
        analysis=raw["analysis"],
        scoring=raw["scoring"],
        output=raw["output"],
        targets=targets,
        routes=list(raw.get("routes") or []),
        course_gpx=raw.get("course_gpx"),
        root=config_path.resolve().parent,
    )


def _parse_target(entry: dict[str, Any]) -> Target:
    """Build a :class:`Target` from one YAML mapping."""
    for required in ("key", "length_mi", "grade_pct"):
        if required not in entry:
            raise ValueError(f"target is missing {required!r}: {entry}")

    length = tuple(float(value) for value in entry["length_mi"])
    grade = tuple(float(value) for value in entry["grade_pct"])
    if len(length) != 2 or length[0] > length[1]:
        raise ValueError(f"target {entry['key']!r} has a bad length_mi range")
    if len(grade) != 2 or grade[0] > grade[1]:
        raise ValueError(f"target {entry['key']!r} has a bad grade_pct range")

    ideal = float(entry.get("ideal_grade_pct", sum(grade) / 2.0))
    return Target(
        key=str(entry["key"]),
        label=str(entry.get("label", entry["key"])),
        race_location=str(entry.get("race_location", "")),
        spec=str(entry.get("spec", "")),
        length_mi=(length[0], length[1]),
        grade_pct=(grade[0], grade[1]),
        ideal_grade_pct=ideal,
        require_paved=bool(entry.get("require_paved", False)),
        prefer_paved=bool(entry.get("prefer_paved", True)),
        want_loopable=bool(entry.get("want_loopable", False)),
        color=str(entry.get("color", "#555555")),
    )


def configure_logging(level: int = logging.INFO) -> None:
    """Set up readable progress logging for the command-line entry points."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(name)-12s %(message)s",
        datefmt="%H:%M:%S",
    )
    # These are chatty and not useful at INFO.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    logging.getLogger("fiona").setLevel(logging.WARNING)
