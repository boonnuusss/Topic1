"""Configuration helpers for the Topic1 ray-wise STEC pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import json
from pathlib import Path
from typing import Any


DEFAULT_REGION = (80.0, 140.0, 0.0, 60.0)


@dataclass(frozen=True)
class Region:
    """Longitude/latitude bounds in degrees."""

    lon_min: float = 80.0
    lon_max: float = 140.0
    lat_min: float = 0.0
    lat_max: float = 60.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.lon_min, self.lon_max, self.lat_min, self.lat_max)


@dataclass
class DatasetBuildConfig:
    """Runtime options for building ray-wise STEC datasets."""

    tec_data_dir: Path
    sp3_files: list[Path]
    output_dir: Path = Path("datasets/processed")
    interim_dir: Path = Path("datasets/interim")
    start_date: date = date(2024, 5, 7)
    end_date: date = date(2024, 5, 7)
    epoch_interval_seconds: int = 900
    select_window_seconds: int = 30
    sp3_tolerance_seconds: int = 900
    min_elevation_deg: float = 10.0
    stec_min: float = 0.0
    stec_max: float = 600.0
    region: Region = field(default_factory=Region)
    stations: list[str] | None = None
    station_glob: str = "*.mat"
    output_format: str = "parquet"
    project_name: str = "raystec_china_202405"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON config file."""

    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError:
            return _load_simple_yaml(path)
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return {}
    if value == "[]":
        return []
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Load the small YAML subset used by this project.

    This fallback keeps the CLI usable in lean environments without PyYAML.
    It supports top-level scalars, top-level lists, and one-level mappings.
    """

    data: dict[str, Any] = {}
    current_key: str | None = None
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if line.startswith("  - ") and current_key:
                data.setdefault(current_key, []).append(_parse_scalar(line[4:]))
                continue
            if line.startswith("  ") and current_key:
                key, value = line.strip().split(":", 1)
                data.setdefault(current_key, {})[key.strip()] = _parse_scalar(value)
                continue
            key, value = line.split(":", 1)
            current_key = key.strip()
            parsed = _parse_scalar(value)
            data[current_key] = [] if parsed == {} and current_key.endswith("files") else parsed
    return data


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    """Resolve relative config paths from the Topic1 project directory."""

    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _as_path_list(
    values: list[str] | str | None,
    base_dir: Path,
) -> list[Path]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    return [_resolve_path(v, base_dir) for v in values]


def parse_date(value: Any) -> date:
    """Parse a date object or ISO date string."""

    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def config_from_mapping(
    data: dict[str, Any],
    base_dir: str | Path | None = None,
) -> DatasetBuildConfig:
    """Convert a dictionary loaded from config into a dataclass."""

    base_dir = Path(base_dir or ".").resolve()
    region_data = data.get("region", {})
    region = Region(
        lon_min=float(region_data.get("lon_min", DEFAULT_REGION[0])),
        lon_max=float(region_data.get("lon_max", DEFAULT_REGION[1])),
        lat_min=float(region_data.get("lat_min", DEFAULT_REGION[2])),
        lat_max=float(region_data.get("lat_max", DEFAULT_REGION[3])),
    )

    return DatasetBuildConfig(
        tec_data_dir=_resolve_path(data["tec_data_dir"], base_dir),
        sp3_files=_as_path_list(data.get("sp3_files"), base_dir),
        output_dir=_resolve_path(data.get("output_dir", "datasets/processed"), base_dir),
        interim_dir=_resolve_path(data.get("interim_dir", "datasets/interim"), base_dir),
        start_date=parse_date(data.get("start_date", "2024-05-07")),
        end_date=parse_date(data.get("end_date", data.get("start_date", "2024-05-07"))),
        epoch_interval_seconds=int(data.get("epoch_interval_seconds", 900)),
        select_window_seconds=int(data.get("select_window_seconds", 30)),
        sp3_tolerance_seconds=int(data.get("sp3_tolerance_seconds", 900)),
        min_elevation_deg=float(data.get("min_elevation_deg", 10.0)),
        stec_min=float(data.get("stec_min", 0.0)),
        stec_max=float(data.get("stec_max", 600.0)),
        region=region,
        stations=data.get("stations"),
        station_glob=str(data.get("station_glob", "*.mat")),
        output_format=str(data.get("output_format", "parquet")).lower(),
        project_name=str(data.get("project_name", "raystec_china_202405")),
    )


def load_dataset_config(path: str | Path) -> DatasetBuildConfig:
    """Load and validate a dataset-building config file."""

    path = Path(path).resolve()
    # Config files live in Topic1/configs, so the project directory is one
    # level above the config directory.
    project_dir = path.parent.parent
    return config_from_mapping(load_config(path), base_dir=project_dir)


def iter_dates(start: date, end: date):
    """Yield dates inclusively."""

    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def iter_epochs(day: date, interval_seconds: int):
    """Yield UTC datetimes for one day at a fixed interval."""

    current = datetime(day.year, day.month, day.day)
    end = current + timedelta(days=1)
    step = timedelta(seconds=interval_seconds)
    while current < end:
        yield current
        current += step
