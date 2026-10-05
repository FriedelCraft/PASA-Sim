"""
config.py

Single source of truth for PASA-Sim run configuration. Every tunable
parameter in the framework lives in one of the dataclasses below and is
set via a YAML file -- no hardcoded values anywhere else in the codebase.

Design
------
Each pipeline concern (run control, city model, location, time, solar
physics, pollution/AOD, performance, output, analysis, plotting) gets its
own small dataclass with sane defaults. ``PASAConfig`` aggregates them.
``PASAConfig.from_yaml`` deep-merges a user's (possibly partial) YAML
against these defaults, so a user only needs to specify the keys they want
to override. ``validate()`` is called automatically and raises
``ConfigError`` with a specific, actionable message on any problem, rather
than failing deep inside the simulation loop hours into a run.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from . import constants as C


class ConfigError(ValueError):
    """Raised when a resolved PASAConfig fails validation."""


# ---------------------------------------------------------------------------
# Generic nested-dataclass-from-dict helper (avoids writing the same
# recursive-construction boilerplate for every section below).
# ---------------------------------------------------------------------------
def _from_dict(cls: type, data: dict[str, Any] | None) -> Any:
    """Recursively build a dataclass instance ``cls`` from a (possibly
    nested, possibly partial) plain dict, falling back to each field's
    default wherever a key is absent. Unknown keys raise ``ConfigError``
    immediately -- a silently-ignored typo in a YAML key is worse than a
    loud failure at startup.
    """
    data = data or {}
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{cls.__name__}: unknown config key(s) {sorted(unknown)}; expected one of {sorted(known)}")

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        field_type = hints.get(f.name, f.type)
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(field_type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge ``override`` onto ``base``, returning a new dict."""
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    simulation_mode: str = "clear"  # "clear" | "pollution"
    run_name: str = "unnamed_run"
    output_dir: str = "outputs/unnamed_run"
    checkpoint_dir: str = "checkpoints/unnamed_run"
    checkpoint_interval_hours: int = 500
    resume: bool = True
    force_restart: bool = False
    random_seed: int = 42
    log_level: str = "INFO"
    log_file: str | None = None

    def validate(self) -> None:
        if self.simulation_mode not in C.SIMULATION_MODES:
            raise ConfigError(f"run.simulation_mode must be one of {C.SIMULATION_MODES}, got {self.simulation_mode!r}")
        if self.checkpoint_interval_hours <= 0:
            raise ConfigError("run.checkpoint_interval_hours must be a positive integer")
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            raise ConfigError(f"run.log_level must be DEBUG/INFO/WARNING/ERROR, got {self.log_level!r}")


@dataclass
class CityModelConfig:
    glb_path: str = "data/city_model.glb"
    roof_tilt_threshold_deg: float = 45.0
    wall_tilt_max_deg: float = 135.0
    ray_offset_scale: float = 1.0e-6  # fraction of mesh bbox diagonal used as ray-origin offset
    min_face_area_m2: float = 0.01

    def validate(self) -> None:
        if not (0.0 < self.roof_tilt_threshold_deg < self.wall_tilt_max_deg <= 180.0):
            raise ConfigError(
                "city_model: require 0 < roof_tilt_threshold_deg < wall_tilt_max_deg <= 180, "
                f"got roof_tilt_threshold_deg={self.roof_tilt_threshold_deg}, wall_tilt_max_deg={self.wall_tilt_max_deg}"
            )
        if self.ray_offset_scale <= 0:
            raise ConfigError("city_model.ray_offset_scale must be positive")
        if self.min_face_area_m2 < 0:
            raise ConfigError("city_model.min_face_area_m2 must be non-negative")


@dataclass
class LocationConfig:
    latitude: float = 28.6752
    longitude: float = 77.2210
    altitude_m: float = 216.0
    timezone: str = "Asia/Kolkata"

    def validate(self) -> None:
        if not (C.LATITUDE_MIN_DEG <= self.latitude <= C.LATITUDE_MAX_DEG):
            raise ConfigError(f"location.latitude out of range: {self.latitude}")
        if not (C.LONGITUDE_MIN_DEG <= self.longitude <= C.LONGITUDE_MAX_DEG):
            raise ConfigError(f"location.longitude out of range: {self.longitude}")


@dataclass
class TimeConfig:
    year: int = 2024
    start: str | None = None  # ISO8601 override; None -> Jan 1 00:00 of `year`
    end: str | None = None  # ISO8601 override; None -> Dec 31 23:00 of `year`
    freq: str = "1h"

    def validate(self) -> None:
        if self.year < 1900 or self.year > 2100:
            raise ConfigError(f"time.year looks implausible: {self.year}")
        if self.freq != "1h":
            raise ConfigError("time.freq must be '1h': energy accumulation assumes hourly samples")
        if (self.start is None) != (self.end is None):
            raise ConfigError("time.start and time.end must be supplied together")


@dataclass
class SolarConfig:
    linke_turbidity_source: str = "climatology"  # "climatology" | "fixed"
    fixed_linke_turbidity: float = 3.0
    airmass_model: str = "kastenyoung1989"

    def validate(self) -> None:
        if self.linke_turbidity_source not in C.LINKE_TURBIDITY_SOURCES:
            raise ConfigError(f"solar.linke_turbidity_source must be one of {C.LINKE_TURBIDITY_SOURCES}")
        if self.airmass_model not in C.AIRMASS_MODELS:
            raise ConfigError(f"solar.airmass_model must be one of {C.AIRMASS_MODELS}")
        if self.fixed_linke_turbidity <= 0:
            raise ConfigError("solar.fixed_linke_turbidity must be positive")


@dataclass
class AODIngestConfig:
    format: str | None = None  # "csv" | "geotiff" | "netcdf" | "numpy"; None -> infer from extension
    path: str | None = None
    value_column: str = "aod_550nm"
    time_column: str = "timestamp"
    lat_column: str = "lat"
    lon_column: str = "lon"
    spatially_varying: bool = False
    interpolation_method: str = "linear"
    max_gap_hours: float = 72.0
    climatological_fallback_value: float = 0.35
    # Satellite AOD products (MAIAC included) are almost always distributed with UTC
    # timestamps. If a source file's timestamps are timezone-naive, they are assumed to
    # be in this timezone and converted to `location.timezone` before being matched
    # against the simulation's hourly index. Ignored if timestamps already carry an
    # explicit UTC offset (e.g. "...+00:00" / "...Z").
    source_timezone: str = "UTC"
    scale_factor: float | None = None  # GeoTIFF override; otherwise use raster metadata/dtype
    start_date: str | None = None  # explicit first daily band date when timestamps are absent

    def validate(self) -> None:
        if self.format is not None and self.format not in C.AOD_FILE_FORMATS:
            raise ConfigError(f"pollution.aod.format must be one of {C.AOD_FILE_FORMATS} or null, got {self.format!r}")
        if self.max_gap_hours <= 0:
            raise ConfigError("pollution.aod.max_gap_hours must be positive")
        if self.scale_factor is not None and self.scale_factor <= 0:
            raise ConfigError("pollution.aod.scale_factor must be positive")
        if not (C.AOD_PHYSICAL_MIN <= self.climatological_fallback_value <= C.AOD_PHYSICAL_MAX):
            raise ConfigError("pollution.aod.climatological_fallback_value outside physical AOD range")


@dataclass
class PollutionConfig:
    attenuation_model: str = C.MODEL_KEY_AOD_BEER_LAMBERT
    aod: AODIngestConfig = field(default_factory=AODIngestConfig)

    def validate(self) -> None:
        self.aod.validate()


@dataclass
class PerformanceConfig:
    n_workers: int = 1
    hour_chunk_size: int = 200
    precision: str = "float32"
    use_embree_if_available: bool = True

    def validate(self) -> None:
        if self.n_workers < 1:
            raise ConfigError("performance.n_workers must be >= 1")
        if self.hour_chunk_size < 1:
            raise ConfigError("performance.hour_chunk_size must be >= 1")
        if self.precision not in C.PRECISION_DTYPES:
            raise ConfigError(f"performance.precision must be one of {C.PRECISION_DTYPES}")


@dataclass
class OutputConfig:
    # Only the two arrays that actually scale with n_faces x n_hours are toggleable --
    # everything else in the NPZ (solar geometry, DNI, AOD group series, per-face
    # geometry/annual totals) is tiny regardless of city size and is always written.
    # Compression is likewise not a toggle: given "simulations are huge" is the whole
    # reason for the NPZ format in the first place, np.savez_compressed is always used.
    save_hourly_irradiance: bool = True
    save_hourly_shadow_mask: bool = True

    def validate(self) -> None:
        return


@dataclass
class AnalysisConfig:
    top_n_buildings: int = 20
    percentiles: list[float] = field(default_factory=lambda: [5, 25, 50, 75, 95])

    def validate(self) -> None:
        if self.top_n_buildings < 1:
            raise ConfigError("analysis.top_n_buildings must be >= 1")
        if any(not (0 <= p <= 100) for p in self.percentiles):
            raise ConfigError("analysis.percentiles must all be within [0, 100]")


@dataclass
class PlottingConfig:
    # Figures are always PNG (static) + HTML/Plotly (interactive) -- see plotting.py's
    # module docstring for why Plotly was chosen over PyVista for the interactive view.
    # dpi and the two colormaps are the only knobs that actually change output.
    dpi: int = 300
    colormap_irradiance: str = "inferno"
    colormap_loss: str = "viridis_r"

    def validate(self) -> None:
        if self.dpi < 72:
            raise ConfigError("plotting.dpi looks too low for publication figures (expected >= 72)")


@dataclass
class PASAConfig:
    run: RunConfig = field(default_factory=RunConfig)
    city_model: CityModelConfig = field(default_factory=CityModelConfig)
    location: LocationConfig = field(default_factory=LocationConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    solar: SolarConfig = field(default_factory=SolarConfig)
    pollution: PollutionConfig = field(default_factory=PollutionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)

    # -- construction ---------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path, simulation_mode: str | None = None) -> "PASAConfig":
        """Load UTF-8 YAML; file paths are relative to the configuration file."""
        path = Path(path).resolve()
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if simulation_mode is not None:
            raw.setdefault("run", {})["simulation_mode"] = simulation_mode
        cfg = cls.from_dict(raw)
        for section, name in (
            (cfg.run, "output_dir"), (cfg.run, "checkpoint_dir"), (cfg.run, "log_file"),
            (cfg.city_model, "glb_path"), (cfg.pollution.aod, "path"),
        ):
            value = getattr(section, name)
            if value is not None:
                value_path = Path(value).expanduser()
                if not value_path.is_absolute():
                    value_path = path.parent / value_path
                setattr(section, name, str(value_path.resolve()))
        return cfg

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PASAConfig":
        cfg = _from_dict(cls, raw)
        cfg.validate()
        return cfg

    # -- validation -------------------------------------------------------
    def validate(self) -> None:
        for section in (
            self.run, self.city_model, self.location, self.time, self.solar,
            self.pollution, self.performance, self.output, self.analysis, self.plotting,
        ):
            section.validate()

        # cross-section checks that no single dataclass can express alone
        if self.run.simulation_mode == "pollution":
            if self.pollution.attenuation_model == C.MODEL_KEY_CLEAR:
                raise ConfigError(
                    "run.simulation_mode='pollution' but pollution.attenuation_model='clear' "
                    "-- pick a real attenuation model (e.g. 'aod_beer_lambert') or switch mode to 'clear'"
                )
            if self.pollution.aod.path is None:
                raise ConfigError("run.simulation_mode='pollution' requires pollution.aod.path to be set")
            if self.pollution.aod.spatially_varying and self.pollution.aod.format == "csv":
                # allowed, but only meaningful if lat/lon columns are present; loader will
                # raise a clearer, data-specific error at load time if they're missing.
                pass
        elif self.run.simulation_mode == "clear":
            # In clear mode the attenuation model is forced to "clear" regardless of what
            # was written in the YAML, so a user can leave the pollution: block untouched
            # when flipping modes back and forth. This is the one place simulation_mode is
            # allowed to affect behavior outside of simulation.build_attenuation_model's
            # factory switch.
            self.pollution.attenuation_model = C.MODEL_KEY_CLEAR

    # -- (de)serialization --------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def resolved_output_dir(self) -> Path:
        return Path(self.run.output_dir)

    def resolved_checkpoint_dir(self) -> Path:
        return Path(self.run.checkpoint_dir)
