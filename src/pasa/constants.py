"""
constants.py

Physical, geodetic, and framework-vocabulary constants for PASA-Sim
(Pollution-Aware Solar Assessment simulator).

Everything in this file is either fixed by physics/geodesy, or defines the
framework's closed vocabulary (e.g. legal simulation modes) so that list is
never duplicated across config validation, CLI help text, and docstrings.
None of these are user-tunable run parameters -- those live in config.py /
config.yaml. If you find yourself wanting to change a value here per-run,
it belongs in the config schema instead.
"""

from typing import Final

# ---------------------------------------------------------------------------
# Geodesy
# ---------------------------------------------------------------------------
#: Mean Earth radius (m). Used ONLY by geometry.enu_to_latlon() for the local
#: flat-Earth approximation that maps a face's local ENU (East-North-Up,
#: meters, origin at the AOI reference point) position to an approximate
#: geographic lat/lon, so it can be matched against a gridded/point AOD
#: product. Error from this spherical approximation is well under 0.5% for
#: AOI extents up to several tens of km -- comfortably covers a district-
#: scale study area such as Civil Lines (~1-2 km across). For a continental-
#: scale domain, replace this with a proper ellipsoidal projection (e.g.
#: via pyproj) instead of relying on it.
EARTH_RADIUS_M: Final[float] = 6_371_000.0

# ---------------------------------------------------------------------------
# Framework vocabulary (single source of truth -- referenced by config
# validation, the model/loader registries, and the CLI so the list of legal
# values is never duplicated or allowed to drift).
# ---------------------------------------------------------------------------
SIMULATION_MODES: Final[tuple[str, ...]] = ("clear", "pollution")
AOD_FILE_FORMATS: Final[tuple[str, ...]] = ("csv", "geotiff", "netcdf", "numpy")
RAY_BACKENDS: Final[tuple[str, ...]] = ("embree", "triangle")
PRECISION_DTYPES: Final[tuple[str, ...]] = ("float32", "float16")
LINKE_TURBIDITY_SOURCES: Final[tuple[str, ...]] = ("climatology", "fixed")
AIRMASS_MODELS: Final[tuple[str, ...]] = (
    "kastenyoung1989",
    "kasten1966",
    "simple",
    "pickering2002",
    "young1994",
    "gueymard1993",
)

# ---------------------------------------------------------------------------
# Physical sanity bounds. These are used for INPUT VALIDATION / robust
# handling of noisy or corrupt remote-sensing data -- they are never used
# inside the physics computation itself, so they don't count as "fake
# constants" standing in for real physical parameters.
# ---------------------------------------------------------------------------
#: MODIS MAIAC AOD (550 nm) essentially never exceeds ~3-4 even during
#: extreme smoke/haze episodes; values outside [0, 5] are treated as sensor
#: artifacts by the AOD loaders and flagged rather than silently trusted.
AOD_PHYSICAL_MIN: Final[float] = 0.0
AOD_PHYSICAL_MAX: Final[float] = 5.0

TILT_MIN_DEG: Final[float] = 0.0
TILT_MAX_DEG: Final[float] = 180.0
AZIMUTH_MIN_DEG: Final[float] = 0.0
AZIMUTH_MAX_DEG: Final[float] = 360.0
LATITUDE_MIN_DEG: Final[float] = -90.0
LATITUDE_MAX_DEG: Final[float] = 90.0
LONGITUDE_MIN_DEG: Final[float] = -180.0
LONGITUDE_MAX_DEG: Final[float] = 180.0

# ---------------------------------------------------------------------------
# NPZ schema version. Bump on any change to the set of keys written by
# io.npz_io.save_simulation_result, so analysis.py / plotting.py can detect
# and reject stale files with a clear error instead of a cryptic KeyError.
# ---------------------------------------------------------------------------
NPZ_SCHEMA_VERSION: Final[str] = "1.0.0"

# ---------------------------------------------------------------------------
# Registry keys for the pluggable attenuation-model system (models/ package).
# Adding a new source (Sentinel, CAMS, MERRA-2, ...) means adding one new
# file that registers itself under a new key here -- no other file changes.
# ---------------------------------------------------------------------------
MODEL_KEY_CLEAR: Final[str] = "clear"
MODEL_KEY_AOD_BEER_LAMBERT: Final[str] = "aod_beer_lambert"
