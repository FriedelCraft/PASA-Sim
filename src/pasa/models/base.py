"""
models/base.py

The pluggability contract for atmospheric attenuation models. Every model
(clear-sky no-op, MODIS MAIAC Beer-Lambert, and any future Sentinel/CAMS/
MERRA-2 model) implements the same `compute_transmission` interface, so
simulation.py and irradiance.py never branch on `simulation_mode` -- they
just call whatever model the config-driven factory handed them.

Adding a new attenuation source later means: subclass BaseAttenuationModel
in a new file under models/, decorate it with @register_model("your_key"),
and add that file's import to models/__init__.py. Nothing in simulation.py,
irradiance.py, or the CLI needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd

_MODEL_REGISTRY: dict[str, "type[BaseAttenuationModel]"] = {}


def register_model(key: str) -> Callable[[type], type]:
    """Class decorator that registers an attenuation model under `key` so
    it can be constructed via `get_model_class(key)` / the simulation
    factory without simulation.py importing the concrete class directly.
    """

    def _decorator(cls: type["BaseAttenuationModel"]) -> type["BaseAttenuationModel"]:
        if key in _MODEL_REGISTRY and _MODEL_REGISTRY[key] is not cls:
            raise ValueError(f"Attenuation model key {key!r} already registered to {_MODEL_REGISTRY[key].__name__}")
        _MODEL_REGISTRY[key] = cls
        return cls

    return _decorator


def get_model_class(key: str) -> type["BaseAttenuationModel"]:
    if key not in _MODEL_REGISTRY:
        raise KeyError(f"No attenuation model registered under {key!r}. Known models: {sorted(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[key]


def available_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


class BaseAttenuationModel(ABC):
    """Common interface for every atmospheric attenuation model.

    A model's only job is to produce a dimensionless transmission factor
    in [0, 1] for a batch of hours (and, for spatially-resolved models,
    for a batch of AOD "pixel groups" -- see io.aod_loaders), such that

        attenuated_DNI = clear_sky_DNI * transmission

    Implementations must be pure functions of their inputs (no hidden
    state mutated between calls) so the same instance can safely be reused
    across hour-chunks and, if performance.n_workers > 1, across worker
    processes.
    """

    #: Registry key this model was constructed under; set by the factory
    #: in simulation.py so it can be recorded in NPZ run metadata.
    name: str = "base"

    @abstractmethod
    def compute_transmission(
        self,
        times: pd.DatetimeIndex,
        solar_zenith_deg: np.ndarray,
    ) -> np.ndarray:
        """Return a (n_groups, n_hours) transmission array in [0, 1] for
        the given hourly timestamps, where n_groups is 1 for a spatially-
        uniform model/run and >1 for a spatially-varying pollution run
        (one row per unique AOD pixel group -- see io.aod_loaders for how
        groups are formed). A spatially-uniform model (e.g. ClearSkyModel)
        should always return shape (1, n_hours).
        """
        raise NotImplementedError

    def group_metadata(self) -> dict:
        """Optional diagnostic metadata about the spatial groups used by
        this model (e.g. group centroid lat/lon), saved into the output
        NPZ for later plotting. Default: no spatial grouping information.
        """
        return {}

    def face_to_group_id(self, n_faces: int) -> np.ndarray:
        """Map every face (0..n_faces-1) to a row index of this model's
        `compute_transmission` output. Default: every face maps to group
        0 -- correct for any spatially-uniform model (ClearSkyModel, or a
        single-value AODAttenuationModel run). Spatially-varying models
        override this to return their precomputed nearest-pixel
        assignment (see AODAttenuationModel._assign_groups).
        """
        return np.zeros(n_faces, dtype=np.int32)
