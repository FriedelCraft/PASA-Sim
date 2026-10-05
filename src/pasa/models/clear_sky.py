"""
models/clear_sky.py

The identity attenuation model: transmission is always 1.0. This exists
specifically so "clear" and "pollution" runs share the exact same
irradiance code path (irradiance.py never checks simulation_mode) -- the
only difference between the two modes is which BaseAttenuationModel
subclass the factory in simulation.py hands to irradiance.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import MODEL_KEY_CLEAR
from .base import BaseAttenuationModel, register_model


@register_model(MODEL_KEY_CLEAR)
class ClearSkyModel(BaseAttenuationModel):
    """Transmission == 1.0 for every hour: no attenuation beyond what the
    Ineichen clear-sky DNI itself already represents.
    """

    name = MODEL_KEY_CLEAR

    def compute_transmission(self, times: pd.DatetimeIndex, solar_zenith_deg: np.ndarray) -> np.ndarray:
        return np.ones((1, len(times)), dtype=np.float32)
