"""
models package

Importing this package registers every built-in attenuation model with the
registry in models.base. To add a new model (Sentinel/CAMS/MERRA-2/...):

    1. Create models/your_source.py defining a BaseAttenuationModel
       subclass decorated with @register_model("your_key").
    2. Add `from . import your_source` below.

No other file needs to change -- simulation.py looks the model up by key
via models.base.get_model_class, it never imports concrete classes.
"""

from .base import BaseAttenuationModel, available_models, get_model_class, register_model
from . import clear_sky  # noqa: F401 (registers "clear")
from . import aod_beer_lambert  # noqa: F401 (registers "aod_beer_lambert")

__all__ = ["BaseAttenuationModel", "available_models", "get_model_class", "register_model"]
