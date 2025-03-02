import importlib.metadata

from . import datatypes, models, optimizers, profile, utils, visualization
from .estimator import Estimator

__version__ = importlib.metadata.version(__package__ or __name__)
"""Package version when the install command ran."""


__all__ = (
    "__version__",
    "Estimator",
    "models",
    "utils",
    "optimizers",
    "datatypes",
    "visualization",
    "profile",
)
