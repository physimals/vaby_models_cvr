"""
VABY_MODELS_CVR: VABY forward models for CVR

(c) 2021 University of Nottingham
"""
try:
    from ._version import __version__, __timestamp__
except ImportError:
    __version__ = "Unknown version"
    __timestamp__ = "Unknown timestamp"

from .petco2 import CvrPetCo2Model
from ._version import __version__

__all__ = [
    "CvrPetCo2Model",
    "__version__"
]
