from .base import BaseOCRAdapter, filter_regions, load_regions
from .chandra import ChandraAdapter
from .paddle import PaddleAdapter
from .surya import SuryaAdapter
from .tesseract import TesseractAdapter

__all__ = [
    "BaseOCRAdapter",
    "ChandraAdapter",
    "PaddleAdapter",
    "SuryaAdapter",
    "TesseractAdapter",
    "filter_regions",
    "load_regions",
]
