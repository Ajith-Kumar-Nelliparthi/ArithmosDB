"""
ArithmosDB — GPU-accelerated vector database
"""
from .index import FlatIndex, IVFIndex
from .utils import load_library

__version__ = "0.1.0"
__all__ = ["FlatIndex", "IVFIndex"]
