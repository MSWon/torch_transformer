import sys
from collections import defaultdict


from importlib_metadata import PackageNotFoundError, metadata  # type: ignore

__all__ = ("__version__", "__description__")


try:
    meta = metadata("nmt")
except PackageNotFoundError:
    meta = defaultdict(lambda: None, version="0.0.0-dev")

__version__ = meta["version"]
__description__ = meta["Summary"]