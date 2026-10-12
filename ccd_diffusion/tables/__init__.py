"""
The tables of the article, one module per table.
"""

from ._datasets import datasets
from ._tracks import num_frames, tracks
from ._models import models

__all__ = [
    "datasets",
    "num_frames",
    "tracks",
    "models",
]
