"""envisionboxbio-apetrack: zero-shot multi-ape detection, tracking and pose estimation in video."""
import os

from .citation import CITATIONS, print_citations
from .config import SMOOTHING, Config
from .pose import KEYPOINTS
from .tracker import ApeTracker

__version__ = "0.1.1"
__all__ = ["ApeTracker", "Config", "SMOOTHING", "KEYPOINTS", "CITATIONS"]

if not os.environ.get("APETRACK_QUIET"):
    print_citations()
