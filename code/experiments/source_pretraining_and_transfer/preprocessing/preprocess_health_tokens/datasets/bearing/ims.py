"""IMS raw-snapshot adapter."""
from ._loaders import build_segments as _build_segments, read_signal as _read_signal
DATASET_NAME = "ims"
def build_segments(): return _build_segments(DATASET_NAME)
def read_signal(segment): return _read_signal(DATASET_NAME, segment)
