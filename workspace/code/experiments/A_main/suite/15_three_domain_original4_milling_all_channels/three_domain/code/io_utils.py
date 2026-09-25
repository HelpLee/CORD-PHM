"""Retry transient Windows/OneDrive file replacement denial; preserve atomic IO."""
import time
from data import write as base_write


def resilient_write(path,obj):
    for attempt in range(20):
        try:
            base_write(path,obj)
            return
        except PermissionError:
            if attempt==19:raise
            time.sleep(min(.1*(attempt+1),.5))
