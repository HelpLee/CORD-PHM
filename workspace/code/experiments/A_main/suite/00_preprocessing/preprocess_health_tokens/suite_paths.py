"""Resolve external raw/NPZ data while keeping preprocessing code in this bundle."""
import os
from pathlib import Path

def data_code_root():
    configured=os.environ.get('HEALTHTOKEN_DATA_CODE_ROOT')
    if configured:
        root=Path(configured).resolve()
        if not (root/'data_phm').is_dir():raise FileNotFoundError(root)
        return root
    for parent in Path(__file__).resolve().parents:
        for candidate in (parent,parent/'code'):
            if (candidate/'data_phm/raw').is_dir():return candidate
    raise FileNotFoundError('Set HEALTHTOKEN_DATA_CODE_ROOT to directory containing data_phm/raw')
