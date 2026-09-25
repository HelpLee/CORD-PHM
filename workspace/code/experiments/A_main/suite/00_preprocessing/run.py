"""Use the bundled preprocessor; pass --out-root for a separate candidate directory."""
import runpy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
runpy.run_module('preprocess_health_tokens.main',run_name='__main__')
