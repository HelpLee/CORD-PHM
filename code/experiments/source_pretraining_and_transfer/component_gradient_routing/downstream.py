"""Reuse the verified E33/E28 downstream protocol with isolated COMPONENT_ROUTING outputs."""
import importlib.util
import sys
from config import SUITE

spec = importlib.util.spec_from_file_location("e33_downstream", SUITE / "transfer_mechanism_comparison/downstream.py")
module = importlib.util.module_from_spec(spec)
sys.modules["e33_downstream"] = module
spec.loader.exec_module(module)
module.main()
