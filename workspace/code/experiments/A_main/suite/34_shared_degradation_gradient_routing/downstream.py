"""Reuse the verified E33/E28 downstream protocol with isolated E34 outputs."""
import importlib.util
import sys
from config import SUITE

spec = importlib.util.spec_from_file_location("e33_downstream", SUITE / "33_joint_transfer_mechanisms/downstream.py")
module = importlib.util.module_from_spec(spec)
sys.modules["e33_downstream"] = module
spec.loader.exec_module(module)
module.main()
