"""Identical E34 downstream protocol; E35 only submits 10% and 20%."""
import importlib.util
import sys
from config import SUITE

if __name__ == '__main__':
    spec = importlib.util.spec_from_file_location('e35_downstream', SUITE / '33_joint_transfer_mechanisms/downstream.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main()
