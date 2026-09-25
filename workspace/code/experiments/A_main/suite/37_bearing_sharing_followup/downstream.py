"""Exact E35/E33 downstream driver, with E37 paths/configuration."""
import importlib.util
import sys
from config import SUITE

if __name__ == '__main__':
    spec = importlib.util.spec_from_file_location('e37_downstream', SUITE/'33_joint_transfer_mechanisms/downstream.py')
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module); module.main()
