"""Identical COMPONENT_ROUTING downstream protocol; CONDITIONAL_ROUTING only submits 10% and 20%."""
import importlib.util
import sys
from config import SUITE

if __name__ == '__main__':
    spec = importlib.util.spec_from_file_location('cord_conditional_downstream', SUITE / 'transfer_mechanism_comparison/downstream.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main()
