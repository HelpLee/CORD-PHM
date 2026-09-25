"""Exact CONDITIONAL_ROUTING/E33 downstream driver, with SELECTED_MODEL paths/configuration."""
import importlib.util
import sys
from config import SUITE

if __name__ == '__main__':
    spec = importlib.util.spec_from_file_location('cord_selected_downstream', SUITE/'transfer_mechanism_comparison/downstream.py')
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module); module.main()
