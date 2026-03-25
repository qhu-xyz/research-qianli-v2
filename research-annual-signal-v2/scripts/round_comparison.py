"""Moved to scripts/evaluation/. This file is a compatibility wrapper."""
import runpy, sys, os
sys.argv[0] = os.path.join(os.path.dirname(__file__), "evaluation", os.path.basename(__file__))
runpy.run_path(sys.argv[0], run_name="__main__")
