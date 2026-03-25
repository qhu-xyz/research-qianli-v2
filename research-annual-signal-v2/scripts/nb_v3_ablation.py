"""Moved to scripts/research/. This file is a compatibility wrapper."""
import runpy, sys, os
sys.argv[0] = os.path.join(os.path.dirname(__file__), "research", os.path.basename(__file__))
runpy.run_path(sys.argv[0], run_name="__main__")
