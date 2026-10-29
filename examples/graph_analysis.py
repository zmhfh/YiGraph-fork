"""Run after installation: python examples/graph_analysis.py"""

import json
from yigraph.demo import run_demo

print(json.dumps(run_demo(), indent=2))
