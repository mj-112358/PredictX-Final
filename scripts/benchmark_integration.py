"""Complete deterministic laboratory fixture for execution comparisons."""

import json

from predictx.data import ROOT


def fixture() -> dict:
    return json.loads((ROOT / "results/refinement/SYNTHETIC_benchmark_input.json").read_text())
