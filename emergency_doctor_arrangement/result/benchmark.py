#!/usr/bin/env python3
"""
CLI runner for the ER scheduling benchmark.

Usage:
  python run_benchmark.py --json path/to/input.json [--seed 123] [--overtime] [--strict]

Input JSON schema matches er_benchmark.build_input_from_dict():
{
  "arrival_rates": [168 floats],
  "mu": 4.0,
  "c_borrow": 20.0,                  // optional
  "include_overtime_in_cost": false, // optional
  "seed": 42,                        // optional
  "doctors": [
    {"id":"D1","origin":"internal","shifts":[{"start":0,"end":7,"tag":"night"}, {"start":7,"end":15,"tag":"day"}]}
  ]
}
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import the benchmark module (placed alongside this script or installed in PYTHONPATH)
try:
    from emergency_doctor_arrangement.utils.er_benchmark import build_input_from_dict, simulate_and_score
except Exception as e:
    print("ERROR: Failed to import er_benchmark module. Make sure er_benchmark.py is available.", file=sys.stderr)
    raise

def main():
    ap = argparse.ArgumentParser(description="Run ER scheduling benchmark from a JSON file.")
    ap.add_argument("--json", required=True, help="Path to input JSON file.")
    ap.add_argument("--seed", type=int, default=None, help="Optional override of RNG seed for reproducibility.")
    ap.add_argument("--overtime", action="store_true", help="Include overtime work time in cost (adds to staff_hours).")
    ap.add_argument("--strict", action="store_true", help="Exit with non-zero code if schedule is infeasible.")
    args = ap.parse_args()

    path = Path(args.json)
    if not path.exists():
        print(f"ERROR: JSON file not found: {path}", file=sys.stderr)
        sys.exit(2)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: Failed to parse JSON: {e}", file=sys.stderr)
        sys.exit(3)

    # Optional overrides from CLI
    if args.seed is not None:
        data["seed"] = args.seed
    if args.overtime:
        data["include_overtime_in_cost"] = True

    # Build input and run
    try:
        inp = build_input_from_dict(data)
    except Exception as e:
        print(f"ERROR: Invalid input structure: {e}", file=sys.stderr)
        sys.exit(4)

    res = simulate_and_score(inp)

    # Pretty print results
    print("=== ER Benchmark Result ===")
    print(f"Feasible:            {res.feasible}")
    if res.violations:
        print("Violations:")
        for v in res.violations:
            print(f" - {v}")
    print(f"Total arrivals:      {res.details.get('arrivals')}")
    print(f"Total wait (hours):  {res.total_wait_hours:.3f}")
    print(f"Staff hours:         {res.staff_hours:.3f}")
    print(f"Borrowed cost:       {res.borrowed_cost:.3f}")
    print(f"Objective:           {res.objective:.3f}")

    if args.strict and (not res.feasible):
        sys.exit(1)

if __name__ == "__main__":
    main()
