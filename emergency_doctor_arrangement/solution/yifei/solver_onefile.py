
# solver_onefile.py
# Single-file pipeline (besides utils.py):
# - Stage-1: compute hourly staffing targets s_target
# - Stage-2: choose borrowed count B by approximate extra wait vs weekly borrow cost
# - Build a concrete weekly schedule (doctors with 8h shifts) for n_inhouse + B
# - Emit JSON that benchmark.py can consume directly
#
# Usage:
#   python solver_onefile.py --in_json /path/to/sample_benchmark_input.json \
#       --out_json schedule_for_benchmark.json \
#       --n_inhouse 11 --b_max 20 --alpha 1.3 --headroom 20 --mode optimal
#
# Then run your provided benchmark:
#   python benchmark.py --json schedule_for_benchmark.json

import sys
from pathlib import Path
import argparse, json, math, numpy as np
from typing import List, Dict, Any, Tuple

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from emergency_doctor_arrangement.utils.utils import load_problem, dump_json, optimal_s_for_hour, waiting_hours

def solve_day_targets(arr_24: List[float], mu: float, alpha: float, headroom: int, mode: str,
                      r_grid=None, t0: float=1.7, k: float=5.0) -> Dict[str, Any]:
    lam = {h: float(arr_24[h]) for h in range(24)}
    lam6, lam7 = lam[6], lam[7]

    def hour_cost(lam, mu, s, alpha):
        w = waiting_hours(lam, mu, s) / max(lam, 1e-9) if lam > 0 else 0.0
        if w >= 1e8:
            return 1e15
        return lam * w + alpha * s

    def eval_cost(s_night: int, r: float):
        s_by_h = {}
        cost_night = 0.0
        for h in range(0,6):
            cost_night += hour_cost(lam[h], mu, s_night, alpha)
            s_by_h[h] = s_night
        lam6_eff = (1.0 - r) * lam6
        cost_night += hour_cost(lam6_eff, mu, s_night, alpha)
        s_by_h[6] = s_night

        lam7_eff = lam7 + r * lam6
        s7 = optimal_s_for_hour(lam7_eff, mu, alpha, headroom)
        cost_day = hour_cost(lam7_eff, mu, s7, alpha)
        s_by_h[7] = s7
        for h in range(8,24):
            sh = optimal_s_for_hour(lam[h], mu, alpha, headroom)
            cost_day += hour_cost(lam[h], mu, sh, alpha)
            s_by_h[h] = sh

        extra_wait = 0.5 * r * lam6  # deferral penalty proxy
        total = cost_night + cost_day + extra_wait
        return total, s_by_h, {"lam6_eff": lam6_eff, "lam7_eff": lam7_eff, "extra_wait": extra_wait}

    lam_night_max_if_r1 = max([lam[h] for h in range(0,6)] + [0.0])
    smin = max(1, math.ceil(lam_night_max_if_r1 / mu))
    smax = max(1, math.ceil(max([lam[h] for h in range(0,7)]) / mu) + headroom)

    best = {"cost": float("inf")}
    if mode == "optimal":
        for s in range(smin, smax+1):
            for r in r_grid:
                total, s_h, dbg = eval_cost(s, float(r))
                if total < best["cost"]:
                    best = {"cost": total, "s_night": s, "r": float(r), "s_by_h": s_h, **dbg}
    else:
        avg_pre = np.mean([lam[h] for h in range(0,6)]) if np.sum([lam[h] for h in range(0,6)])>0 else 0.0
        ratio = lam6 / max(avg_pre, 1e-9) if avg_pre>0 else 0.0
        r_hat = max(0.0, min(1.0, (ratio - t0) / k))
        for s in range(smin, smax+1):
            total, s_h, dbg = eval_cost(s, r_hat)
            if total < best["cost"]:
                best = {"cost": total, "s_night": s, "r": float(r_hat), "s_by_h": s_h, **dbg}
    return best

def stage1_targets(arrival_rates: List[float], mu: float, alpha: float, headroom: int,
                   mode: str="optimal", r_steps: int=11, t0: float=1.7, k: float=5.0) -> Tuple[List[int], List[Dict[str, Any]]]:
    days = len(arrival_rates) // 24
    r_grid = np.linspace(0, 1, max(2, r_steps))
    s_target = []
    per_day = []
    for d in range(days):
        day_arr = arrival_rates[d*24:(d+1)*24]
        res = solve_day_targets(day_arr, mu, alpha, headroom, mode, r_grid=r_grid, t0=t0, k=k)
        s_by_h = res["s_by_h"]
        s_target.extend([int(s_by_h[h]) for h in range(24)])
        per_day.append({
            "day_index": d,
            "s_night": int(res["s_night"]),
            "r": float(res["r"]),
            "lam6_eff": float(res["lam6_eff"]),
            "lam7_eff": float(res["lam7_eff"]),
            "day_cost_proxy": float(res["cost"])
        })
    return s_target, per_day

def choose_borrowed_B(arrival_rates: List[float], mu: float, s_target: List[int],
                      n_inhouse: int, c_borrow: float, b_max: int) -> Dict[str, Any]:
    best = None
    for B in range(0, b_max+1):
        cap = n_inhouse + B
        extra = 0.0
        for l, s_t in zip(arrival_rates, s_target):
            s_act = min(s_t, cap)
            base = waiting_hours(l, mu, s_t)
            neww = waiting_hours(l, mu, s_act)
            extra += max(0.0, neww - base)
        total_cost = extra + c_borrow * B
        cand = {"B": B, "extra_wait_hours": extra, "total_cost": total_cost}
        if (best is None) or (cand["total_cost"] < best["total_cost"]):
            best = cand
    return best

def build_weekly_schedule(n_inhouse: int, B: int) -> List[Dict[str, Any]]:
    """
    Build a concrete weekly schedule with fixed 8h shifts (0-8, 8-16, 16-24 each day) for all doctors.
    All inhouse and borrowed doctors work the same 5-on/2-off rotation starting day 0,
    but staggered so that coverage is uniform.
    """
    HORIZON = 168
    SHIFT_BLOCKS = [(0,8,"night"), (8,16,"day"), (16,24,"evening")]
    total_docs = n_inhouse + B

    def doctor_id(i, origin):
        return f"{'I' if origin=='internal' else 'X'}{i+1}"

    doctors = []
    # Split 5/2 pattern across the week; simple cyclic assignment per doctor
    # Each working day → 3 shifts per day (we'll assign 1 shift per day per doctor)
    # For simplicity, each doctor works 5 days x 1 shift/day (8h) per week.
    work_days = 5
    off_days = 2
    cycle = work_days + off_days

    # Distribute start day offsets to spread workforce
    for i in range(total_docs):
        origin = "internal" if i < n_inhouse else "borrowed"
        did = doctor_id(i, origin)
        start_offset = i % cycle  # shift the 5-on/2-off window
        shifts = []
        for day in range(7):
            # Determine if this doctor works on this day in the 5/2 pattern
            if ((day - start_offset) % cycle) < work_days:
                # assign one shift among the three slots, stagger by doctor index for balance
                slot_idx = (i + day) % 3
                start_h = day*24 + SHIFT_BLOCKS[slot_idx][0]
                end_h = day*24 + SHIFT_BLOCKS[slot_idx][1]
                tag = SHIFT_BLOCKS[slot_idx][2]
                shifts.append({"start": int(start_h), "end": int(end_h), "tag": tag})
        doctors.append({"id": did, "origin": origin, "shifts": shifts})
    return doctors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="Path to input problem JSON (arrival_rates, mu, c_borrow).")
    ap.add_argument("--out_json", required=True, help="Output JSON that benchmark.py can consume.")
    ap.add_argument("--n_inhouse", type=int, default=11)
    ap.add_argument("--alpha", type=float, default=1.3)
    ap.add_argument("--headroom", type=int, default=20)
    ap.add_argument("--mode", choices=["optimal","heuristic"], default="optimal")
    ap.add_argument("--b_max", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--include_overtime_in_cost", action="store_true")
    args = ap.parse_args()

    prob = load_problem(args.in_json)
    arr = list(map(float, prob["arrival_rates"]))
    mu = float(prob["mu"])
    c_borrow = float(prob.get("c_borrow", 20.0))

    # Stage-1 targets
    s_target, per_day = stage1_targets(arr, mu, args.alpha, args.headroom, mode=args.mode)

    # Stage-2: choose borrowed B
    best = choose_borrowed_B(arr, mu, s_target, n_inhouse=int(args.n_inhouse),
                             c_borrow=c_borrow, b_max=int(args.b_max))
    B = int(best["B"])

    # Build schedule with n_inhouse + B
    doctors = build_weekly_schedule(int(args.n_inhouse), B)

    out = {
        "arrival_rates": arr,
        "mu": mu,
        "c_borrow": c_borrow,
        "include_overtime_in_cost": bool(args.include_overtime_in_cost),
        "seed": args.seed if args.seed is not None else prob.get("seed", None),
        "doctors": doctors
    }
    dump_json(out, args.out_json)

    # Also emit a small metadata sidecar for transparency (not required by benchmark)
    meta = {
        "alpha": args.alpha,
        "headroom": args.headroom,
        "mode": args.mode,
        "chosen_B": B,
        "stage2_extra_wait_hours": float(best["extra_wait_hours"]),
        "stage2_total_cost_proxy": float(best["total_cost"]),
    }
    meta_path = args.out_json.replace(".json", "_meta.json")
    dump_json(meta, meta_path)
    print(f"Wrote schedule JSON to: {args.out_json}")
    print(f"Wrote meta JSON to: {meta_path}")
    print(f"Chosen borrowed B = {B} (proxy objective = extra_wait + c_borrow * B)")

if __name__ == "__main__":
    main()
