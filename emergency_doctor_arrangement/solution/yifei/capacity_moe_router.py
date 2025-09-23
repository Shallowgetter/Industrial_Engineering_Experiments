#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mixture-of-Experts (MOE) capacity planner for night surge at 6–7.
Decisions per day:
  - s_night (integer, fixed for hours 0..6)
  - r in [0,1]: fraction of 6–7 arrivals deferred to 7–8
Cost:
  sum_{hours} [lambda * E[Wq] + alpha * s] + 0.5 * r * lambda6

Modes:
  --mode optimal   : grid-search r in [0,1] and s_night integer
  --mode heuristic : router picks r_hat = clip((ratio6/avg_pre - t0)/k, 0, 1),
                     then s_night is optimized given r_hat.

Outputs:
  moe_day.csv  -> per-day (s_night, r, cost breakdown)
  moe_24h.csv  -> per-hour staffing and E[Wq] mins
"""
import argparse, math, re
from functools import lru_cache
import numpy as np
import pandas as pd

def load_lambdas(xlsx_path: str):
    raw = pd.read_excel(xlsx_path, sheet_name=" 数据", header=None)
    m = re.search(r'μ[:：]\s*([0-9]+(\.[0-9]+)?)', str(raw.iloc[1,0]))
    mu = float(m.group(1)) if m else 6.0
    hours = list(raw.iloc[4, 1:1+24].astype(str).values)
    days = raw.iloc[5:12, 0].astype(str).tolist()
    lambda_table = raw.iloc[5:12, 1:1+24].astype(float).values
    df = pd.DataFrame(lambda_table, index=days, columns=range(24)).reset_index().rename(columns={"index":"day"})
    return df, mu

def erlang_c(lam: float, mu: float, s: int) -> float:
    if s <= 0: return 1.0
    rho = lam / (mu * s)
    if rho >= 1.0: return 1.0
    a = lam / mu
    sum_terms = 0.0
    term = 1.0
    for n in range(0, s):
        if n > 0: term *= a / n
        sum_terms += term
    term *= a / s
    top = term / (1.0 - rho)
    return top / (sum_terms + top)

@lru_cache(maxsize=None)
def ewq(lam: float, mu: float, s: int) -> float:
    if s<=0: return 1e9
    rho = lam/(mu*s)
    if rho>=1.0: return 1e9
    pw = erlang_c(lam, mu, s)
    return pw / (mu*s - lam)

def hour_cost(lam: float, mu: float, s: int, alpha: float) -> float:
    w = ewq(lam, mu, s)
    if w >= 1e8: return 1e15
    return lam*w + alpha*s

def optimal_s_for_hour(lam: float, mu: float, alpha: float, headroom: int) -> int:
    s_min = max(1, math.ceil(lam/mu))
    best_s, best_c = None, float('inf')
    for s in range(max(1, s_min-1), s_min + headroom + 1):
        c = hour_cost(lam, mu, s, alpha)
        if c < best_c:
            best_s, best_c = s, c
    return int(best_s)

def solve_day(df_row: pd.Series, mu: float, alpha: float, headroom: int, mode: str,
              r_grid=None, t0=1.7, k=5.0):
    lam = {h: float(df_row[h]) for h in range(24)}
    lam6, lam7 = lam[6], lam[7]

    # helper to compute total cost given (s_night, r)
    def eval_cost(s_night: int, r: float):
        s_by_h = {}
        cost_night = 0.0
        # hours 0..5 with s_night
        for h in range(0,6):
            cost_night += hour_cost(lam[h], mu, s_night, alpha)
            s_by_h[h] = s_night
        # hour 6 with lam6_eff
        lam6_eff = (1.0-r)*lam6
        cost_night += hour_cost(lam6_eff, mu, s_night, alpha)
        s_by_h[6] = s_night

        cost_day = 0.0
        lam7_eff = lam7 + r*lam6
        s7 = optimal_s_for_hour(lam7_eff, mu, alpha, headroom)
        cost_day += hour_cost(lam7_eff, mu, s7, alpha)
        s_by_h[7] = s7
        for h in range(8,24):
            sh = optimal_s_for_hour(lam[h], mu, alpha, headroom)
            cost_day += hour_cost(lam[h], mu, sh, alpha)
            s_by_h[h] = sh

        extra_wait = 0.5 * r * lam6  # deferral residual
        total = cost_night + cost_day + extra_wait

        ewq_mins = {h: (ewq(lam[h], mu, s_by_h[h])*60 if h!=6 and h!=7 else None) for h in range(24)}
        ewq_mins[6] = ewq(lam6_eff, mu, s_night)*60
        ewq_mins[7] = ewq(lam7_eff, mu, s7)*60
        return total, cost_night, cost_day, extra_wait, lam6_eff, lam7_eff, s_by_h, ewq_mins

    # bounds for s_night
    lam_night_max_if_r1 = max([lam[h] for h in range(0,6)] + [0.0])
    smin = max(1, math.ceil(lam_night_max_if_r1/mu))
    smax = max(1, math.ceil(max([lam[h] for h in range(0,7)])/mu) + headroom)

    if mode == "optimal":
        best = {"cost": float('inf')}
        for s in range(smin, smax+1):
            for r in r_grid:
                total, cn, cd, ew, l6e, l7e, s_h, wmins = eval_cost(s, float(r))
                if total < best["cost"]:
                    best = {"cost": total, "s_night": s, "r": float(r),
                            "cn": cn, "cd": cd, "ew": ew, "l6e": l6e, "l7e": l7e,
                            "s_by_h": s_h, "ewq_mins": wmins}
        return best
    else:  # heuristic
        # router picks r_hat from features
        avg_pre = np.mean([lam[h] for h in range(0,6)])
        ratio = lam6 / max(avg_pre, 1e-9)
        r_hat = max(0.0, min(1.0, (ratio - t0)/k))
        # optimize s_night given r_hat
        best = {"cost": float('inf')}
        for s in range(smin, smax+1):
            total, cn, cd, ew, l6e, l7e, s_h, wmins = eval_cost(s, r_hat)
            if total < best["cost"]:
                best = {"cost": total, "s_night": s, "r": float(r_hat),
                        "cn": cn, "cd": cd, "ew": ew, "l6e": l6e, "l7e": l7e,
                        "s_by_h": s_h, "ewq_mins": wmins}
        return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx_path", type=str)
    ap.add_argument("--alpha", type=float, default=1.3)
    ap.add_argument("--headroom", type=int, default=20)
    ap.add_argument("--mode", choices=["optimal","heuristic"], default="optimal")
    ap.add_argument("--t0", type=float, default=1.7, help="heuristic threshold")
    ap.add_argument("--k", type=float, default=5.0, help="heuristic slope")
    ap.add_argument("--r_steps", type=int, default=11, help="grid points for r in [0,1]")
    args = ap.parse_args()

    df, mu = load_lambdas(args.xlsx_path)
    r_grid = np.linspace(0,1,max(2,args.r_steps))

    rows_day, rows_24h = [], []
    for _, row in df.iterrows():
        res = solve_day(row, mu, args.alpha, args.headroom, args.mode,
                        r_grid=r_grid, t0=args.t0, k=args.k)
        rows_day.append({
            "day": row["day"],
            "s_night": res["s_night"],
            "r": res["r"],
            "total_cost": res["cost"],
            "cost_night": res["cn"],
            "cost_day": res["cd"],
            "extra_wait": res["ew"],
            "lam6_eff": res["l6e"],
            "lam7_eff": res["l7e"]
        })
        for h in range(24):
            rows_24h.append({
                "day": row["day"], "hour": h,
                "s_opt": res["s_by_h"][h],
                "E[Wq]_mins": res["ewq_mins"][h]
            })
    pd.DataFrame(rows_day).to_csv("moe_day.csv", index=False)
    pd.DataFrame(rows_24h).to_csv("moe_24h.csv", index=False)

if __name__ == "__main__":
    main()
