#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-2 weekly borrowed-doctor planning on top of a given hourly staffing target s_target.
Minimizes: weekly borrowed cost + extra waiting when capacity < target.
Inputs:
  - Excel data file (to read λ and μ)
  - Hourly target CSV (columns: day, hour (0-23), s_opt)
Parameters:
  --n_inhouse   number of in-house doctors (default 11)
  --c_borrow    weekly cost per borrowed doctor (default 20)
  --b_max       max borrowed candidates (default 12)
Outputs:
  - search_curve.csv    (B vs total_cost)
  - best_hourly.csv     (per-hour shortage & extra waiting)
  - best_daily.csv      (daily aggregates)
"""
import argparse, re, math, pandas as pd
from functools import lru_cache

def load_lambda_mu(xlsx_path: str):
    raw = pd.read_excel(xlsx_path, sheet_name=" 数据", header=None)
    import re
    m = re.search(r'μ[:：]\s*([0-9]+(\.[0-9]+)?)', str(raw.iloc[1,0]))
    mu = float(m.group(1)) if m else 6.0
    days = raw.iloc[5:12, 0].astype(str).tolist()
    lam_tbl = raw.iloc[5:12, 1:1+24].astype(float).values
    df = pd.DataFrame(lam_tbl, index=days, columns=range(24)).reset_index().rename(columns={"index":"day"})
    rec = []
    for _, row in df.iterrows():
        for h in range(24):
            rec.append({"day": row["day"], "hour": h, "lambda": float(row[h])})
    return pd.DataFrame(rec), mu

def erlang_c(lam: float, mu: float, s: int) -> float:
    if s <= 0: return 1.0
    rho = lam/(mu*s)
    if rho >= 1.0: return 1.0
    a = lam/mu
    sum_terms = 0.0
    term = 1.0
    for n in range(0, s):
        if n > 0: term *= a/n
        sum_terms += term
    term *= a/s
    top = term / (1.0 - rho)
    return top / (sum_terms + top)

@lru_cache(maxsize=None)
def ewq(lam: float, mu: float, s: int) -> float:
    if s <= 0: return 1e9
    rho = lam/(mu*s)
    if rho >= 1.0: return 1e9
    pw = erlang_c(lam, mu, s)
    return pw / (mu*s - lam)

def waiting_hours(lam: float, mu: float, s: int) -> float:
    return lam * ewq(lam, mu, s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx_path", type=str)
    ap.add_argument("target_csv", type=str, help="CSV with columns: day,hour(0-23),s_opt")
    ap.add_argument("--n_inhouse", type=int, default=11)
    ap.add_argument("--c_borrow", type=float, default=20.0)
    ap.add_argument("--b_max", type=int, default=12)
    args = ap.parse_args()

    lam_df, mu = load_lambda_mu(args.xlsx_path)
    st = pd.read_csv(args.target_csv)
    assert {"day","hour","s_opt"}.issubset(st.columns), "target_csv must have day,hour,s_opt"
    merged = lam_df.merge(st[["day","hour","s_opt"]], on=["day","hour"], how="left")
    assert merged["s_opt"].notna().all(), "Missing s_target for some hours"

    summaries = []
    details_by_B = {}
    for B in range(0, args.b_max+1):
        cap = args.n_inhouse + B
        rows = []
        total_extra = 0.0
        for _, r in merged.iterrows():
            lam = float(r["lambda"]); s_t = int(r["s_opt"])
            s_act = min(s_t, cap)
            base = waiting_hours(lam, mu, s_t)
            neww = waiting_hours(lam, mu, s_act)
            extra = max(0.0, neww - base)
            rows.append({"day": r["day"], "hour": int(r["hour"]), "lambda": lam,
                         "s_target": s_t, "s_actual": s_act,
                         "shortage": max(0, s_t - s_act), "extra_wait_hours": extra})
            total_extra += extra
        weekly_cost = args.c_borrow * B
        total_cost = weekly_cost + total_extra
        summaries.append({"borrowed_B": B, "total_cost": total_cost,
                          "weekly_borrow_cost": weekly_cost,
                          "total_extra_wait_hours": total_extra})
        dfB = pd.DataFrame(rows)
        daily = dfB.groupby("day").agg(total_shortage=("shortage","sum"),
                                       extra_wait_hours=("extra_wait_hours","sum")).reset_index()
        details_by_B[B] = {"hourly": dfB, "daily": daily}

    summary = pd.DataFrame(summaries).sort_values("total_cost")
    summary.to_csv("search_curve.csv", index=False)
    bestB = int(summary.iloc[0]["borrowed_B"])
    details_by_B[bestB]["hourly"].to_csv("best_hourly.csv", index=False)
    details_by_B[bestB]["daily"].to_csv("best_daily.csv", index=False)

if __name__ == "__main__":
    main()
