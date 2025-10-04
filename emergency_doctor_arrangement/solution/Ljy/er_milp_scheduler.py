#!/usr/bin/env python3
"""
er_gurobi_scheduler.py

Gurobi-based MILP scheduler for ED weekly scheduling.
- Replaces PuLP version with gurobipy for speed and MIP start.
- Provides:
    * Erlang-C precomputation L[h,k]
    * Candidate shifts generation (white shifts + night shifts)
    * Full individual-doctor MILP with your constraints
    * Greedy heuristic to produce MIP-start
    * Two modes to find minimal borrowed doctors:
        - two-stage (recommended): minimize sum(borrow) then optimize full objective with sum(borrow)==min_borrow
        - binary-search: binary search on allowed borrow count
    * Outputs CSVs: schedule_assignments.csv, borrowed_doctors.csv, coverage_by_hour.csv, solution_summary.csv, L_hk_table.csv
Usage:
    python er_gurobi_scheduler.py --mode two_stage  --time_limit 3600 --k_cover 20 --max_borrow 8
Requirements:
    pip install gurobipy pandas numpy
    (Gurobi license must be available -- see Gurobi docs)
"""

import math, itertools, argparse, sys, time
from collections import defaultdict
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# ---------------------------
# Default Parameters (changeable via CLI)
# ---------------------------
MU = 6.0
BORROW_COST = 20.0
NUM_BASE_DOCTORS = 11
DEFAULT_MAX_BORROW = 8
DEFAULT_K_MAX_COVER = 20
# solver defaults
DEFAULT_TIME_LIMIT = 3600  # seconds for main solves
DEFAULT_FEAS_TIME_LIMIT = 600  # for feasibility checks in binary search
DEFAULT_MIPGAP = 0.005
DEFAULT_FEAS_MIPGAP = 0.02

# Lambda table (7x24) from user
LAMBDA_TABLE = [
    [8.40,5.49,3.88,4.00,4.20,6.73,21.97,43.76,28.18,25.38,25.40,23.26,11.75,11.20,13.07,19.45,18.76,19.69,15.57,29.39,38.76,31.66,15.96,9.61],
    [5.31,3.16,5.61,1.94,4.46,7.56,25.86,39.92,27.97,27.90,19.04,17.43,8.70,19.26,14.74,16.07,19.79,13.16,31.48,33.67,36.39,21.99,13.04,7.73],
    [6.77,4.28,2.43,1.73,6.80,4.62,25.82,44.01,34.12,18.63,29.10,14.03,14.63,16.32,15.15,22.69,24.80,30.30,40.57,37.47,25.07,17.95,7.81,9.71],
    [12.82,6.17,4.79,3.90,4.46,6.26,11.08,33.77,29.86,13.26,17.44,19.74,11.75,19.83,24.87,11.01,16.00,15.43,25.25,33.21,27.62,18.60,13.80,8.71],
    [4.07,7.36,3.91,3.08,2.68,9.09,25.83,47.81,32.02,22.54,19.06,14.15,14.01,18.07,18.00,13.12,13.23,13.56,31.77,32.72,23.57,15.18,7.42,8.33],
    [5.88,6.25,4.00,7.01,2.74,8.51,23.07,42.15,28.01,20.17,26.46,14.40,13.92,18.19,22.79,16.89,25.10,21.46,32.74,23.43,22.21,16.04,14.48,10.71],
    [2.94,3.12,2.00,3.50,1.37,4.26,28.67,49.35,13.56,24.03,21.52,13.27,17.94,22.97,24.58,28.18,33.35,34.37,30.65,31.58,25.98,13.93,7.24,5.35]
]
LAM_168 = [LAMBDA_TABLE[d][h] for d in range(7) for h in range(24)]
HOURS = list(range(168))

# ---------------------------
# helper functions
# ---------------------------
def erlang_c_Wq(lam, mu, c):
    """Erlang-C Wq hours; inf if unstable"""
    if c <= 0:
        return float('inf')
    a = lam / mu
    rho = a / c
    if rho >= 1.0:
        return float('inf')
    sum_terms = 0.0
    term = 1.0
    for n in range(0, c):
        if n == 0:
            term = 1.0
        else:
            term = term * a / n
        sum_terms += term
    a_c_over_cfact = term * a / c
    denom = sum_terms + a_c_over_cfact / (1.0 - rho)
    P_wait = (a_c_over_cfact / (1.0 - rho)) / denom
    Wq = P_wait / (c * mu - lam)
    return Wq

def precompute_L_hk(K_max, mu=MU):
    L_hk = {h: {} for h in HOURS}
    for h in HOURS:
        lam = LAM_168[h]
        for k in range(0, K_max + 1):
            if k == 0:
                L_hk[h][k] = lam * 1e3
            else:
                Wq = erlang_c_Wq(lam, mu, k)
                L_hk[h][k] = lam * Wq if not math.isinf(Wq) else lam * 1e3
    return L_hk

def generate_white_shifts_for_day(day_idx):
    shifts = []
    for start in range(7, 22):  # 7..21
        for dur in range(3, 9):  # 3..8
            end = start + dur
            if end <= 24:
                ghrs = tuple(day_idx * 24 + h for h in range(start, end))
                shifts.append((start, dur, ghrs))
    return shifts

def build_shift_list():
    shift_list = []
    sid = 0
    day_shifts = defaultdict(list)
    for d in range(7):
        for (start, dur, ghrs) in generate_white_shifts_for_day(d):
            shift_list.append((sid, d, start, dur, ghrs, False))  # False => not night
            day_shifts[d].append(sid)
            sid += 1
    # add night shifts (0-7) for each day
    night_shift_ids = {}
    for d in range(7):
        ghrs = tuple(d*24 + h for h in range(0,7))
        shift_list.append((sid, d, 0, 7, ghrs, True))
        night_shift_ids[d] = sid
        day_shifts[d].append(sid)
        sid += 1
    return shift_list, day_shifts, night_shift_ids

# Greedy heuristic to produce a feasible assignment (for MIP start)
def greedy_assignment_from_ct(ct_by_day, night_by_day, num_doctors_total):
    """Return greedy assignments as list of (j, sid)
       - Simple greedy per day: assign night doctors first (same j numbers), then for white hours fill shifts greedily.
       - This is a heuristic used to give MIP start; not guaranteed optimal.
    """
    shift_list_local, day_shifts_local, night_shift_ids_local = build_shift_list()
    # Map shift id -> info
    shift_info = {sid: {"day": d, "start": start, "dur": dur, "ghrs": ghrs, "is_night": is_n} for (sid,d,start,dur,ghrs,is_n) in shift_list_local}
    assignments = []  # tuples (j,sid)
    # doctor ids 1..num_doctors_total
    # simple scheme: assign lowest-numbered doctors as permanent night doctors for each day as needed
    # For uniqueness keep track of per-doctor day assignments to respect daily constraints approximately
    doctor_busy_hours = {j: set() for j in range(1, num_doctors_total+1)}
    doctor_shifts = {j: [] for j in range(1, num_doctors_total+1)}
    next_free_doc = 1
    # First assign night doctors per day
    for d in range(7):
        need = night_by_day[d]
        # choose doctors not yet busy this day
        for i in range(need):
            if next_free_doc > num_doctors_total:
                break
            j = next_free_doc
            nid = night_shift_ids_local[d]
            assignments.append((j, nid))
            doctor_shifts[j].append(nid)
            doctor_busy_hours[j].update(shift_info[nid]["ghrs"])
            next_free_doc += 1
    # Now for white hours, greedy by hour: create required coverage per hour from ct_by_day
    # Build required per global hour
    required = {}
    for d in range(7):
        for h in range(7,24):
            required[d*24+h] = ct_by_day[d][h]
    # For each global hour from 7..23 per day assign available doctors by creating shifts starting at h with longest possible
    for d in range(7):
        for h_local in range(7,24):
            h = d*24 + h_local
            need = required[h]
            while True:
                cur_cov = sum(1 for (doc,sid) in assignments if h in shift_info[sid]["ghrs"])
                if cur_cov >= need:
                    break
                # try to assign a new doctor with a shift starting at h of max duration
                assigned_flag = False
                for dur in range(min(8,24-h_local), 2, -1):
                    # find shift id with start h_local and dur
                    candidate = None
                    for sid in day_shifts_local[d]:
                        inf = shift_info[sid]
                        if not inf["is_night"] and inf["start"] == h_local and inf["dur"] == dur:
                            candidate = sid; break
                    if candidate is None:
                        continue
                    # find a doctor free for those hours
                    allocated = False
                    for j in range(1, num_doctors_total+1):
                        # check daily constraints roughly: up to 2 shifts and up to 12 hours per day
                        day_shift_count = sum(1 for s in doctor_shifts[j] if shift_info[s]["day"]==d and not shift_info[s]["is_night"])
                        day_hours = sum(shift_info[s]["dur"] for s in doctor_shifts[j] if shift_info[s]["day"]==d and not shift_info[s]["is_night"])
                        if day_shift_count >= 2 or day_hours + dur > 12:
                            continue
                        # check overlap
                        if any(g in doctor_busy_hours[j] for g in shift_info[candidate]["ghrs"]):
                            continue
                        # assign
                        assignments.append((j, candidate))
                        doctor_shifts[j].append(candidate)
                        doctor_busy_hours[j].update(shift_info[candidate]["ghrs"])
                        allocated = True
                        break
                    if allocated:
                        assigned_flag = True
                        break
                if not assigned_flag:
                    # no available doctor to assign -> break and leave shortage
                    break
    # convert assignments to dict by shift
    assign_by_shift = defaultdict(list)
    for j,sid in assignments:
        assign_by_shift[sid].append(j)
    return assignments, assign_by_shift

# ---------------------------
# MILP builder using gurobipy
# ---------------------------
def build_model(K_max=DEFAULT_K_MAX_COVER, max_borrow=DEFAULT_MAX_BORROW, time_limit=DEFAULT_TIME_LIMIT):
    # Precompute L_hk
    print("Precomputing L[h,k] with K_max =", K_max)
    L_hk = precompute_L_hk(K_max)
    pd.DataFrame([{"hour":h, "lambda":LAM_168[h], **{f"L_k_{k}":L_hk[h][k] for k in range(K_max+1)}} for h in HOURS]).to_csv("L_hk_table.csv", index=False)
    print("Saved L_hk_table.csv")
    # build shift list
    shift_list, day_shifts, night_shift_ids = build_shift_list()
    shift_info = {sid: {"day": d, "start": start, "dur": dur, "ghrs": ghrs, "is_night": is_n} for (sid,d,start,dur,ghrs,is_n) in shift_list}
    # doctors range
    total_doctors = NUM_BASE_DOCTORS + max_borrow
    doctors = list(range(1, total_doctors+1))
    borrowed_doctors = list(range(NUM_BASE_DOCTORS+1, total_doctors+1))
    # create model
    model = gp.Model("ED_weekly")
    model.setParam('OutputFlag', 1)
    # Variables: z[j,sid] binary
    z = {}
    for j in doctors:
        for (sid, d, start, dur, ghrs, is_n) in shift_list:
            z[j, sid] = model.addVar(vtype=GRB.BINARY, name=f"z_{j}_{sid}")
    # borrow vars
    borrow = {}
    for j in borrowed_doctors:
        borrow[j] = model.addVar(vtype=GRB.BINARY, name=f"borrow_{j}")
    # off vars per doctor per day
    off = {}
    for j in doctors:
        for d in range(7):
            off[j,d] = model.addVar(vtype=GRB.BINARY, name=f"off_{j}_{d}")
    # y[h,k] coverage selection
    y = {}
    for h in HOURS:
        for k in range(0, K_max+1):
            y[h,k] = model.addVar(vtype=GRB.BINARY, name=f"y_{h}_{k}")
    model.update()
    # -----------------------
    # Constraints
    # -----------------------
    # Borrow definition (if any assignment to borrowed doctor -> borrow=1)
    M_big = 1000
    for j in borrowed_doctors:
        model.addConstr(gp.quicksum(z[j,sid] for (sid,_,_,_,_,_) in shift_list) <= M_big * borrow[j], name=f"borrow_def_up_{j}")
        model.addConstr(borrow[j] <= gp.quicksum(z[j,sid] for (sid,_,_,_,_,_) in shift_list), name=f"borrow_def_low_{j}")
    # Off-day: if off[j,d]==1 then no shifts covering [d:7 .. d+1:7) window
    for j in doctors:
        for d in range(7):
            window_start = d*24 + 7
            window_end = (d+1)*24 + 7  # exclusive
            banned_sids = [sid for (sid, dd, start, dur, ghrs, is_n) in shift_list for sid2 in [sid] if any(window_start <= h < window_end for h in shift_info[sid]["ghrs"])]
            # above comprehension double-loop is awkward; we'll build properly:
            banned_sids = []
            for (sid, dd, start, dur, ghrs, is_n) in shift_list:
                if any(window_start <= h < window_end for h in ghrs):
                    banned_sids.append(sid)
            if banned_sids:
                model.addConstr(gp.quicksum(z[j,s] for s in banned_sids) <= M_big * (1 - off[j,d]), name=f"off_def_{j}_{d}")
    # at least one off-day per doctor
    for j in doctors:
        model.addConstr(gp.quicksum(off[j,d] for d in range(7)) >= 1, name=f"one_off_{j}")
    # night constraints: if night assigned then forbid shifts in prior 8h and subsequent 24h and same day white shifts
    for j in doctors:
        for d in range(7):
            nid = night_shift_ids[d]
            night_var = z[j, nid]
            # previously compute forbidden shifts
            prev_day = (d - 1) % 7
            prev_window = set(prev_day*24 + h for h in range(16,24))
            post_window = set(range(d*24 + 7, d*24 + 31))  # d:7 .. d+1:7
            forb = []
            for (sid, dd, start, dur, ghrs, is_n) in shift_list:
                if sid == nid:
                    continue
                if any(h in prev_window for h in ghrs) or any(h in post_window for h in ghrs) or (start >=7 and start < 24 and dd == d):
                    forb.append(sid)
            for sid in forb:
                model.addConstr(z[j,sid] + night_var <= 1, name=f"night_forbid_{j}_{d}_{sid}")
    # daily white constraints: per doctor per day at most 2 white shifts and daily hours <=12
    for j in doctors:
        for d in range(7):
            white_sids = [sid for (sid, dd, start, dur, ghrs, is_n) in shift_list if dd==d and start >=7 and start <24]
            if white_sids:
                model.addConstr(gp.quicksum(z[j,sid] for sid in white_sids) <= 2, name=f"max_2white_{j}_{d}")
                model.addConstr(gp.quicksum(z[j,sid]*shift_info[sid]["dur"] for sid in white_sids) <= 12, name=f"daily_hours_{j}_{d}")
    # gap constraint: for a given doctor, two white shifts same day with gap < 2 or overlap cannot both be assigned
    for j in doctors:
        for d in range(7):
            sids = [sid for (sid, dd, start, dur, ghrs, is_n) in shift_list if dd==d and start >=7]
            for a, b in itertools.combinations(sids, 2):
                a_start = shift_info[a]["start"]; a_end = a_start + shift_info[a]["dur"]
                b_start = shift_info[b]["start"]; b_end = b_start + shift_info[b]["dur"]
                if a_start <= b_start:
                    gap = b_start - a_end
                else:
                    gap = a_start - b_end
                if gap < 2 or not (a_end <= b_start or b_end <= a_start):
                    model.addConstr(z[j,a] + z[j,b] <= 1, name=f"gap_forbid_{j}_{d}_{a}_{b}")
    # per-week night limit <=2
    for j in doctors:
        model.addConstr(gp.quicksum(z[j, night_shift_ids[d]] for d in range(7)) <= 2, name=f"week_night_{j}")
    # at least 1 doctor each hour
    for h in HOURS:
        model.addConstr(gp.quicksum(z[j,sid] for j in doctors for (sid, dd, start, dur, ghrs, is_n) in shift_list if h in ghrs) >= 1, name=f"at_least_one_{h}")
    # no doctor overlap: for each doc and hour sum shifts covering h <=1
    for j in doctors:
        for h in HOURS:
            model.addConstr(gp.quicksum(z[j,sid] for (sid, dd, start, dur, ghrs, is_n) in shift_list if h in ghrs) <= 1, name=f"no_overlap_{j}_{h}")
    # coverage matching: sum_j sum_sh covering h z[j,sh] == sum_k k*y[h,k]
    for h in HOURS:
        lhs = gp.quicksum(z[j,sid] for j in doctors for (sid, dd, start, dur, ghrs, is_n) in shift_list if h in ghrs)
        rhs = gp.quicksum(k * y[h,k] for k in range(0, K_max+1))
        model.addConstr(lhs == rhs, name=f"cover_balance_{h}")
        model.addConstr(gp.quicksum(y[h,k] for k in range(0, K_max+1)) == 1, name=f"cover_choice_{h}")
    # done constraints
    model.update()
    # objective variables: waiting approx + 1.3 * doctor-hours + BORROW_COST * sum(borrow)
    wait_term = gp.quicksum(L_hk[h][k] * y[h,k] for h in HOURS for k in range(0, K_max+1))
    doc_hours_term = gp.quicksum(shift_info[sid]["dur"] * z[j,sid] for j in doctors for (sid, _, _, _, _, _) in shift_list)
    borrow_term = BORROW_COST * gp.quicksum(borrow[j] for j in borrowed_doctors)
    total_obj = wait_term + 1.3 * doc_hours_term + borrow_term
    model.setObjective(total_obj, GRB.MINIMIZE)
    return model, z, y, borrow, off, shift_list, shift_info, day_shifts, night_shift_ids, L_hk, doctors, borrowed_doctors

# ---------------------------
# Two-stage and binary search controllers
# ---------------------------
def find_min_borrow_two_stage(K_max, max_borrow, timelimit, mipgap, greedy_assign_func):
    # Build model but with objective = minimize sum(borrow)
    model, z, y, borrow, off, shift_list, shift_info, day_shifts, night_shift_ids, L_hk, doctors, borrowed_doctors = build_model(K_max, max_borrow, timelimit)
    model.ModelName = "Stage1_min_borrow"
    # change objective to minimize sum(borrow)
    model.setObjective(gp.quicksum(borrow[j] for j in borrowed_doctors), GRB.MINIMIZE)
    # solver params
    model.Params.TimeLimit = timelimit
    model.Params.MIPGap = max(0.0001, mipgap)  # small gap ok
    model.Params.Threads = 8
    # MIP start from greedy
    print("Generating greedy MIP-start...")
    # compute ct_by_day and night_by_day for greedy (we need coverage demands per hour currently derived from L_hk picking k that corresponds to required c_t ???)
    # For MIP start we can use a simpler greedy based on per-hour demand estimated by prior c_t or use estimate: choose k that minimizes L_hk (not necessary).
    # We'll construct a simple greedy: for each day, set night doctors as previous heuristics: use  max k for hours 0..6 that gives Wq<=1
    # Simpler: produce a naive assignment: assign default night doctors = ceil(max lam (0..6)/MU) etc.
    # For brevity, we'll build greedy using earlier greedy heuristic using ct_by_day derived from L_hk pick minimal k with L_hk<=lam*1 (Wq<=1)
    K = K_max
    # derive ct_by_day
    ct_by_day = {d:{} for d in range(7)}
    night_by_day = {}
    for d in range(7):
        # hours 0..6 choose minimal k so that Wq<=1 (approx using L_hk -> Wq = L_hk/lam)
        # but we precomputed L_hk; convert back to Wq = L_hk/lam
        candidates = []
        # find minimal c that yields Wq <= 1 for hours 0..6 simultaneously -> brute force
        chosen_c = None
        for c in range(1, K+1):
            ok = True
            for h_local in range(0,7):
                h = d*24 + h_local
                lam = LAM_168[h]
                Wq = (L_hk[h][c]/lam) if lam>0 else 0
                if Wq > 1.0:
                    ok = False; break
            if ok:
                chosen_c = c; break
        if chosen_c is None:
            chosen_c = min(K, max(1, math.ceil(max(LAM_168[d*24 + h] for h in range(0,7))/MU) ))  # fallback
        night_by_day[d] = chosen_c
        # for white hours 7..23 choose minimal k s.t. Wq<=1
        for h_local in range(7,24):
            h = d*24 + h_local
            chosen = None
            for c in range(1, K+1):
                lam = LAM_168[h]
                Wq = (L_hk[h][c]/lam) if lam>0 else 0
                if Wq <= 1.0:
                    chosen = c; break
            if chosen is None: chosen = K
            ct_by_day[d][h_local] = chosen
    # produce greedy assignment
    assignments, assign_by_shift = greedy_assign_func(ct_by_day, night_by_day, NUM_BASE_DOCTORS + max_borrow)
    # set starts for z variables
    print("Applying MIP start from greedy heuristic (assignments count:", len(assignments), ")")
    for (j,sid) in assignments:
        var = z.get((j,sid))
        if var is not None:
            var.Start = 1.0
    # Optimize
    print("Optimizing stage 1 (minimize borrow)...")
    model.optimize()
    status = model.Status
    if status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.FEASIBLE]:
        # get minimal borrow value (if feasible)
        min_borrow = int(round(model.ObjVal)) if model.Status==GRB.OPTIMAL else None
        # better: fetch sum(borrow) value if feasible
        if model.SolCount > 0:
            min_borrow = int(sum(int(borrow[j].X) for j in borrowed_doctors))
        else:
            min_borrow = None
    else:
        min_borrow = None
    print("Stage1 result status:", model.Status, "min_borrow:", min_borrow)
    return min_borrow, model, (z,y,borrow,off,shift_list,shift_info,day_shifts,night_shift_ids,L_hk,doctors,borrowed_doctors)

def optimize_with_fixed_borrow(existing_model_pack, fixed_borrow, timelimit, mipgap):
    # existing_model_pack contains model objects returned from build_model inside stage1
    model, z, y, borrow, off, shift_list, shift_info, day_shifts, night_shift_ids, L_hk, doctors, borrowed_doctors = existing_model_pack
    # add constraint sum(borrow) == fixed_borrow
    model.addConstr(gp.quicksum(borrow[j] for j in borrowed_doctors) == fixed_borrow, name="fix_borrow_count")
    # set objective back to full objective if it was changed (it was changed in build_model to full object already; in stage1 we set actual objective to min borrow)
    # but in our implementation build_model set full objective first, then stage1 replaced; here we assume model currently has full objective in model.getObjective? To be safe, reassign:
    wait_term = gp.quicksum(L_hk[h][k] * y[h,k] for h in HOURS for k in L_hk[h].keys())
    doc_hours_term = gp.quicksum(shift_info[sid]["dur"] * z[j,sid] for j in doctors for (sid, _, _, _, _, _) in shift_list)
    borrow_term = BORROW_COST * gp.quicksum(borrow[j] for j in borrowed_doctors)
    model.setObjective(wait_term + 1.3 * doc_hours_term + borrow_term, GRB.MINIMIZE)
    model.Params.TimeLimit = timelimit
    model.Params.MIPGap = mipgap
    model.Params.Threads = 8
    print("Optimizing stage2 with fixed borrow =", fixed_borrow)
    model.optimize()
    return model

def binary_search_min_borrow(K_max, max_borrow, feas_time_limit, feas_mipgap, greedy_assign_func):
    # binary search LB..UB on borrow count where feasibility means model has feasible solution within time limit
    LB = 0
    UB = max_borrow
    found = None
    last_feasible_model = None
    while LB <= UB:
        mid = (LB + UB) // 2
        print(f"Binary search testing mid borrow = {mid}")
        # build model fresh every iteration (costly), but safer
        model, z, y, borrow, off, shift_list, shift_info, day_shifts, night_shift_ids, L_hk, doctors, borrowed_doctors = build_model(K_max, max_borrow)
        # add sum(borrow) <= mid constraint
        model.addConstr(gp.quicksum(borrow[j] for j in borrowed_doctors) <= mid, name="borrow_leq_mid")
        # set objective maybe trivial to speed up (minimize doctor hours) but here just use full objective
        model.Params.TimeLimit = feas_time_limit
        model.Params.MIPGap = feas_mipgap
        model.Params.Threads = 8
        # apply MIP start (greedy)
        print("Applying greedy MIP-start for feasibility check...")
        # derive ct_by_day and night_by_day as earlier (simple)
        # (reuse greedy_assign_func usage)
        ct_by_day = {d:{} for d in range(7)}
        night_by_day = {}
        K = K_max
        for d in range(7):
            chosen_c = None
            for c in range(1, K+1):
                ok = True
                for h_local in range(0,7):
                    h = d*24 + h_local
                    lam = LAM_168[h]
                    Wq = (L_hk[h][c]/lam) if lam>0 else 0
                    if Wq > 1.0:
                        ok = False; break
                if ok:
                    chosen_c = c; break
            if chosen_c is None:
                chosen_c = min(K, max(1, math.ceil(max(LAM_168[d*24 + h] for h in range(0,7))/MU) ))
            night_by_day[d] = chosen_c
            for h_local in range(7,24):
                h = d*24 + h_local
                chosen = None
                for c in range(1, K+1):
                    lam = LAM_168[h]
                    Wq = (L_hk[h][c]/lam) if lam>0 else 0
                    if Wq <= 1.0:
                        chosen = c; break
                if chosen is None: chosen = K
                ct_by_day[d][h_local] = chosen
        assignments, assign_by_shift = greedy_assign_func(ct_by_day, night_by_day, NUM_BASE_DOCTORS + max_borrow)
        for (j,sid) in assignments:
            var = z.get((j,sid))
            if var is not None:
                var.Start = 1.0
        print("Running solver for feasibility...")
        model.optimize()
        status = model.Status
        if status in [GRB.OPTIMAL, GRB.FEASIBLE, GRB.TIME_LIMIT] and model.SolCount>0:
            # feasible
            found = mid
            last_feasible_model = model
            UB = mid - 1
            print(f"Feasible with mid={mid}, trying lower")
        else:
            LB = mid + 1
            print(f"Infeasible with mid={mid}, trying higher")
    if found is None:
        return None, None
    else:
        return found, last_feasible_model

# ---------------------------
# Utilities to extract outputs from model
# ---------------------------
def extract_solution_and_write_csv(model, z, borrow, shift_list, shift_info, doctors, borrowed_doctors, prefix="solution"):
    # get assignments
    assign_rows = []
    for (j,sid), var in z.items():
        val = var.X if var is not None else 0
        if val > 0.5:
            si = shift_info[sid]
            assign_rows.append({"doctor": int(j), "day": int(si["day"]), "start": int(si["start"]), "dur": int(si["dur"]), "shift_id": sid, "is_night": bool(si["start"]==0)})
    df_assign = pd.DataFrame(assign_rows)
    df_assign.to_csv(f"{prefix}_schedule_assignments.csv", index=False)
    # borrowed
    borrow_rows = []
    for j in borrowed_doctors:
        val = int(borrow[j].X) if borrow[j].X is not None else 0
        borrow_rows.append({"doctor": j, "borrowed": val})
    pd.DataFrame(borrow_rows).to_csv(f"{prefix}_borrowed_doctors.csv", index=False)
    # coverage per hour (y)
    cover_rows = []
    # reconstruct y vars via model.getVarByName? We passed y as dict earlier; assume available in closure
    # We'll attempt to access y through model.getAttr
    # safer: we expect model to have variables named y_h_k; iterate variables to find those names
    y_vars = {v.VarName: v for v in model.getVars() if v.VarName.startswith("y_")}
    # each y var name is y_h_k
    for h in HOURS:
        chosen_k = None
        for k in range(0, max_k_global+1):
            name = f"y_{h}_{k}"
            v = y_vars.get(name)
            if v is not None and v.X > 0.5:
                chosen_k = k
                break
        cover_rows.append({"hour": h, "lambda": LAM_168[h], "chosen_k": chosen_k, "L_hk": L_hk_global[h][chosen_k] if chosen_k is not None else None})
    pd.DataFrame(cover_rows).to_csv(f"{prefix}_coverage_by_hour.csv", index=False)
    # summary
    total_borrowed = sum(int(borrow[j].X) for j in borrowed_doctors)
    total_doc_hours = 0
    for (j,sid), var in z.items():
        if var.X > 0.5:
            total_doc_hours += shift_info[sid]["dur"]
    total_waiting_hours_est = sum((L_hk_global[h][int(cover_rows[h]["chosen_k"])] if cover_rows[h]["chosen_k"] is not None else 0) for h in range(168))
    summary = {"objective": model.ObjVal, "borrowed_doctors": int(total_borrowed), "total_doc_hours": int(total_doc_hours), "estimated_wait_hours": float(total_waiting_hours_est)}
    pd.DataFrame([summary]).to_csv(f"{prefix}_solution_summary.csv", index=False)
    print(f"Wrote {prefix}_*.csv files")

# ---------------------------
# Main entry: CLI and orchestration
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gurobi ED weekly scheduler")
    parser.add_argument("--mode", choices=["two_stage", "binary_search"], default="two_stage", help="Search mode")
    parser.add_argument("--k_cover", type=int, default=DEFAULT_K_MAX_COVER, help="K_MAX_COVER")
    parser.add_argument("--max_borrow", type=int, default=DEFAULT_MAX_BORROW, help="MAX_BORROW")
    parser.add_argument("--time_limit", type=int, default=DEFAULT_TIME_LIMIT, help="Time limit for main solve (s)")
    parser.add_argument("--feas_time_limit", type=int, default=DEFAULT_FEAS_TIME_LIMIT, help="Time limit for feasibility checks (binary search) (s)")
    parser.add_argument("--mipgap", type=float, default=DEFAULT_MIPGAP, help="MIPGap for final solves")
    parser.add_argument("--feas_mipgap", type=float, default=DEFAULT_FEAS_MIPGAP, help="MIPGap for feasibility checks")
    args = parser.parse_args()

    K_max = args.k_cover
    max_borrow = args.max_borrow
    timelimit = args.time_limit
    feas_timelimit = args.feas_time_limit
    mipgap = args.mipgap
    feas_mipgap = args.feas_mipgap

    print("Starting with K_max =", K_max, "max_borrow =", max_borrow, "mode =", args.mode)
    # build base L_hk and shift list globally for output access
    L_hk_global = precompute_L_hk(K_max)
    max_k_global = K_max

    if args.mode == "two_stage":
        # Stage1: find min borrow
        min_borrow, stage1_model, pack = find_min_borrow_two_stage(K_max, max_borrow, timelimit, feas_mipgap, greedy_assignment_from_ct)
        if min_borrow is None:
            print("Stage1 failed to find feasible solution within time limit/params.")
            sys.exit(1)
        print("Minimal borrow found (stage1) =", min_borrow)
        # Stage2: fix borrow and optimize original objective (we have pack)
        # pack returned (model, z, y, borrow, ...) but stage1 replaced objective; we need to reuse or rebuild model; easiest to rebuild
        model2, z2, y2, borrow2, off2, shift_list2, shift_info2, day_shifts2, night_shift_ids2, L_hk2, doctors2, borrowed_doctors2 = build_model(K_max, max_borrow, timelimit)
        # add fix borrow constraint
        model2.addConstr(gp.quicksum(borrow2[j] for j in borrowed_doctors2) == min_borrow, name="fix_borrow_stage2")
        # set full objective (already inside build_model) - it's set; ensure params
        model2.Params.TimeLimit = timelimit
        model2.Params.MIPGap = mipgap
        model2.Params.Threads = 8
        # produce greedy MIP start again
        print("Generating greedy MIP-start for stage2...")
        # compute ct_by_day and night_by_day again (same as stage1)
        # (we replicate stage1's simple ct_by_day that the greedy uses)
        # For brevity, we re-use the greedy function to produce assignments
        ct_by_day = {d:{} for d in range(7)}
        night_by_day = {}
        K = K_max
        for d in range(7):
            chosen_c = None
            for c in range(1, K+1):
                ok = True
                for h_local in range(0,7):
                    h = d*24 + h_local
                    lam = LAM_168[h]
                    Wq = (L_hk_global[h][c]/lam) if lam>0 else 0
                    if Wq > 1.0:
                        ok = False; break
                if ok:
                    chosen_c = c; break
            if chosen_c is None:
                chosen_c = min(K, max(1, math.ceil(max(LAM_168[d*24 + h] for h in range(0,7))/MU) ))
            night_by_day[d] = chosen_c
            for h_local in range(7,24):
                h = d*24 + h_local
                chosen = None
                for c in range(1, K+1):
                    lam = LAM_168[h]
                    Wq = (L_hk_global[h][c]/lam) if lam>0 else 0
                    if Wq <= 1.0:
                        chosen = c; break
                if chosen is None: chosen = K
                ct_by_day[d][h_local] = chosen
        assignments, assign_by_shift = greedy_assignment_from_ct(ct_by_day, night_by_day, NUM_BASE_DOCTORS + max_borrow)
        for (j,sid) in assignments:
            var = z2.get((j,sid))
            if var is not None:
                var.Start = 1.0
        print("Optimizing stage2 (final objective)...")
        model2.optimize()
        # extract and save solution
        print("Stage2 status:", model2.Status)
        # expose global references for extraction
        z_used = z2; borrow_used = borrow2; shift_info_used = {sid: {"day":d,"start":start,"dur":dur,"ghrs":ghrs} for (sid,d,start,dur,ghrs,_) in shift_list2}
        # set globals used by extract function
        globals()['l_hk_table'] = L_hk_global
        globals()['max_k_global'] = K_max
        globals()['shift_info'] = shift_info_used
        globals()['shift_list'] = shift_list2
        globals()['z_vars'] = z_used
        globals()['borrow_vars'] = borrow_used
        # write outputs (simple)
        assign_rows = []
        for (j,sid), var in z_used.items():
            if var.X > 0.5:
                si = shift_info_used[sid]
                assign_rows.append({"doctor": int(j), "day": int(si["day"]), "start": int(si["start"]), "dur": int(si["dur"]), "shift_id": sid})
        pd.DataFrame(assign_rows).to_csv("stage2_schedule_assignments.csv", index=False)
        borrow_rows = [{"doctor": j, "borrowed": int(borrow_used[j].X)} for j in borrowed_doctors2]
        pd.DataFrame(borrow_rows).to_csv("stage2_borrowed_doctors.csv", index=False)
        print("Saved stage2 CSV files.")
    else:
        # binary search path
        found, model_feas = binary_search_min_borrow(K_max, max_borrow, feas_timelimit, feas_mipgap, greedy_assignment_from_ct)
        if found is None:
            print("Binary search could not find feasible borrow count up to max_borrow.")
            sys.exit(1)
        print("Binary result minimal borrow:", found)
        # with model_feas (last feasible model), re-optimize full objective with sum(borrow)==found
        # rebuild fresh model to be safe and optimize final objective
        model3, z3, y3, borrow3, off3, shift_list3, shift_info3, day_shifts3, night_shift_ids3, L_hk3, doctors3, borrowed_doctors3 = build_model(K_max, max_borrow, timelimit)
        model3.addConstr(gp.quicksum(borrow3[j] for j in borrowed_doctors3) == found, name="fix_borrow_bin")
        model3.Params.TimeLimit = timelimit
        model3.Params.MIPGap = mipgap
        model3.Params.Threads = 8
        # apply greedy MIP-start
        ct_by_day = {d:{} for d in range(7)}
        night_by_day = {}
        K = K_max
        for d in range(7):
            chosen_c = None
            for c in range(1, K+1):
                ok = True
                for h_local in range(0,7):
                    h = d*24 + h_local
                    lam = LAM_168[h]
                    Wq = (L_hk_global[h][c]/lam) if lam>0 else 0
                    if Wq > 1.0:
                        ok = False; break
                if ok:
                    chosen_c = c; break
            if chosen_c is None:
                chosen_c = min(K, max(1, math.ceil(max(LAM_168[d*24 + h] for h in range(0,7))/MU) ))
            night_by_day[d] = chosen_c
            for h_local in range(7,24):
                h = d*24 + h_local
                chosen = None
                for c in range(1, K+1):
                    lam = LAM_168[h]
                    Wq = (L_hk_global[h][c]/lam) if lam>0 else 0
                    if Wq <= 1.0:
                        chosen = c; break
                if chosen is None: chosen = K
                ct_by_day[d][h_local] = chosen
        assignments, assign_by_shift = greedy_assignment_from_ct(ct_by_day, night_by_day, NUM_BASE_DOCTORS + max_borrow)
        for (j,sid) in assignments:
            var = z3.get((j,sid))
            if var is not None:
                var.Start = 1.0
        print("Optimizing final model after binary search...")
        model3.optimize()
        # output CSVs
        assign_rows = []
        for (j,sid), var in z3.items():
            if var.X > 0.5:
                si = shift_info3[sid]
                assign_rows.append({"doctor": int(j), "day": int(si["day"]), "start": int(si["start"]), "dur": int(si["dur"]), "shift_id": sid})
        pd.DataFrame(assign_rows).to_csv("binary_schedule_assignments.csv", index=False)
        borrow_rows = [{"doctor": j, "borrowed": int(borrow3[j].X)} for j in borrowed_doctors3]
        pd.DataFrame(borrow_rows).to_csv("binary_borrowed_doctors.csv", index=False)
        print("Saved binary-search CSV files.")
    print("Done.")

