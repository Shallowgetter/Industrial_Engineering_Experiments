# utils.py
# Common utilities: data loading, Erlang C/queueing formulas, staffing helpers, JSON I/O.

from __future__ import annotations
import json, math
from dataclasses import dataclass
from typing import List, Dict, Any

# ---------- I/O ----------

def load_problem(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Validate minimal schema
    assert "arrival_rates" in data and len(data["arrival_rates"]) % 24 == 0, "arrival_rates must be 24*k length"
    assert "mu" in data, "mu required"
    return data

def dump_json(obj: Dict[str, Any], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ---------- Queueing (M/M/s per hour approximation) ----------

def erlang_c(lam: float, mu: float, s: int) -> float:
    if s <= 0:
        return 1.0
    rho = lam / (mu * s)
    if rho >= 1.0:
        return 1.0
    a = lam / mu
    # compute sum_{n=0}^{s-1} a^n / n!
    sum_terms = 0.0
    term = 1.0
    for n in range(s):
        if n > 0:
            term *= a / n
        sum_terms += term
    term *= a / s
    top = term / (1.0 - rho)
    return top / (sum_terms + top)

def ewq(lam: float, mu: float, s: int) -> float:
    # Expected waiting time in queue (hours)
    if s <= 0:
        return 1e9
    rho = lam / (mu * s)
    if rho >= 1.0:
        return 1e9
    pw = erlang_c(lam, mu, s)
    return pw / (mu * s - lam)

def waiting_hours(lam: float, mu: float, s: int) -> float:
    return lam * ewq(lam, mu, s)

def hour_cost(lam: float, mu: float, s: int, alpha: float) -> float:
    w = ewq(lam, mu, s)
    if w >= 1e8:
        return 1e15
    return lam * w + alpha * s

def optimal_s_for_hour(lam: float, mu: float, alpha: float, headroom: int) -> int:
    s_min = max(1, math.ceil(lam / mu))
    best_s, best_c = None, float("inf")
    for s in range(max(1, s_min - 1), s_min + headroom + 1):
        c = hour_cost(lam, mu, s, alpha)
        if c < best_c:
            best_s, best_c = s, c
    return int(best_s)

# ---------- Helpers on schedules ----------

def hours_active_from_doctors(doctors: List[Dict[str, Any]], horizon_hours: int) -> List[int]:
    """Return capacity (active doctor count) per hour over the horizon."""
    cap = [0] * horizon_hours
    for d in doctors:
        for sh in d.get("shifts", []):
            start, end = int(sh["start"]), int(sh["end"])
            for h in range(max(0, start), min(horizon_hours, end)):
                cap[h] += 1
    return cap

def total_doctor_hours(doctors: List[Dict[str, Any]]) -> int:
    tot = 0
    for d in doctors:
        for sh in d.get("shifts", []):
            tot += int(sh["end"]) - int(sh["start"])
    return tot

def borrowed_doctor_count(doctors: List[Dict[str, Any]]) -> int:
    seen = set()
    for d in doctors:
        if d.get("origin", "").lower() == "borrowed":
            seen.add(d.get("id", ""))
    return len(seen)

def objective_value(arrival_rates: List[float], mu: float, staffing: List[int], include_overtime_in_cost: bool,
                    c_borrow: float, doctors: List[Dict[str, Any]], alpha: float = 1.3) -> Dict[str, float]:
    """Compute the weekly objective using per-hour M/M/s approximation for waiting.
    min( sum_wait_hours + alpha * doctor_hours + c_borrow * (#borrowed unique) )
    """
    assert len(arrival_rates) == len(staffing), "arrival_rates and staffing length mismatch"
    wait_sum = 0.0
    for lam, s in zip(arrival_rates, staffing):
        wait_sum += waiting_hours(lam, mu, s)
    doc_hours = total_doctor_hours(doctors)
    # Overtime not modeled separately here; respect include_overtime_in_cost if needed later.
    borrowed = borrowed_doctor_count(doctors)
    obj = wait_sum + alpha * doc_hours + c_borrow * borrowed
    return {
        "total_wait_hours": wait_sum,
        "total_doctor_hours": float(doc_hours),
        "borrowed_count": float(borrowed),
        "borrow_cost": c_borrow * borrowed,
        "alpha": alpha,
        "objective": obj
    }
