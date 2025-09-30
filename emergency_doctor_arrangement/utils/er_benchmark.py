from dataclasses import dataclass, field
from typing import List, Optional, Dict
import math
import numpy as np

WEEK_HOURS = 7 * 24
DAY_START = 7
NIGHT_START = 0
NIGHT_END = 7
DAY_END = 24
INTERNAL_CAP = 11
DEFAULT_C_BORROW = 20.0

@dataclass(frozen=True)
class Shift:
    start: int
    end: int
    tag: str = "day"
    def __post_init__(self):
        if not (0 <= self.start < self.end <= WEEK_HOURS):
            raise ValueError(f"Invalid shift bounds: [{self.start}, {self.end})")
        if not (float(self.start).is_integer() and float(self.end).is_integer()):
            raise ValueError("Shifts must be integer-hour aligned.")
        if self.tag not in ("day", "night"):
            raise ValueError("Shift tag must be 'day' or 'night'.")

@dataclass
class Doctor:
    id: str
    origin: str = "internal"  # "internal" or "borrowed"
    shifts: List[Shift] = field(default_factory=list)
    def sorted_shifts(self) -> List[Shift]:
        return sorted(self.shifts, key=lambda s: (s.start, s.end))

@dataclass
class BenchmarkInput:
    arrival_rates: List[float]
    mu: float
    doctors: List[Doctor]
    c_borrow: float = DEFAULT_C_BORROW
    seed: int = 42
    include_overtime_in_cost: bool = False

@dataclass
class EvalResult:
    feasible: bool
    violations: List[str]
    total_wait_hours: float
    staff_hours: float
    borrowed_cost: float
    objective: float
    details: Dict

def hour_to_day(hour: int) -> int:
    return hour // 24

def within(h: float, start: float, end: float) -> bool:
    return start <= h < end

def merge_total_hours(shifts: List[Shift]) -> int:
    if not shifts:
        return 0
    slots = []
    for sh in sorted(shifts, key=lambda s: (s.start, s.end)):
        if slots and sh.start <= slots[-1][1]:
            slots[-1][1] = max(slots[-1][1], sh.end)
        else:
            slots.append([sh.start, sh.end])
    return sum(b - a for a, b in slots)

def count_on_duty_by_hour(doctors: List[Doctor]):
    import numpy as _np
    p = _np.zeros(WEEK_HOURS, dtype=int)
    for doc in doctors:
        for sh in doc.shifts:
            for h in range(sh.start, sh.end):
                if 0 <= h < WEEK_HOURS:
                    p[h] += 1
    return p

def validate_schedule(doctors: List[Doctor]) -> List[str]:
    violations = []
    internal_ids = {d.id for d in doctors if d.origin == "internal"}
    if len(internal_ids) > INTERNAL_CAP:
        violations.append(f"(6) Too many internal doctors listed: {len(internal_ids)} > {INTERNAL_CAP}")
    for d in doctors:
        sfts = d.sorted_shifts()
        for i in range(1, len(sfts)):
            if sfts[i-1].end > sfts[i].start:
                violations.append(f"Doctor {d.id}: overlapping shifts [{sfts[i-1].start},{sfts[i-1].end}) and [{sfts[i].start},{sfts[i].end}).")
        day_white_hours = {k: 0 for k in range(7)}
        day_white_spans = {k: [] for k in range(7)}
        night_days = []
        for sh in sfts:
            d0 = hour_to_day(sh.start)
            if sh.tag == "night":
                if not (sh.start % 24 == NIGHT_START and sh.end % 24 == NIGHT_END and sh.end - sh.start == 7):
                    violations.append(f"(7) Doctor {d.id} night shift not exactly 00:00-07:00: [{sh.start},{sh.end}).")
                night_days.append(d0)
            else:
                if not (sh.start % 24 >= DAY_START and sh.end % 24 <= DAY_END and hour_to_day(sh.start) == hour_to_day(sh.end - 1)):
                    violations.append(f"(8) Doctor {d.id} day shift must lie within 07:00-24:00 same day: [{sh.start},{sh.end}).")
                length = sh.end - sh.start
                if not (3 <= length <= 8):
                    violations.append(f"(8) Doctor {d.id} day shift length must be 3..8h: got {length}h at [{sh.start},{sh.end}).")
                day_white_hours[d0] += length
                day_white_spans[d0].append((sh.start, sh.end))
        for day in range(7):
            if day_white_hours[day] > 12:
                violations.append(f"(8) Doctor {d.id} has {day_white_hours[day]} day-hours on day {day}, exceeds 12.")
            if len(day_white_spans[day]) > 2:
                violations.append(f"(9) Doctor {d.id} has >2 day shifts on day {day}.")
            if len(day_white_spans[day]) == 2:
                (a1, b1), (a2, b2) = sorted(day_white_spans[day])
                gap = a2 - b1
                if gap < 2:
                    violations.append(f"(9) Doctor {d.id} two day shifts on day {day} gap {gap}h < 2h.")
        for nd in night_days:
            win_start = nd * 24 - 8
            if win_start < 0:
                win_start = 0
            win_end = nd * 24
            for sh in sfts:
                if sh.end > win_start and sh.start < win_end:
                    violations.append(f"(10) Doctor {d.id} works within 8h before night shift on day {nd}: overlaps [{win_start},{win_end}).")
                    break
        for nd in night_days:
            rest_start = nd * 24 + 7
            rest_end   = (nd + 1) * 24 + 7
            for sh in sfts:
                if sh.end > rest_start and sh.start < rest_end:
                    violations.append(f"(11) Doctor {d.id} works during mandatory 24h rest after night on day {nd}: [{rest_start},{rest_end}).")
                    break
        if len(night_days) > 2:
            violations.append(f"(12) Doctor {d.id} has {len(night_days)} night shifts (>2).")
        has_valid_rest_day = False
        night_end_marks = {nd * 24 + 7 for nd in night_days}
        for dday in range(7):
            w_start = dday * 24 + 7
            w_end   = (dday + 1) * 24 + 7
            intersects = any((sh.end > w_start and sh.start < w_end) for sh in sfts)
            if not intersects:
                if w_start not in night_end_marks:
                    has_valid_rest_day = True
                    break
        if not has_valid_rest_day:
            violations.append(f"(13) Doctor {d.id} lacks an additional 24h rest day (07:00..next 07:00) beyond mandatory post-night rest.")
    coverage = count_on_duty_by_hour(doctors)
    if np.any(coverage == 0):
        zeros = np.where(coverage == 0)[0].tolist()
        violations.append(f"(15) No doctor on duty during hours: {zeros[:10]}{'...' if len(zeros) > 10 else ''}")
    return violations

def generate_arrivals(lambdas: List[float], seed: int = 42) -> np.ndarray:
    if len(lambdas) != WEEK_HOURS:
        raise ValueError(f"arrival_rates must have length {WEEK_HOURS}.")
    rng = np.random.default_rng(seed)
    arrivals = []
    for h, lam in enumerate(lambdas):
        if lam < 0:
            raise ValueError("Arrival rates must be non-negative.")
        n = rng.poisson(lam)
        if n > 0:
            u = rng.random(n)
            arrivals.extend((h + u).tolist())  # fixed
    arrivals = np.array(sorted(arrivals), dtype=float)
    return arrivals

def simulate_and_score(inp: BenchmarkInput):
    violations = validate_schedule(inp.doctors)
    feasible = len(violations) == 0
    if inp.mu <= 0:
        violations.append("Service rate mu must be positive.")
        feasible = False

    arrivals = generate_arrivals(inp.arrival_rates, seed=inp.seed)
    docs = inp.doctors
    n_docs = len(docs)
    next_free = np.zeros(n_docs, dtype=float)
    overtime_sum = np.zeros(n_docs, dtype=float)
    staff_hours_scheduled = sum(merge_total_hours(d.shifts) for d in docs)
    rng = np.random.default_rng(inp.seed + 1)
    doc_shifts_sorted = [d.sorted_shifts() for d in docs]

    def within(h, s, e): return s <= h < e
    def project(i_doc: int, t0: float) -> Optional[float]:
        for sh in doc_shifts_sorted[i_doc]:
            if t0 < sh.start: return float(sh.start)
            if within(t0, sh.start, sh.end): return t0
        return None

    total_wait = 0.0
    jobs = 0

    for a in arrivals:
        best_idx = -1
        best_start = math.inf
        best_shift = None
        for i in range(n_docs):
            t0 = max(a, next_free[i])
            start = project(i, t0)
            if start is None: continue
            if start < best_start:
                best_start = start
                best_idx = i
                best_shift = None
                for s in doc_shifts_sorted[i]:
                    if within(start, s.start, s.end):
                        best_shift = s
                        break
        if best_idx < 0:
            violations.append("No available doctor for an arrival; schedule likely violates (15).")
            feasible = False
            break
        svc = rng.exponential(1.0 / inp.mu)
        finish = best_start + svc
        total_wait += max(0.0, best_start - a)
        next_free[best_idx] = finish
        jobs += 1
        if best_shift is not None and finish > best_shift.end:
            overtime_sum[best_idx] += (finish - best_shift.end)

    staff_hours = float(staff_hours_scheduled)
    if inp.include_overtime_in_cost:
        staff_hours += float(overtime_sum.sum())
    borrowed = sum(1 for d in docs if d.origin == "borrowed" and len(d.shifts) > 0)
    borrow_cost = borrowed * inp.c_borrow
    objective = total_wait + 1.3 * staff_hours + borrow_cost

    details = {
        "arrivals": int(jobs),
        "overtime_by_doctor": {docs[i].id: float(overtime_sum[i]) for i in range(n_docs)},
        "scheduled_hours_by_doctor": {docs[i].id: merge_total_hours(docs[i].shifts) for i in range(n_docs)},
        "borrowed_doctors": borrowed,
    }

    return EvalResult(
        feasible=feasible,
        violations=violations,
        total_wait_hours=float(total_wait),
        staff_hours=float(staff_hours),
        borrowed_cost=float(borrow_cost),
        objective=float(objective),
        details=details
    )

def build_input_from_dict(data: Dict) -> BenchmarkInput:
    arr = data["arrival_rates"]
    mu = float(data["mu"])
    c_borrow = float(data.get("c_borrow", DEFAULT_C_BORROW))
    seed = int(data.get("seed", 42))
    include_ot = bool(data.get("include_overtime_in_cost", False))
    docs: List[Doctor] = []
    for d in data["doctors"]:
        shifts = [Shift(int(s["start"]), int(s["end"]), s.get("tag", "day")) for s in d.get("shifts", [])]
        docs.append(Doctor(id=d["id"], origin=d.get("origin", "internal"), shifts=shifts))
    return BenchmarkInput(arrival_rates=arr, mu=mu, doctors=docs, c_borrow=c_borrow, seed=seed, include_overtime_in_cost=include_ot)
