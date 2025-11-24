import math
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

# -----------------------------
# Input data
# -----------------------------
mu = 6.0
lambda_data = [
    [8.40, 5.49, 3.88, 4.00, 4.20, 6.73, 21.97, 43.76, 28.18, 25.38, 25.40, 23.26,
     11.75, 11.20, 13.07, 19.45, 18.76, 19.69, 15.57, 29.39, 38.76, 31.66, 15.96, 9.61],
    [5.31, 3.16, 5.61, 1.94, 4.46, 7.56, 25.86, 39.92, 27.97, 27.90, 19.04, 17.43,
     8.70, 19.26, 14.74, 16.07, 19.79, 13.16, 31.48, 33.67, 36.39, 21.99, 13.04, 7.73],
    [6.77, 4.28, 2.43, 1.73, 6.80, 4.62, 25.82, 44.01, 34.12, 18.63, 29.10, 14.03,
     14.63, 16.32, 15.15, 22.69, 24.80, 30.30, 40.57, 37.47, 25.07, 17.95, 7.81, 9.71],
    [12.82, 6.17, 4.79, 3.90, 4.46, 6.26, 11.08, 33.77, 29.86, 13.26, 17.44, 19.74,
     11.75, 19.83, 24.87, 11.01, 16.00, 15.43, 25.25, 33.21, 27.62, 18.60, 13.80, 8.71],
    [4.07, 7.36, 3.91, 3.08, 2.68, 9.09, 25.83, 47.81, 32.02, 22.54, 19.06, 14.15,
     14.01, 18.07, 18.00, 13.12, 13.23, 13.56, 31.77, 32.72, 23.57, 15.18, 7.42, 8.33],
    [5.88, 6.25, 4.00, 7.01, 2.74, 8.51, 23.07, 42.15, 28.01, 20.17, 26.46, 14.40,
     13.92, 18.19, 22.79, 16.89, 25.10, 21.46, 32.74, 23.43, 22.21, 16.04, 14.48, 10.71],
    [2.94, 3.12, 2.00, 3.50, 1.37, 4.26, 28.67, 49.35, 13.56, 24.03, 21.52, 13.27,
     17.94, 22.97, 24.58, 28.18, 33.35, 34.37, 30.65, 31.58, 25.98, 13.93, 7.24, 5.35]
]
# weekly hours
HOURS_PER_DAY = 24
DAYS = 7
TOTAL_HOURS = HOURS_PER_DAY * DAYS

# Cost weights
WAIT_WEIGHT = 1.0
WORK_WEIGHT = 1.3
SECONDMENT_COST = 20.0

BASE_DOCTORS = 11  # existing ED doctors
MAX_NIGHT_PER_WEEK = 2

# Random seed for reproducibility
random.seed(123)
np.random.seed(123)

# -----------------------------
# Utilities: time indexing
# -----------------------------
def dh_to_t(d: int, h: int) -> int:
    return d * HOURS_PER_DAY + h

def t_to_dh(t: int) -> Tuple[int, int]:
    return t // HOURS_PER_DAY, t % HOURS_PER_DAY

def flat_lambda() -> np.ndarray:
    arr = np.zeros(TOTAL_HOURS, dtype=float)
    for d in range(DAYS):
        for h in range(HOURS_PER_DAY):
            arr[dh_to_t(d, h)] = lambda_data[d][h]
    return arr

LAMBDA = flat_lambda()

# -----------------------------
# Schedule representation
# -----------------------------
class DoctorSchedule:
    def __init__(self):
        self.nights = [False] * DAYS
        # day_shifts[d] is list of (start_hour, end_hour) with start in [7,23], end in (start+3..24], end exclusive
        self.day_shifts = [[] for _ in range(DAYS)]
        # A "rest day" flag meaning from day d 7:00 to day d+1 7:00 no work
        self.rest_day = [False] * DAYS

    def clone(self):
        c = DoctorSchedule()
        c.nights = list(self.nights)
        c.day_shifts = [list(s) for s in self.day_shifts]
        c.rest_day = list(self.rest_day)
        return c

def hours_matrix_of_doctor(doc: DoctorSchedule) -> np.ndarray:
    on = np.zeros(TOTAL_HOURS, dtype=int)
    for d in range(DAYS):
        # night 0-6 if active (we treat night as 0..6 inclusive of hours 0..6)
        if doc.nights[d]:
            for h in range(0, 7):
                on[dh_to_t(d, h)] = 1
        # day shifts 7-24
        for (s, e) in doc.day_shifts[d]:
            # ensure bounds
            s_clamped = max(7, min(23, int(s)))
            e_clamped = max(s_clamped+3, min(24, int(e)))
            for h in range(s_clamped, e_clamped):
                on[dh_to_t(d, h)] = 1
        # rest day enforcement not here (we expect rest_day to correspond to no shifts)
    return on

# -----------------------------
# Feasibility checks and repair
# -----------------------------
def enforce_constraints(doc: DoctorSchedule):
    # Ensure day shifts in [7,24], each segment length in [3,8], no overlaps, gap >=2 if two shifts, total <=12
    for d in range(DAYS):
        repaired = []
        for (s, e) in doc.day_shifts[d]:
            # convert to ints and clamp to local day hours
            s = int(s)
            e = int(e)
            # clamp to [7,24], but preserve minimal length
            s = max(7, min(23, s))
            # ensure e at least s+3
            e = max(s + 3, e)
            e = min(24, e)
            # cap single segment length to 8
            if e - s > 8:
                e = s + 8
            # final safety
            if e - s < 3:
                s = max(7, e - 3)
                e = s + 3
            repaired.append((s, e))
        # sort and merge overlaps carefully, but ensure no single merged span >8
        repaired.sort()
        merged = []
        for (s, e) in repaired:
            if not merged:
                merged.append([s, e])
            else:
                ps, pe = merged[-1]
                if s <= pe:  # overlap or touch -> merge but cap to max length 8
                    new_s = ps
                    new_e = max(pe, e)
                    # If merging would create a segment >8, try to split preserving first segment length
                    if new_e - new_s > 8:
                        # keep first segment as ps..ps+8, push remainder to next if >=3
                        first_end = ps + 8
                        rem_start = first_end + 2  # leave gap 2 to next if possible
                        rem_len = new_e - rem_start
                        merged[-1][1] = first_end
                        if rem_len >= 3 and rem_start < 24:
                            rem_end = min(24, rem_start + rem_len)
                            merged.append([rem_start, rem_end])
                        # else drop overflow
                    else:
                        merged[-1][1] = new_e
                else:
                    merged.append([s, e])
        # ensure at most 2 shifts and gap >=2 between them
        if len(merged) > 2:
            # prefer keeping earlier shifts up to capacity; trim extras
            merged = merged[:2]
        if len(merged) == 2:
            if merged[1][0] - merged[0][1] < 2:
                # shift second to ensure gap 2 if possible, else drop second
                gap_needed = 2 - (merged[1][0] - merged[0][1])
                new_start = min(merged[1][0] + gap_needed, 24 - 3)
                new_end = new_start + (merged[1][1] - merged[1][0])
                new_end = min(24, new_end)
                if new_end - new_start >= 3:
                    merged[1][0], merged[1][1] = new_start, new_end
                else:
                    merged = [merged[0]]
        # enforce total hours cap 12 (reduce last shift first)
        total = sum(e - s for s, e in merged)
        while total > 12 and merged:
            s, e = merged[-1]
            can_reduce = (e - s) - 3
            reduce_by = min(total - 12, can_reduce)
            if reduce_by > 0:
                merged[-1][1] = e - reduce_by
            else:
                merged.pop()
            total = sum(e - s for s, e in merged)
        # final clamp to ints and ensure start>=7 end<=24
        cleaned = []
        for s, e in merged:
            s = int(max(7, min(23, s)))
            e = int(min(24, max(s+3, e)))
            if e - s >= 3:
                cleaned.append((s, e))
        doc.day_shifts[d] = cleaned

    # Night shift rules:
    # - if night[d] = True, then previous day d-1 hours 16..23 must be free, and after night must rest 24h (day d:7..23 and day d+1:0..6 must be free)
    # - at most MAX_NIGHT_PER_WEEK nights
    nights = list(doc.nights)
    # create on-hours snapshot BEFORE modifying nights (to check conflicts)
    pre_on = hours_matrix_of_doctor(doc)

    for d in range(DAYS):
        if nights[d]:
            prev = (d - 1) % DAYS
            # if prev day has any hours in 16..23 -> cannot keep night[d]
            conflict_pre = any(pre_on[dh_to_t(prev, h)] == 1 for h in range(16, 24))
            # if day d has any day_shifts in 7..23 -> conflict post
            conflict_post = any(pre_on[dh_to_t(d, h)] == 1 for h in range(7, 24)) and any(h >=7 and pre_on[dh_to_t(d,h)]==1 for h in range(7,24))
            # also next day early hours should be empty (but if there are nights next day those are separate)
            nxt = (d + 1) % DAYS
            conflict_post = conflict_post or any(pre_on[dh_to_t(nxt, h)] == 1 for h in range(0, 7))
            if conflict_pre or conflict_post:
                nights[d] = False

    # enforce max nights per week
    if sum(nights) > MAX_NIGHT_PER_WEEK:
        idxs = [i for i, v in enumerate(nights) if v]
        keep = set(idxs[:MAX_NIGHT_PER_WEEK])
        nights = [nights[i] if i in keep else False for i in range(DAYS)]
    doc.nights = nights

    # after final nights, enforce clearing day_shifts on night days and the 24h rest window
    for d in range(DAYS):
        if doc.nights[d]:
            # clear any white shifts on day d (7..23)
            doc.day_shifts[d] = []
            # also ensure the next day early hours 0..6 have no white shifts (they are already only for day_shifts of next day, so ensure next day's day_shifts do not include 0..6)
            # Because day_shifts are constrained to [7,24), there's nothing to do for 0..6 except ensure next day's nights false if needed
            doc.rest_day[(d + 0) % DAYS] = False  # this day is not a voluntary rest day, it's a post-night enforced rest

    # enforce at least one rest day: choose a day with minimal white hours and no night adjacency conflicts
    def mark_rest_day(doc: DoctorSchedule):
        best_d = None
        best_load = 1e9
        for d in range(DAYS):
            # rest day must not conflict with nights on d or d-1 or d+1 (we require full 7:00..next7:00 free)
            if doc.nights[d] or doc.nights[(d+1) % DAYS] or doc.nights[(d-1) % DAYS]:
                continue
            load = sum(e - s for s, e in doc.day_shifts[d])
            if load < best_load:
                best_load = load
                best_d = d
        if best_d is None:
            # fallback: pick day with minimal white hours ignoring nights
            loads = [sum(e - s for s, e in doc.day_shifts[d]) for d in range(DAYS)]
            best_d = int(np.argmin(loads))
        doc.rest_day = [False]*DAYS
        doc.rest_day[best_d] = True
        # clear day best_d (7..23)
        doc.day_shifts[best_d] = []
        # ensure next day has no night that would break the 24h window
        doc.nights[(best_d+1) % DAYS] = False
        return best_d

    _ = mark_rest_day(doc)

    return doc

# -----------------------------
# Random schedule generators
# -----------------------------
def random_white_shifts_for_day() -> List[Tuple[int, int]]:
    # 0, 1, or 2 shifts with probability
    k = random.choices([0,1,2], weights=[0.4,0.4,0.2])[0]
    shifts = []
    hours_left = 12
    last_end = None
    for i in range(k):
        length = random.randint(3, min(8, hours_left))
        # choose start in 7..(24-length)
        start = random.randint(7, 24 - length)
        end = start + length
        if last_end is not None and start - last_end < 2:
            delta = 2 - (start - last_end)
            start = min(start + delta, 24 - length)
            end = start + length
        shifts.append((start, end))
        hours_left -= length
        last_end = end
        if hours_left < 3:
            break
    shifts.sort()
    return shifts

def random_doctor_schedule() -> DoctorSchedule:
    doc = DoctorSchedule()
    # random nights up to 2 (choose 0..2 days)
    night_count = random.randint(0, min(2, DAYS))
    if night_count > 0:
        picks = random.sample(range(DAYS), night_count)
        for d in picks:
            doc.nights[d] = True
    for d in range(DAYS):
        if not doc.nights[d]:
            doc.day_shifts[d] = random_white_shifts_for_day()
        else:
            doc.day_shifts[d] = []
    enforce_constraints(doc)
    return doc

# -----------------------------
# Decode team schedule to hourly staffing
# -----------------------------
def staffing_from_team(team: List[DoctorSchedule]) -> np.ndarray:
    staff = np.zeros(TOTAL_HOURS, dtype=int)
    for doc in team:
        staff += hours_matrix_of_doctor(doc)
    # DO NOT clamp to 1 here — we must fix staffing via repair logic, not by artificial clamping.
    return staff

# -----------------------------
# Queue evaluator (fluid Mt/M/pt approximation)
# -----------------------------
def evaluate_schedule(team: List[DoctorSchedule], seconded_count: int) -> Dict:
    staff = staffing_from_team(team)
    # arrivals per hour
    A = LAMBDA.copy()
    Q = 0.0
    total_wait = 0.0
    served_total = 0.0
    for t in range(TOTAL_HOURS):
        arrival = A[t]
        service_capacity = mu * max(0, staff[t])
        # waiting people at the start of hour Q wait this full hour
        total_wait += Q
        demand = Q + arrival
        served = min(demand, service_capacity)
        served_total += served
        Q = max(0.0, demand - service_capacity)
    total_work_hours = int(np.sum(staff))
    cost = WAIT_WEIGHT * total_wait + WORK_WEIGHT * total_work_hours + SECONDMENT_COST * seconded_count
    return {
        "cost": cost,
        "total_wait": total_wait,
        "total_work_hours": total_work_hours,
        "seconded": seconded_count,
        "staffing": staff,
        "served_total": served_total,
        "ending_queue": Q
    }

# -----------------------------
# Helper: check if we can add a white shift into doc on day d covering hour h
# -----------------------------
def can_add_white_shift(doc: DoctorSchedule, d:int, start:int, end:int) -> bool:
    # start,end in local day hours [7,24]
    # check length bounds
    length = end - start
    if length < 3 or length > 8:
        return False
    # check no overlap with existing day_shifts
    for (s,e) in doc.day_shifts[d]:
        if not (end <= s or start >= e):
            return False
    # check total day hours would not exceed 12
    total = sum(e - s for s, e in doc.day_shifts[d]) + length
    if total > 12:
        return False
    # check night conflicts: cannot be night on this day or adjacent nights that would violate pre/post constraints
    if doc.nights[d] or doc.nights[(d-1)%DAYS] or doc.nights[(d+1)%DAYS]:
        # if night present nearby, adding a white shift may violate 16..23 pre-night rule or post-night rest
        # disallow in conservative approach
        return False
    # gap constraints: if existing shift and this new would be second shift, require gap >=2
    shifts = sorted(doc.day_shifts[d] + [(start,end)])
    if len(shifts) >= 2:
        # check adjacent gap
        for i in range(len(shifts)-1):
            if shifts[i+1][0] - shifts[i][1] < 2:
                return False
    return True

# -----------------------------
# Genetic Algorithm optimizer
# -----------------------------
class GAOptimizer:
    def __init__(self, base_doctors=BASE_DOCTORS, max_seconded=10, pop_size=50, generations=200, elite_frac=0.1, mutation_rate=0.2):
        self.base_doctors = base_doctors
        self.max_seconded = max_seconded
        self.pop_size = pop_size
        self.generations = generations
        self.elite_frac = elite_frac
        self.mutation_rate = mutation_rate

    def random_team(self, seconded:int) -> List[DoctorSchedule]:
        n = self.base_doctors + seconded
        return [random_doctor_schedule() for _ in range(n)]

    def mutate_doc(self, doc: DoctorSchedule) -> DoctorSchedule:
        c = doc.clone()
        op = random.random()
        if op < 0.25:
            # toggle a night
            d = random.randrange(DAYS)
            c.nights[d] = not c.nights[d]
        elif op < 0.65:
            # tweak a day shift
            d = random.randrange(DAYS)
            if c.day_shifts[d] and random.random() < 0.7:
                i = random.randrange(len(c.day_shifts[d]))
                s, e = c.day_shifts[d][i]
                s += random.choice([-1, 1])
                e += random.choice([-1, 1])
                # clamp
                s = max(7, min(23, s))
                e = max(s+3, min(24, e))
                c.day_shifts[d][i] = (s, e)
            else:
                # add or remove
                if random.random() < 0.5 and len(c.day_shifts[d]) < 2:
                    length = random.randint(3, 8)
                    s = random.randint(7, 24 - length)
                    c.day_shifts[d].append((s, s+length))
                elif c.day_shifts[d]:
                    c.day_shifts[d].pop(random.randrange(len(c.day_shifts[d])))
        else:
            # shift a rest day
            idx = random.randrange(DAYS)
            c.rest_day = [False]*DAYS
            c.rest_day[idx] = True
        enforce_constraints(c)
        return c

    def crossover(self, a: List[DoctorSchedule], b: List[DoctorSchedule]) -> List[DoctorSchedule]:
        # uniform crossover per doctor
        n = len(a)
        child = []
        for i in range(n):
            parent = a if random.random() < 0.5 else b
            child.append(parent[i].clone())
        return child

    def repair_team_min_staff(self, team: List[DoctorSchedule]):
        # Ensure actual coverage: if some hour has zero actual assigned, try to assign shifts to existing doctors.
        H = TOTAL_HOURS
        mat = [hours_matrix_of_doctor(doc) for doc in team]
        raw_staff = np.sum(mat, axis=0)
        zeros = np.where(raw_staff == 0)[0]
        if len(zeros) == 0:
            return
        # process each zero hour; attempt to assign a 3h white shift covering that hour
        for t in zeros:
            d, h = t_to_dh(t)
            assigned = False
            # prefer doctors with spare capacity on that day
            idxs = list(range(len(team)))
            random.shuffle(idxs)
            for i in idxs:
                doc = team[i].clone()
                # if hour is before 7, try to assign night on day d (i.e., doc.nights[d]=True) if allowed
                if h < 7:
                    # check if it's safe to set a night: no work on prev 16..23 and no day_shifts on d 7..23 and no conflict with other nights
                    prev = (d - 1) % DAYS
                    conflict_pre = any(hours_matrix_of_doctor(doc)[dh_to_t(prev, hh)] == 1 for hh in range(16, 24))
                    conflict_post = any(hours_matrix_of_doctor(doc)[dh_to_t(d, hh)] == 1 for hh in range(7, 24))
                    if not conflict_pre and not conflict_post:
                        doc.nights[d] = True
                        enforce_constraints(doc)
                        # check if now covers t
                        if hours_matrix_of_doctor(doc)[t] == 1:
                            team[i] = doc
                            assigned = True
                            break
                        else:
                            # revert
                            continue
                else:
                    # attempt to insert a 3h shift that covers h, respecting constraints
                    # choose start such that h in [start, end-1]
                    start_candidates = list(range(max(7, h - 2), min(h, 24 - 3) + 1))
                    random.shuffle(start_candidates)
                    for s in start_candidates:
                        e = s + 3
                        if can_add_white_shift(doc, d, s, e):
                            doc.day_shifts[d].append((s, e))
                            enforce_constraints(doc)
                            if hours_matrix_of_doctor(doc)[t] == 1:
                                team[i] = doc
                                assigned = True
                                break
                    if assigned:
                        break
            # if not assigned, try a more forceful repair: replace a randomly picked doctor by a fresh random doctor that covers t
            if not assigned:
                for _ in range(8):
                    i = random.randrange(len(team))
                    newdoc = random_doctor_schedule()
                    # if needed, force the newdoc to cover t
                    if h < 7:
                        newdoc.nights[d] = True
                        enforce_constraints(newdoc)
                    else:
                        s = max(7, min(h, 24 - 3))
                        newdoc.day_shifts[d].append((s, s+3))
                        enforce_constraints(newdoc)
                    if hours_matrix_of_doctor(newdoc)[t] == 1:
                        team[i] = newdoc
                        assigned = True
                        break
            # if still not assigned, leave it (evaluator will detect queue high); proceed to next zero
            # Note: in practice this aggressive replace should resolve almost all zeros
        return

    def initial_population(self) -> List[Tuple[List[DoctorSchedule], int]]:
        pop = []
        candidates_seconded = list(range(0, self.max_seconded+1))
        for _ in range(self.pop_size):
            k = random.choice(candidates_seconded)
            team = self.random_team(k)
            self.repair_team_min_staff(team)
            pop.append((team, k))
        return pop

    def evolve(self):
        population = self.initial_population()
        best = None
        for gen in range(self.generations):
            evaluated = []
            for team, k in population:
                res = evaluate_schedule(team, k)
                evaluated.append((res["cost"], team, k, res))
            evaluated.sort(key=lambda x: x[0])
            if best is None or evaluated[0][0] < best[0]:
                best = evaluated[0]
                print(f"Gen {gen}: best cost={best[0]:.2f}, wait={best[3]['total_wait']:.2f}, work={best[3]['total_work_hours']}, K={best[2]}")
            elite_n = max(1, int(self.elite_frac * self.pop_size))
            elites = evaluated[:elite_n]
            next_pop = []
            for i in range(elite_n):
                # deep clone the team
                team_clone = [doc.clone() for doc in elites[i][1]]
                next_pop.append((team_clone, elites[i][2]))
            while len(next_pop) < self.pop_size:
                # tournament selection
                def tournament():
                    k = 3
                    cand = random.sample(evaluated, k)
                    cand.sort(key=lambda x: x[0])
                    return cand[0]
                p1 = tournament()
                p2 = tournament()
                team1, k1 = p1[1], p1[2]
                team2, k2 = p2[1], p2[2]
                k_child = k1 if random.random() < 0.5 else k2
                n = self.base_doctors + k_child
                def resize(team, n):
                    if len(team) == n:
                        return [doc.clone() for doc in team]
                    elif len(team) > n:
                        return [team[i].clone() for i in random.sample(range(len(team)), n)]
                    else:
                        added = [random_doctor_schedule() for _ in range(n - len(team))]
                        return [doc.clone() for doc in team] + added
                A = resize(team1, n)
                B = resize(team2, n)
                child_team = self.crossover(A, B)
                # mutate
                for i in range(len(child_team)):
                    if random.random() < self.mutation_rate:
                        child_team[i] = self.mutate_doc(child_team[i])
                # repair coverage
                self.repair_team_min_staff(child_team)
                next_pop.append((child_team, k_child))
            population = next_pop
        return best

# -----------------------------
# Build readable outputs
# -----------------------------
def export_schedule(best_team: List[DoctorSchedule], k_seconded: int, eval_res: Dict):
    n = len(best_team)
    records = []
    for i, doc in enumerate(best_team):
        mat = hours_matrix_of_doctor(doc)
        for t in range(TOTAL_HOURS):
            d, h = t_to_dh(t)
            on = int(mat[t])
            typ = "off"
            if on == 1:
                # determine if night or day by schedule content
                if doc.nights[d] and h < 7:
                    typ = "night"
                elif h >= 7:
                    typ = "day"
                else:
                    typ = "on"
            records.append({
                "doctor_id": i+1,
                "day": d,
                "hour": h,
                "on_duty": on,
                "type": typ
            })
    df = pd.DataFrame.from_records(records)
    df.to_csv("schedule2.csv", index=False, encoding="utf-8-sig")

    # staffing summary
    staff = staffing_from_team(best_team)
    rows = []
    for t in range(TOTAL_HOURS):
        d, h = t_to_dh(t)
        rows.append({
            "day": d, "hour": h, "staff": int(staff[t]), "lambda": LAMBDA[t], "service_cap": mu * int(staff[t])
        })
    pd.DataFrame(rows).to_csv("staffing_summary2.csv", index=False, encoding="utf-8-sig")

    summary = {
        "total_cost": eval_res["cost"],
        "total_wait_hours": eval_res["total_wait"],
        "total_work_hours": int(eval_res["total_work_hours"]),
        "seconded_doctors": k_seconded,
        "ending_queue": eval_res["ending_queue"],
        "served_total": eval_res["served_total"],
        "avg_staff": float(np.mean(staff)),
        "max_staff": int(np.max(staff)),
        "min_staff": int(np.min(staff))
    }
    pd.DataFrame([summary]).to_csv("summary2.csv", index=False, encoding="utf-8-sig")
    print("Saved schedule.csv, staffing_summary.csv, summary.csv")

# -----------------------------
# Main
# -----------------------------
def main():
    ga = GAOptimizer(
        base_doctors=BASE_DOCTORS,
        max_seconded=7,
        pop_size=60,
        generations=120,
        elite_frac=0.15,
        mutation_rate=0.25
    )
    best_cost, best_team, best_k, best_res = ga.evolve()
    print("Best objective:", best_cost)
    print("Details:", best_res)
    export_schedule(best_team, best_k, best_res)

if __name__ == "__main__":
    main()
