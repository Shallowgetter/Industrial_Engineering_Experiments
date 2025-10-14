# deterministic_coverage_scheduler_full.py
# 说明: 将下列代码保存为 .py 文件并运行。需安装 pandas: pip install pandas

import math
import pandas as pd

# ========== 参数 ==========
mu = 6.0  # 医生服务速率（人/小时）
c_borrow = 20.0  # 借调医生每周成本
num_permanent = 11  # 急诊本部固定医生数量

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
arrival_rates = [x for day in lambda_data for x in day]  # flatten -> 168 hours
hours = len(arrival_rates)  # should be 168

# ========== 生成候选班次 ==========
def generate_candidate_shifts():
    shifts = []
    # night shifts: fixed 0..7 for each day (hours 0..6)
    for day in range(7):
        start = day * 24 + 0
        shifts.append({
            'type': 'night',
            'start': start,
            'end': start + 7,
            'length': 7,
            'hours': [h % hours for h in range(start, start + 7)],
            'label': f'Night_d{day}'
        })
    # white shifts: starts 7..23, length 3..8, must finish by midnight of that day
    for day in range(7):
        for start_hour in range(7, 24):
            for L in range(3, 9):  # 3..8
                start = day * 24 + start_hour
                end = start + L
                if end <= day * 24 + 24:
                    shifts.append({
                        'type': 'white',
                        'start': start,
                        'end': end,
                        'length': L,
                        'hours': [h % hours for h in range(start, end)],
                        'label': f'White_d{day}_s{start_hour}_L{L}'
                    })
    return shifts

CANDIDATE_SHIFTS = generate_candidate_shifts()

# ========== 医生类 ==========
class Doctor:
    def __init__(self, doc_id, borrowed=False):
        self.id = doc_id
        self.borrowed = borrowed
        self.shifts = []          # list of shift dicts
        self.hours = set()        # set of worked hour indices (0..167)
        self.night_count = 0      # number of night shifts assigned

    def _post_night_rest_intervals(self, include_simulated=None):
        """
        返回所有夜班后强制休息 24 小时的区间集合（每个为一组小时的 set，已 mod hours）。
        include_simulated: 一个 shift dict，如果提供则把它也当作已分配的来计算（用于 can_take 中模拟）。
        """
        intervals = []
        shifts = list(self.shifts)
        if include_simulated is not None:
            shifts = shifts + [include_simulated]
        for s in shifts:
            if s['type'] == 'night':
                rest_start = s['end']
                rest_hours = set((h % hours) for h in range(rest_start, rest_start + 24))
                intervals.append(rest_hours)
        return intervals

    def _has_full_24h_off_possible(self, include_simulated=None):
        """
        判断在考虑现有分配（可选地包含 include_simulated）后，
        是否仍存在至少一个合法的整天休息区间 [d*24+7, d*24+31)
        且该整天休息不是某个夜班后的强制24h休息（因为题目要求夜班后的那24h不能用于满足“每周至少休息1整天”）。
        采用保守策略：分配后必须立刻保证存在这样的整天休息。
        """
        # 计算将来将被占用的小时集合
        new_hours = set(self.hours)
        if include_simulated is not None:
            for h in include_simulated['hours']:
                new_hours.add(h % hours)

        # 计算夜班后强制休息的小时集合（每个夜班后的24h）
        post_night_intervals = self._post_night_rest_intervals(include_simulated=include_simulated)

        # 对于每个日 (0..6)，构建该日的 07:00..次日07:00 的小时集合，并检查：
        #  - 该集合与 new_hours 无交集（即整天都没工作）
        #  - 该集合 **不是** 等于任意一个 post-night-interval（如果相等，则该整天是夜班后的强制休息，不能算）
        for day in range(7):
            day_interval = set(((day * 24 + 7 + offset) % hours) for offset in range(24))  # 07:00..next07
            # must be fully free of worked hours
            if day_interval & new_hours:
                continue
            # must not be identical to any night-post-rest interval (we treat identical as disallowed)
            identical_to_post_night = False
            for pn in post_night_intervals:
                if pn == day_interval:
                    identical_to_post_night = True
                    break
            if identical_to_post_night:
                continue
            # found at least one valid full-day off
            return True
        return False

    def can_take(self, shift):
        # 1) no overlap with existing worked hours
        for h in shift['hours']:
            if (h % hours) in self.hours:
                return False

        # 2) night-specific constraints
        if shift['type'] == 'night':
            # a) previous 8 hours must not have work (考虑周环绕)
            prev_8 = [ (h % hours) for h in range(shift['start'] - 8, shift['start']) ]
            if any((ph in self.hours) for ph in prev_8):
                return False
            # b) can't have more than 2 nights
            if self.night_count >= 2:
                return False

        # 3) white shift daily constraints
        if shift['type'] == 'white':
            day = shift['start'] // 24
            # a) total white hours on that day <= 12
            hours_on_day = [h for h in self.hours if h // 24 == day]
            if len(hours_on_day) + shift['length'] > 12:
                return False
            # b) at most two white shifts per day
            shifts_on_day = [s for s in self.shifts if s['type'] == 'white' and (s['start'] // 24) == day]
            if len(shifts_on_day) >= 2:
                return False
            # c) if one existing white shift on that day, ensure at least 2h gap between them
            if len(shifts_on_day) == 1:
                other = shifts_on_day[0]
                if not (shift['end'] <= other['start'] - 2 or other['end'] <= shift['start'] - 2):
                    return False

        # 4) night-post-rest overlap: any existing night's post-night-rest 24h cannot overlap with new shift
        #    i.e., cannot assign a shift whose hours intersect with any post-night rest interval
        for s in self.shifts:
            if s['type'] == 'night':
                rest_start = s['end']
                rest_set = set((h % hours) for h in range(rest_start, rest_start + 24))
                if any(((h % hours) in rest_set) for h in shift['hours']):
                    return False
        # Also consider if simulated shift is a night: ensure its post-night-rest wouldn't be overlapped by existing hours
        if shift['type'] == 'night':
            rest_start = shift['end']
            rest_set = set((h % hours) for h in range(rest_start, rest_start + 24))
            if any((h in self.hours) for h in rest_set):
                # cannot assign a night if the mandatory 24h rest after it would overlap already scheduled work
                return False

        # 5) if assigning this shift would make it impossible to have at least one valid full 24h off (07..next07),
        #    then reject (conservative feasibility check to satisfy constraint 13).
        if not self._has_full_24h_off_possible(include_simulated=shift):
            return False

        return True

    def assign(self, shift):
        # perform assignment (caller should have ensured can_take True)
        self.shifts.append(shift)
        for h in shift['hours']:
            self.hours.add(h % hours)
        if shift['type'] == 'night':
            self.night_count += 1


# ========== 贪心覆盖逻辑 ==========
required = [max(1, math.ceil(arrival_rates[t] / mu)) for t in range(hours)]
perms = [Doctor(f'D{d+1}', borrowed=False) for d in range(num_permanent)]
borrows = []
active = [0] * hours

# Initialize active based on any pre-assigned shifts (none), so start zeros
# Greedy: for each hour t, if coverage < required, try to assign the longest candidate shift covering t
for t in range(hours):
    # loop until coverage meets required or no more assignments possible
    safety_counter = 0
    while active[t] < required[t]:
        safety_counter += 1
        if safety_counter > 3000:
            # prevent infinite loop in degenerate cases
            break

        assigned = False
        # candidate shifts that cover hour t
        candidates = [s for s in CANDIDATE_SHIFTS if (t % hours) in s['hours']]
        # sort by longer shifts first (cover more hours), tie-breaker: prefer white over night? keep length priority
        candidates = sorted(candidates, key=lambda s: -s['length'])

        # try existing permanent doctors first
        for doc in perms:
            for s in candidates:
                if doc.can_take(s):
                    doc.assign(s)
                    for h in s['hours']:
                        active[h % hours] += 1
                    assigned = True
                    break
            if assigned:
                break
        if assigned:
            continue

        # try to borrow a new doctor
        new_doc = Doctor(f'B{len(borrows) + 1}', borrowed=True)
        assigned2 = False
        for s in candidates:
            if new_doc.can_take(s):
                new_doc.assign(s)
                for h in s['hours']:
                    active[h % hours] += 1
                borrows.append(new_doc)
                perms.append(new_doc)  # add to the pool so future assignments may use them too
                assigned2 = True
                break
        if assigned2:
            continue

        # if neither permanent nor a new borrow could take any candidate covering t, break out (can't cover further)
        break

# ========== 保存 CSV ==========
schedule = {}
for d in perms:
    schedule[d.id] = {
        'borrowed': d.borrowed,
        'hours': [1 if h in d.hours else 0 for h in range(hours)],
        'total_hours': len(d.hours),
        'shifts': d.shifts
    }

def erlang_c_wait(lmbda, mu, s):
    # conservative Erlang-C approximate wait; if s<=0 or rho>=1 returns large wait
    if s <= 0:
        return 1e6
    rho = lmbda / (s * mu)
    if rho >= 1:
        return 1e6
    a = lmbda / mu
    sumt = sum([a ** k / math.factorial(k) for k in range(s)])
    last = a ** s / (math.factorial(s) * (1 - rho))
    P0 = 1.0 / (sumt + last)
    C = last * P0
    Wq = C / (s * mu - lmbda)
    return Wq

Wq = [erlang_c_wait(arrival_rates[t], mu, max(1, active[t])) for t in range(hours)]
total_wait = sum(arrival_rates[t] * Wq[t] for t in range(hours))
total_doc_hours = sum(info['total_hours'] for info in schedule.values())
num_borrowed = sum(1 for info in schedule.values() if info['borrowed'])
total_borrow_cost = num_borrowed * c_borrow
objective = total_wait + 1.3 * total_doc_hours + total_borrow_cost

print("=== 调度结果摘要 ===")
print("总医生人数（含借调）:", len(schedule))
print("借调医生数:", num_borrowed)
print("总医生工时（小时）:", total_doc_hours)
print("估计的一周总病人等待时间（小时）: {:.2f}".format(total_wait))
print("借调总成本:", total_borrow_cost, "目标值(合成):", objective)

# doctor_schedule.csv
sched_df = pd.DataFrame({did: info['hours'] for did, info in schedule.items()})
sched_df.index = [f'H{h}' for h in range(hours)]
sched_df.to_csv('doctor_schedule.csv')

# hourly_coverage.csv
hourly = pd.DataFrame({
    'lambda': arrival_rates,
    'required': required,
    'active': active,
    'Wq': Wq,
    'day': [h // 24 for h in range(hours)],
    'hour_of_day': [h % 24 for h in range(hours)]
}, index=[f'H{h}' for h in range(hours)])
hourly.to_csv('hourly_coverage.csv')

# ========== 新增：医生排班时间汇总 human-readable ==========
summary_rows = []
for did, info in schedule.items():
    for shift in info['shifts']:
        summary_rows.append({
            'doctor': did,
            'borrowed': info['borrowed'],
            'type': shift['type'],
            'day': (shift['start'] // 24) % 7,
            'start_hour': shift['start'] % 24,
            'end_hour': shift['end'] % 24,
            'length': shift['length'],
            'label': shift['label']
        })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('doctor_schedule_summary.csv', index=False)

print("已生成完整排班汇总: doctor_schedule_summary.csv")
print("原 doctor_schedule.csv 和 hourly_coverage.csv 也已更新完成。")
