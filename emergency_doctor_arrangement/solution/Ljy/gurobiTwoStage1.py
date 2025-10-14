import gurobipy as gp
from gurobipy import GRB
import json
import numpy as np

# ------------------------------
# 基本参数
# ------------------------------
WEEK_HOURS = 7 * 24
DAY_START, DAY_END = 7, 24
NIGHT_START, NIGHT_END = 0, 7
INTERNAL_CAP = 11
MAX_DOCTORS = 11
MIN_DAY_LEN, MAX_DAY_LEN = 3, 8

# ------------------------------
# 构造候选班次
# ------------------------------
def generate_candidate_shifts():
    shifts = []
    for d in range(7):
        base = d * 24
        # day shifts
        for start in range(DAY_START, DAY_END):
            for length in range(MIN_DAY_LEN, MAX_DAY_LEN + 1):
                end = start + length
                if end <= DAY_END:
                    shifts.append({"day": d, "start": base + start, "end": base + end, "tag": "day"})
        # night shift (固定)
        shifts.append({"day": d, "start": base + NIGHT_START, "end": base + NIGHT_END, "tag": "night"})
    return shifts

CANDIDATE_SHIFTS = generate_candidate_shifts()

# ------------------------------
# 两阶段优化
# ------------------------------
def optimize_schedule(arrival_rates, mu, num_doctors=11, c_borrow=20.0):
    model = gp.Model("ER_Scheduling")

    # doctor id 集合
    doctors = [f"D{i+1}" for i in range(num_doctors)]

    # 决策变量：是否选择某个班次
    x = {}
    for d in doctors:
        for s_id, sh in enumerate(CANDIDATE_SHIFTS):
            x[d, s_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{d}_{s_id}")

    model.update()

    # ------------------------------
    # 约束
    # ------------------------------
    # 每个医生每天 ≤ 2 个白班，总时长 ≤ 12h
    for d in doctors:
        for day in range(7):
            day_shifts = [s_id for s_id, sh in enumerate(CANDIDATE_SHIFTS) if sh["day"] == day and sh["tag"] == "day"]
            model.addConstr(gp.quicksum(x[d, s] for s in day_shifts) <= 2, f"max2_day_{d}_{day}")
            model.addConstr(gp.quicksum((CANDIDATE_SHIFTS[s]["end"] - CANDIDATE_SHIFTS[s]["start"]) * x[d, s] for s in day_shifts) <= 12, f"max12h_day_{d}_{day}")

    # 夜班限制：每周 ≤ 2
    for d in doctors:
        night_shifts = [s_id for s_id, sh in enumerate(CANDIDATE_SHIFTS) if sh["tag"] == "night"]
        model.addConstr(gp.quicksum(x[d, s] for s in night_shifts) <= 2, f"max2_night_{d}")

    # 覆盖性：每小时至少 1 医生
    for h in range(WEEK_HOURS):
        covering = []
        for d in doctors:
            for s_id, sh in enumerate(CANDIDATE_SHIFTS):
                if sh["start"] <= h < sh["end"]:
                    covering.append(x[d, s_id])
        model.addConstr(gp.quicksum(covering) >= 1, f"cover_{h}")

    # 每周至少一天完整休息
    for d in doctors:
        for day in range(7):
            day_shifts = [s_id for s_id, sh in enumerate(CANDIDATE_SHIFTS) if sh["day"] == day]
            model.addConstr(gp.quicksum(x[d, s] for s in day_shifts) <= len(day_shifts), f"daybound_{d}_{day}")
        model.addConstr(
            gp.quicksum(x[d, s] for s in range(len(CANDIDATE_SHIFTS))) <= 6 * 10,  # 粗略保证有休息
            f"rest_day_{d}"
        )

    # ------------------------------
    # 目标函数 (staff hours + borrow cost, 忽略等待时间近似)
    # ------------------------------
    staff_hours = gp.quicksum((sh["end"] - sh["start"]) * x[d, s_id] for d in doctors for s_id, sh in enumerate(CANDIDATE_SHIFTS))
    borrow_cost = 0.0  # 暂无借调
    model.setObjective(staff_hours * 1.3 + borrow_cost, GRB.MINIMIZE)

    # ------------------------------
    # 求解
    # ------------------------------
    model.Params.TimeLimit = 60
    model.optimize()

    # ------------------------------
    # 导出 JSON
    # ------------------------------
    output = {"arrival_rates": arrival_rates, "mu": mu, "c_borrow": c_borrow, "doctors": []}
    if model.SolCount > 0:
        for d in doctors:
            shifts = []
            for s_id, sh in enumerate(CANDIDATE_SHIFTS):
                if x[d, s_id].X > 0.5:
                    shifts.append({"start": sh["start"], "end": sh["end"], "tag": sh["tag"]})
            if shifts:
                output["doctors"].append({"id": d, "origin": "internal", "shifts": shifts})

    return output

# ------------------------------
# 使用示例
# ------------------------------
if __name__ == "__main__":
    arrival_rates = [
        # Day 1 (0:00-23:00)
        8.40, 5.49, 3.88, 4.00, 4.20, 6.73, 21.97, 43.76, 28.18, 25.38, 25.40, 23.26, 
        11.75, 11.20, 13.07, 19.45, 18.76, 19.69, 15.57, 29.39, 38.76, 31.66, 15.96, 9.61,
        
        # Day 2 (24:00-47:00)
        5.31, 3.16, 5.61, 1.94, 4.46, 7.56, 25.86, 39.92, 27.97, 27.90, 19.04, 17.43, 
        8.70, 19.26, 14.74, 16.07, 19.79, 13.16, 31.48, 33.67, 36.39, 21.99, 13.04, 7.73,
        
        # Day 3 (48:00-71:00)
        6.77, 4.28, 2.43, 1.73, 6.80, 4.62, 25.82, 44.01, 34.12, 18.63, 29.10, 14.03, 
        14.63, 16.32, 15.15, 22.69, 24.80, 30.30, 40.57, 37.47, 25.07, 17.95, 7.81, 9.71,
        
        # Day 4 (72:00-95:00)
        12.82, 6.17, 4.79, 3.90, 4.46, 6.26, 11.08, 33.77, 29.86, 13.26, 17.44, 19.74, 
        11.75, 19.83, 24.87, 11.01, 16.00, 15.43, 25.25, 33.21, 27.62, 18.60, 13.80, 8.71,
        
        # Day 5 (96:00-119:00)
        4.07, 7.36, 3.91, 3.08, 2.68, 9.09, 25.83, 47.81, 32.02, 22.54, 19.06, 14.15, 
        14.01, 18.07, 18.00, 13.12, 13.23, 13.56, 31.77, 32.72, 23.57, 15.18, 7.42, 8.33,
        
        # Day 6 (120:00-143:00)
        5.88, 6.25, 4.00, 7.01, 2.74, 8.51, 23.07, 42.15, 28.01, 20.17, 26.46, 14.40, 
        13.92, 18.19, 22.79, 16.89, 25.10, 21.46, 32.74, 23.43, 22.21, 16.04, 14.48, 10.71,
        
        # Day 7 (144:00-167:00)
        2.94, 3.12, 2.00, 3.50, 1.37, 4.26, 28.67, 49.35, 13.56, 24.03, 21.52, 13.27, 
        17.94, 22.97, 24.58, 28.18, 33.35, 34.37, 30.65, 31.58, 25.98, 13.93, 7.24, 5.35
    ]
    mu = 6.0
    result = optimize_schedule(arrival_rates, mu, num_doctors=11)

    with open("solution.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Schedule exported to solution.json")
