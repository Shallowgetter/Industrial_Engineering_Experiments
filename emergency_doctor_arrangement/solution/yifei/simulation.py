"""
离散事件仿真（DES）——单周服务系统随机仿真
- 分段常数泊松到达（按小时）
- 指数服务时间，单队列 FCFS，多服务器（医生）
- 医生按班次上/下线；下线时不截断在服服务，但完成后不再接新病人
- 若某小时无人上班，强制注入“保底医生”（internal-guard），以满足“至少一人值守”
- 精确跟踪到“医生个体”以统计跨班加班（可选）
- 成本口径：等待总时间、医生总上班时间（可选含加班）、借调成本、目标值

使用：
    python simulate_week.py --input /path/to/sample.json --runs 200 --seed 123

也可作为模块导入，调用 simulate_week(input_json_path, runs, seed)
"""

import json
import math
import heapq
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from collections import deque, defaultdict
import random


HOURS_PER_WEEK = 7 * 24  # 168
EPS = 1e-9


@dataclass
class Shift:
    start: float  # [start, end), 单位：小时（从周一0点起算）
    end: float


@dataclass
class Doctor:
    id: str
    origin: str  # "internal" | "borrowed" | "internal-guard"
    shifts: List[Shift]  # 不重叠、按时间升序
    busy: bool = False
    idle: bool = False  # 仅当 on-duty 且不忙时为 True，可接新单
    current_service_end: Optional[float] = None
    current_shift_end: Optional[float] = None
    total_overtime: float = 0.0

    def on_duty(self, t: float) -> bool:
        """t 是否落在任一班次内（[start, end)）"""
        # 简单线性扫描；班次数通常不多，足够快
        for s in self.shifts:
            if s.start - EPS <= t < s.end - EPS:
                return True
        return False

    def shift_end_containing(self, t: float) -> Optional[float]:
        """返回包含 t 的班次结束时刻（若 t 不在班内返回 None）"""
        for s in self.shifts:
            if s.start - EPS <= t < s.end - EPS:
                return s.end
        return None

    @property
    def planned_hours(self) -> float:
        return sum(max(0.0, s.end - s.start) for s in self.shifts)


@dataclass(order=True)
class Event:
    time: float
    priority: int   # 0: Departure, 1: CapacityChange, 2: Arrival
    seq: int        # 稳定排序用
    kind: str = field(compare=False)  # "depart"|"cap"|"arrive"
    payload: Any = field(compare=False, default=None)


def _merge_contiguous_hours_to_shifts(hours: List[int]) -> List[Shift]:
    """将一组离散整点小时（例如 [5,6,7,10,11]）合并为若干连续班段"""
    if not hours:
        return []
    hours = sorted(hours)
    merged = []
    start = hours[0]
    prev = hours[0]
    for h in hours[1:]:
        if h == prev + 1:
            prev = h
            continue
        # 断开
        merged.append(Shift(start=float(start), end=float(prev + 1)))
        start = h
        prev = h
    merged.append(Shift(start=float(start), end=float(prev + 1)))
    return merged


def _build_doctors(raw_doctors: List[Dict[str, Any]], enforce_min_one_per_hour=True) -> Tuple[List[Doctor], List[int]]:
    """
    从 json doctors 构建 Doctor 列表，同时检查每小时至少 1 人在岗。
    若发现某小时无人值守，注入 internal-guard 医生覆盖该小时（成本计入 planned_hours）。
    返回：(doctors, coverage_counts[168])
    """
    # 初始医生
    doctors: List[Doctor] = []
    coverage = [0] * HOURS_PER_WEEK  # 每小时在岗医生数（不含在服的加班，仅按班表）

    for d in raw_doctors:
        shifts = [Shift(float(s["start"]), float(s["end"])) for s in d.get("shifts", [])]
        # 规范化：限制在 [0, 168]，并去除无效段
        norm = []
        for s in shifts:
            a = max(0.0, min(float(HOURS_PER_WEEK), s.start))
            b = max(0.0, min(float(HOURS_PER_WEEK), s.end))
            if b - a > EPS:
                norm.append(Shift(a, b))
        norm.sort(key=lambda x: x.start)
        doc = Doctor(id=str(d["id"]), origin=str(d.get("origin", "internal")), shifts=norm)
        doctors.append(doc)
        # 加 coverage
        for s in norm:
            h0 = int(math.floor(s.start + EPS))
            h1 = int(math.floor(s.end - EPS))  # s.end 属于下个小时的起点，不覆盖
            # 例如 [3.0,7.0) 覆盖 3,4,5,6 四个小时
            for h in range(h0, min(h1 + 1, HOURS_PER_WEEK)):
                if s.start - EPS <= h < s.end - EPS:
                    coverage[h] += 1

    if enforce_min_one_per_hour:
        # 找出空缺小时，注入“保底医生”
        empty_hours = [h for h, c in enumerate(coverage) if c <= 0]
        if empty_hours:
            guard_shifts = _merge_contiguous_hours_to_shifts(empty_hours)
            guard = Doctor(
                id="internal-guard-0",
                origin="internal-guard",
                shifts=guard_shifts,
            )
            doctors.append(guard)
            # 更新覆盖
            for s in guard_shifts:
                h0 = int(math.floor(s.start + EPS))
                h1 = int(math.floor(s.end - EPS))
                for h in range(h0, min(h1 + 1, HOURS_PER_WEEK)):
                    if s.start - EPS <= h < s.end - EPS:
                        coverage[h] += 1

    return doctors, coverage


def _generate_arrivals_by_hour(rng: random.Random, arrival_rates: List[float]) -> List[float]:
    """按小时泊松计数，再均匀撒入该小时，返回一周全部到达时刻（升序）"""
    arrivals = []
    for h, lam in enumerate(arrival_rates):
        lam = max(0.0, float(lam))
        # 采样 Poisson(lam)
        # Python stdlib 没有直接泊松采样，这里用 Knuth 法，lam 常见数量级可接受；也可换成 numpy/random 若允许
        # 为避免极大 lam 的性能问题，可在 lam>40 时用正态近似/伽马分解；此处先用简单实现
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= rng.random()
            if p <= L:
                break
        N = k - 1

        # 均匀撒入该小时
        for _ in range(N):
            u = rng.random()
            arrivals.append(h + u)
    arrivals.sort()
    return arrivals


def _expovariate(rng: random.Random, mu: float) -> float:
    """服务时间 ~ Exp(mu)；mu 为速率(人/小时)"""
    if mu <= 0:
        raise ValueError("mu must be > 0")
    # random.expovariate 以 λ 为参数，表示均值 1/λ
    return rng.expovariate(mu)


def simulate_week_once(cfg: Dict[str, Any], base_seed: int, run_index: int) -> Dict[str, Any]:
    """
    单次仿真，返回一个 dict：
    {
      "total_wait": float,
      "avg_wait": float,
      "n_arrivals": int,
      "doctor_work_hours": float,
      "doctor_work_hours_incl_ot": float,
      "borrow_cost": float,
      "objective": float,
      "objective_incl_ot": float
    }
    """
    rng = random.Random(base_seed + run_index)

    # 1) 读取配置
    arrival_rates = cfg["arrival_rates"]  # 长度 168
    mu = float(cfg["mu"])
    c_borrow = float(cfg.get("c_borrow", 20.0))
    include_overtime_in_cost_flag = bool(cfg.get("include_overtime_in_cost", False))

    raw_doctors = cfg.get("doctors", [])
    doctors, coverage = _build_doctors(raw_doctors, enforce_min_one_per_hour=True)

    # 2) 预生成到达
    arrival_times = _generate_arrivals_by_hour(rng, arrival_rates)

    # 3) 初始化事件堆
    # 优先级：Departure=0, CapacityChange=1, Arrival=2（确保同刻先出站再扩容再到达）
    ev_heap: List[Event] = []
    seq = 0

    # 到达事件
    for t in arrival_times:
        heapq.heappush(ev_heap, Event(time=t, priority=2, seq=seq, kind="arrive", payload=None))
        seq += 1

    # 整点切换事件（含 t=0 与 t=168）
    for h in range(0, HOURS_PER_WEEK + 1):
        heapq.heappush(ev_heap, Event(time=float(h), priority=1, seq=seq, kind="cap", payload=None))
        seq += 1

    # 4) 状态
    queue = deque()  # 存 (arr_time)
    total_wait = 0.0
    n_arrivals = len(arrival_times)

    # 维护“可接新单”的医生集合（id -> Doctor）
    id2doc = {d.id: d for d in doctors}
    idle_on_duty = set()

    def refresh_idle_set_at(t: float):
        """在‘整点切换’或医生状态变化后刷新可用集合（非忙、且在岗）"""
        idle_on_duty.clear()
        for d in doctors:
            if (not d.busy) and d.on_duty(t):
                d.idle = True
                idle_on_duty.add(d.id)
            else:
                d.idle = False

    def assign_as_possible(t: float):
        """在时刻 t，尽可能把排队病人派给空闲且当班的医生，FCFS"""
        nonlocal total_wait, seq
        while queue and idle_on_duty:
            # 取最早到达的病人
            arr_t = queue[0]
            # 选一个可用医生（集合中弹出即可；医生间公平性不影响 FCFS）
            doc_id = idle_on_duty.pop()
            d = id2doc[doc_id]

            # 该医生必须 on-duty 且不忙
            if d.busy or (not d.on_duty(t)):
                # 不符合则跳过（理论上不会发生，但稳妥处理）
                continue

            # 派单
            queue.popleft()
            wait = max(0.0, t - arr_t)
            total_wait += wait

            st_end = d.shift_end_containing(t)
            if st_end is None:
                # 罕见：如果某医生不在岗却进来，做容错：不统计加班（相当于 st_end=t）
                st_end = t

            service = _expovariate(rng, mu)
            end_t = t + service

            # 更新医生状态
            d.busy = True
            d.idle = False
            d.current_service_end = end_t
            d.current_shift_end = st_end

            # 推出 Departure 事件
            heapq.heappush(ev_heap, Event(time=end_t, priority=0, seq=seq, kind="depart", payload={"doc_id": d.id}))
            seq += 1

        # 循环终止：或队空，或无人可接

    # 初始时刻 t=0：刷新 idle 集合并尝试派单（队列为空，不会派上）
    refresh_idle_set_at(0.0)

    # 5) 事件主循环
    current_time = 0.0
    while ev_heap:
        ev = heapq.heappop(ev_heap)
        t = ev.time
        current_time = t

        if ev.kind == "depart":
            doc_id = ev.payload["doc_id"]
            d = id2doc[doc_id]
            # 加班统计
            if d.current_service_end is not None and d.current_shift_end is not None:
                ot = max(0.0, d.current_service_end - d.current_shift_end)
                if ot > 0:
                    d.total_overtime += ot
            # 释放医生
            d.busy = False
            d.current_service_end = None
            d.current_shift_end = None
            # 若此刻在岗，加入可接集合；否则保持不可接
            if d.on_duty(t):
                d.idle = True
                idle_on_duty.add(d.id)
            else:
                d.idle = False

            # 完成后尝试派单
            assign_as_possible(t)

        elif ev.kind == "cap":
            # 整点切换：更新哪些医生在岗（空闲者可接，新上岗立即可接；下岗者若空闲则不可再接）
            refresh_idle_set_at(t)
            # 扩容后可能能接更多
            assign_as_possible(t)

        elif ev.kind == "arrive":
            # 新病人进入系统
            queue.append(t)
            # 尝试派单
            assign_as_possible(t)

        else:
            raise RuntimeError("Unknown event kind")

        # 结束条件：到达事件仅到 168h；但服务可拖尾。我们一直处理到事件堆空即可。

    # 6) 统计与成本
    # 医生总上班时间（计划）：各自班次长度求和
    planned_hours = sum(d.planned_hours for d in doctors)
    # 加班（跨班尾声）：已在医生个体层面累计
    sum_overtime = sum(d.total_overtime for d in doctors)

    # 借调成本：去重借调医生数 × c_borrow
    borrowed_ids = {d.id for d in doctors if d.origin == "borrowed"}
    borrow_cost = c_borrow * len(borrowed_ids)

    objective = total_wait + 1.3 * planned_hours + borrow_cost
    objective_incl_ot = total_wait + 1.3 * (planned_hours + sum_overtime) + borrow_cost

    avg_wait = (total_wait / n_arrivals) if n_arrivals > 0 else 0.0

    return {
        "total_wait": total_wait,
        "avg_wait": avg_wait,
        "n_arrivals": n_arrivals,
        "doctor_work_hours": planned_hours,
        "doctor_overtime_hours": sum_overtime,
        "doctor_work_hours_incl_ot": planned_hours + sum_overtime,
        "borrow_cost": borrow_cost,
        "objective": objective,
        "objective_incl_ot": objective_incl_ot,
        "borrowed_doctors": len(borrowed_ids),
    }


def simulate_week(input_json_path: str, runs: int = 100, seed: int = 42, save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    读取 json，执行多次仿真，返回总体结果。
    若 save_path 不为空，会把 “每次结果 + 汇总统计” 保存为 JSON。
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 基本检查
    arrival_rates = cfg.get("arrival_rates", [])
    if len(arrival_rates) != HOURS_PER_WEEK:
        raise ValueError(f"'arrival_rates' 必须为长度 {HOURS_PER_WEEK} 的数组")

    seed = int(cfg.get("seed", seed))

    per_run = []
    for r in range(runs):
        res = simulate_week_once(cfg, base_seed=seed, run_index=r)
        per_run.append(res)

    def _mean(key: str) -> float:
        vals = [x[key] for x in per_run]
        return sum(vals) / len(vals) if vals else 0.0

    def _std(key: str) -> float:
        vals = [x[key] for x in per_run]
        if len(vals) <= 1:
            return 0.0
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        return math.sqrt(var)

    summary = {
        "runs": runs,
        "seed": seed,
        "mean_total_wait": _mean("total_wait"),
        "std_total_wait": _std("total_wait"),
        "mean_avg_wait": _mean("avg_wait"),
        "std_avg_wait": _std("avg_wait"),
        "mean_doctor_work_hours": _mean("doctor_work_hours"),
        "std_doctor_work_hours": _std("doctor_work_hours"),
        "mean_doctor_overtime_hours": _mean("doctor_overtime_hours"),
        "std_doctor_overtime_hours": _std("doctor_overtime_hours"),
        "mean_doctor_work_hours_incl_ot": _mean("doctor_work_hours_incl_ot"),
        "std_doctor_work_hours_incl_ot": _std("doctor_work_hours_incl_ot"),
        "mean_borrow_cost": _mean("borrow_cost"),
        "std_borrow_cost": _std("borrow_cost"),
        "mean_objective": _mean("objective"),
        "std_objective": _std("objective"),
        "mean_objective_incl_ot": _mean("objective_incl_ot"),
        "std_objective_incl_ot": _std("objective_incl_ot"),
        "mean_borrowed_doctors": _mean("borrowed_doctors"),
    }

    out = {"summary": summary, "runs": per_run}

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="输入 JSON（sample.json 同结构）")
    parser.add_argument("--runs", type=int, default=100, help="仿真次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（若 JSON 里有 seed，会覆盖此处）")
    parser.add_argument("--save", type=str, default="", help="结果保存到 JSON 路径（可选）")
    args = parser.parse_args()

    res = simulate_week(args.input, runs=args.runs, seed=args.seed, save_path=(args.save or None))
    # 终端简要打印
    print(json.dumps(res["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
