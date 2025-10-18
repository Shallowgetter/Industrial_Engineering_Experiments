"""
独立仿真器模块 - 用于模拟急诊室运作并计算等待时间
"""

import random
import math
import heapq
from typing import List, Dict, Tuple, Optional


class Doctor:
    def __init__(self, doc_id: str, shifts: List[Tuple[int, int]]):
        self.id = doc_id
        self.shifts = shifts
        self.busy_until: Optional[float] = None

    def on_duty(self, t: float) -> bool:
        """判断在绝对时间 t（小时）时医生是否值班"""
        hour = int(math.floor(t)) % 168
        for s, e in self.shifts:
            if s <= hour < e:
                return True
        return False

    def is_idle_at(self, t: float) -> bool:
        """医生在 time t 是否空闲（且在班）"""
        return (self.busy_until is None or self.busy_until <= t) and self.on_duty(t)


class Patient:
    def __init__(self, pid: int, arrival_time: float):
        self.id = pid
        self.arrival_time = arrival_time
        self.start_service_time: Optional[float] = None


def poisson_knuth(lmbda: float, rng: random.Random) -> int:
    """Knuth 算法生成 Poisson(lmbda) 的样本"""
    if lmbda <= 0:
        return 0
    L = math.exp(-lmbda)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def exp_sample(rate: float, rng: random.Random) -> float:
    """指数分布采样，返回单位：小时"""
    if rate <= 0:
        return float('inf')
    return rng.expovariate(rate)


def generate_week_arrivals(arrival_rates: List[float], rng: random.Random) -> List[float]:
    """生成一周到达事件"""
    arrivals = []
    if len(arrival_rates) != 168:
        raise ValueError("arrival_rates 长度应为 168（每周小时数）")
    for i, lam in enumerate(arrival_rates):
        n = poisson_knuth(lam, rng)
        for _ in range(n):
            t = i + rng.random()
            arrivals.append(t)
    arrivals.sort()
    return arrivals


_event_seq = 0


def push_event(pq, time, ev_type, payload):
    global _event_seq
    heapq.heappush(pq, (time, _event_seq, (ev_type, payload)))
    _event_seq += 1


def pop_event(pq):
    return heapq.heappop(pq)


def simulate_one_week(arrival_rates: List[float],
                      doctors: List[Doctor],
                      mu: float,
                      rng: random.Random) -> float:
    """仿真一周，返回病人总等待时间（小时）"""
    WEEK_HOURS = 168.0
    arrivals = generate_week_arrivals(arrival_rates, rng)

    pq = []
    queue: List[int] = []
    patients: Dict[int, Patient] = {}
    total_wait = 0.0
    pid_counter = 0

    for t in arrivals:
        pid = pid_counter
        pid_counter += 1
        patients[pid] = Patient(pid, t)
        push_event(pq, t, 'arrival', pid)

    while pq:
        time, _, (ev_type, payload) = pop_event(pq)
        if time >= WEEK_HOURS:
            break

        if ev_type == 'arrival':
            pid = payload
            idle_idx = None
            for i, doc in enumerate(doctors):
                if doc.on_duty(time) and (doc.busy_until is None or doc.busy_until <= time):
                    idle_idx = i
                    break
            if idle_idx is not None:
                patient = patients[pid]
                patient.start_service_time = time
                wait = patient.start_service_time - patient.arrival_time
                total_wait += wait
                st = exp_sample(mu, rng)
                finish_t = time + st
                doctors[idle_idx].busy_until = finish_t
                push_event(pq, finish_t, 'departure', idle_idx)
            else:
                queue.append(pid)

        elif ev_type == 'departure':
            doc_idx = payload
            doc = doctors[doc_idx]
            if queue:
                next_pid = queue.pop(0)
                patient = patients[next_pid]
                current_time = time
                chosen_idx = None
                if doc.on_duty(current_time):
                    chosen_idx = doc_idx
                else:
                    for i, d in enumerate(doctors):
                        if d.on_duty(current_time) and (d.busy_until is None or d.busy_until <= current_time):
                            chosen_idx = i
                            break
                if chosen_idx is None:
                    queue.insert(0, next_pid)
                    doc.busy_until = None
                    continue
                patient.start_service_time = current_time
                total_wait += (patient.start_service_time - patient.arrival_time)
                st = exp_sample(mu, rng)
                finish_t = current_time + st
                doctors[chosen_idx].busy_until = finish_t
                push_event(pq, finish_t, 'departure', chosen_idx)
            else:
                doc.busy_until = None
        else:
            raise RuntimeError("未知事件类型")

    return total_wait


def run_simulation_multiple(arrival_rates: List[float], doctors: List[Doctor],
                            mu: float, n_runs: int = 10, seed: int = 42) -> float:
    """运行多次仿真取平均值"""
    total_waits = []
    for run in range(n_runs):
        # 每次仿真创建新的医生实例
        doctors_copy = [Doctor(d.id, d.shifts) for d in doctors]
        seed_run = seed + run
        rng = random.Random(seed_run)
        week_wait = simulate_one_week(arrival_rates, doctors_copy, mu, rng)
        total_waits.append(week_wait)

    return sum(total_waits) / len(total_waits)