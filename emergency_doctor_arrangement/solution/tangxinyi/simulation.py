#!/usr/bin/env python3
"""
简化事件驱动急诊（ED）仿真器（Python）

- 读取医生排班 JSON（与用户示例格式兼容）
- 每小时到达率按 arrival_rates（168 个小时）
- 服务时间为指数分布（速率 mu，单位：人/小时）
- 全局 FIFO 排队，先到先服务；无离开（no reneging）
- 事件驱动（arrival / departure），每次仿真覆盖一周 [0,168)
- 重复 n_runs 次（默认 1000），输出每周总等待时间的平均值

参考：用户提供的 C++ 仿真代码的事件驱动结构与到达生成逻辑（已简化）。:contentReference[oaicite:1]{index=1}
"""
import json
import heapq
import argparse
import random
import math
from typing import List, Dict, Tuple, Optional

# ---------- 辅助随机采样（不依赖 numpy） ----------
def poisson_knuth(lmbda: float, rng: random.Random) -> int:
    """Knuth 算法生成 Poisson(lmbda) 的样本（当 lmbda 不特别大时表现良好）。"""
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
    """指数分布采样，返回单位：小时（rate 为 mu，人/小时）"""
    if rate <= 0:
        return float('inf')
    return rng.expovariate(rate)

# ---------- 医生 / 病人 数据结构 ----------
class Doctor:
    def __init__(self, doc_id: str, shifts: List[Tuple[int,int]]):
        self.id = doc_id
        # shifts: list of (start_hour_in_week, end_hour_in_week), integers 0..167
        self.shifts = shifts
        self.busy_until: Optional[float] = None  # None 表示空闲；否则是服务完成的绝对时间

    def on_duty(self, t: float) -> bool:
        """判断在绝对时间 t（小时）时医生是否值班（按一周 168 小时循环）"""
        hour = int(math.floor(t)) % 168
        for s,e in self.shifts:
            # interpret shifts as [start, end) in hour indices
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

# ---------- 事件队列 ----------
# 事件是三元组 (time, seq, (type, payload))
# 使用 seq 防止相同时间的事件比较失败
# type: 'arrival'  'departure'
# payload: 对于 arrival -> patient_id, 对于 departure -> doctor_index
_event_seq = 0
def push_event(pq, time, ev_type, payload):
    global _event_seq
    heapq.heappush(pq, (time, _event_seq, (ev_type, payload)))
    _event_seq += 1

def pop_event(pq):
    return heapq.heappop(pq)

# ---------- 生成一周到达事件 ----------
def generate_week_arrivals(arrival_rates: List[float], rng: random.Random) -> List[float]:
    """
    对 168 个小时中的每一小时 i：
      - 使用 Poisson(lambda=arrival_rates[i]) 得到本小时到达人数 n
      - 将这 n 个到达时刻在 [i, i+1) 内均匀分布采样
    返回一周内所有到达时间（浮点小时，相对于周开始 0）
    """
    arrivals = []
    if len(arrival_rates) != 168:
        raise ValueError("arrival_rates 长度应为 168（每周小时数）")
    for i, lam in enumerate(arrival_rates):
        n = poisson_knuth(lam, rng)
        for _ in range(n):
            # 均匀分布在 [i, i+1)
            t = i + rng.random()
            arrivals.append(t)
    arrivals.sort()
    return arrivals

# ---------- 单次一周仿真 ----------
def simulate_one_week(arrival_rates: List[float],
                      doctors: List[Doctor],
                      mu: float,
                      rng: random.Random) -> float:
    """
    仿真一周（0..168小时，总时长 = 168），返回这一周的**病人总等待时间**（小时）
    """
    WEEK_HOURS = 168.0
    # 生成所有到达事件（时间在 [0,168)）
    arrivals = generate_week_arrivals(arrival_rates, rng)

    # 初始化
    pq = []  # 事件队列
    queue: List[int] = []  # 保存等待患者的 id （FIFO, 存患者 id）
    patients: Dict[int, Patient] = {}
    total_wait = 0.0
    pid_counter = 0

    # push arrival events
    for t in arrivals:
        pid = pid_counter
        pid_counter += 1
        patients[pid] = Patient(pid, t)
        push_event(pq, t, 'arrival', pid)

    # 处理事件
    while pq:
        time, _, (ev_type, payload) = pop_event(pq)
        # 我们只处理在本周内的事件；任何超出 WEEK_HOURS 的事件会被忽略（arrival 本身均 < 168）
        if time >= WEEK_HOURS:
            break

        if ev_type == 'arrival':
            pid = payload
            # 找一位在班且空闲的医生（优先选择 busy_until is None 或 <= time）
            idle_idx = None
            for i, doc in enumerate(doctors):
                if doc.on_duty(time) and (doc.busy_until is None or doc.busy_until <= time):
                    idle_idx = i
                    break
            if idle_idx is not None:
                # 立即开始服务
                patient = patients[pid]
                patient.start_service_time = time
                wait = patient.start_service_time - patient.arrival_time
                total_wait += wait
                # 生成服务完成事件
                st = exp_sample(mu, rng)  # 单位：小时
                finish_t = time + st
                doctors[idle_idx].busy_until = finish_t
                push_event(pq, finish_t, 'departure', idle_idx)
            else:
                # 全忙，加入队尾
                queue.append(pid)

        elif ev_type == 'departure':
            doc_idx = payload
            doc = doctors[doc_idx]
            # 该医生完成当前患者服务
            # 检查队列里是否还有病人且医生仍在班（允许下班后把当前病人看完，但如果下一个病人需要由值班医生接手则看下是否值班）
            if queue:
                # 分配下一个病人（FIFO）
                next_pid = queue.pop(0)
                patient = patients[next_pid]
                # 如果医生在 time 时已经下班（按 doc.on_duty），我们仍任其完成当前 patient，但如果 doctor 不在班且不能继续接收下一位，
                # 为简化：只要 doctor.on_duty at current time，便直接由该 doctor 接手；否则在队列前寻找其他空闲在班医生；
                current_time = time
                if doc.on_duty(current_time) and (doc.busy_until is None or doc.busy_until <= current_time):
                    # this branch rarely true because we are processing departure where doc.busy_until == time
                    pass

                # find a doctor to serve next patient: prefer the same doctor if still on duty; otherwise find any idle doctor on duty
                chosen_idx = None
                # prefer same doctor if on duty
                if doc.on_duty(current_time):
                    chosen_idx = doc_idx
                else:
                    for i, d in enumerate(doctors):
                        if d.on_duty(current_time) and (d.busy_until is None or d.busy_until <= current_time):
                            chosen_idx = i
                            break
                # if still None, we must wait until a doctor finishes (i.e. queue remains); to reflect that, we push the patient back and set doc.busy_until=None
                if chosen_idx is None:
                    # No doctor free or on duty right now; put patient back to front and mark this doc idle
                    queue.insert(0, next_pid)
                    doc.busy_until = None
                    continue
                # start service
                patient.start_service_time = current_time
                total_wait += (patient.start_service_time - patient.arrival_time)
                st = exp_sample(mu, rng)
                finish_t = current_time + st
                doctors[chosen_idx].busy_until = finish_t
                push_event(pq, finish_t, 'departure', chosen_idx)
            else:
                # 队列空，医生空闲
                doc.busy_until = None
        else:
            raise RuntimeError("未知事件类型")

    return total_wait

# ---------- 主流程 ----------
def load_schedule_from_json(path: str) -> Tuple[List[float], List[Doctor]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    arrival_rates = data.get("arrival_rates")
    if arrival_rates is None:
        raise ValueError("JSON 中缺少 arrival_rates 字段")
    # parse doctors
    doctors_data = data.get("doctors", [])
    doctors: List[Doctor] = []
    for doc in doctors_data:
        doc_id = doc.get("id", "")
        shifts = []
        for sh in doc.get("shifts", []):
            # shift start/end assumed to be integers in [0,168)
            s = int(sh.get("start"))
            e = int(sh.get("end"))
            shifts.append((s, e))
        doctors.append(Doctor(doc_id, shifts))
    return arrival_rates, doctors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=str, default="input.json",
                        help="D:\pycharm\Industrial_Engineering_Experiments\emergency_doctor_arrangement\result\input\input.json")
    parser.add_argument("--mu", type=float, default=4.0, help="每位医生服务速率 μ（人/小时）")
    parser.add_argument("--runs", type=int, default=5000, help="仿真重复次数（>=5000）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    arrival_rates, doctors_template = load_schedule_from_json(args.schedule)
    if len(arrival_rates) != 168:
        print("警告：arrival_rates 长度不是 168（周小时数）；请检查 JSON 输入。")

    rng_master = random.Random(args.seed)
    total_weekly_waits = []
    for run in range(args.runs):
        # 为保证每次仿真医生是独立对象，复制 doctors（shifts 是只读的）
        doctors = [Doctor(d.id, d.shifts) for d in doctors_template]
        # 每次仿真使用独立 RNG（由 rng_master 派生 seed），以获得可重复但不同的样本
        seed_run = rng_master.randint(0, 2**31-1)
        rng = random.Random(seed_run)
        week_wait = simulate_one_week(arrival_rates, doctors, args.mu, rng)
        total_weekly_waits.append(week_wait)
        # optional progress print every 100 runs
        if (run+1) % 100 == 0:
            print(f"Completed {run+1}/{args.runs} runs; last week total wait = {week_wait:.4f} hours")

    avg_weekly_total_wait = sum(total_weekly_waits) / len(total_weekly_waits)
    # 输出两个格式：小时与分钟
    print("\n--- Simulation Result ---")
    print(f"Runs: {args.runs}")
    print(f"Average total waiting time per week = {avg_weekly_total_wait:.6f} hours "
          f"= {avg_weekly_total_wait*60:.3f} minutes")
    # 如果需要返回每次结果或写 CSV，可在此扩展

if __name__ == "__main__":
    main()
