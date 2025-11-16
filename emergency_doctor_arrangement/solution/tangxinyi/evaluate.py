#!/usr/bin/env python3
"""

功能：
  - 调用 simulation.py 进行仿真
  - 从仿真结果提取平均等待时间
  - 读取 input.json 计算医生总上班时间、借调医生数量
  - 计算题目目标函数：
        目标 = 平均等待时间 + 1.3 * 医生总上班时间 + 20 * 借调医生数量
使用方式：
  python evaluate_schedule.py --input input.json --sim simulation.py
"""

import json
import subprocess
import argparse
import re
from pathlib import Path

def run_simulation(sim_file: str, input_file: str, mu=6.0, runs=1000, seed=42):
    """运行 simulation.py 并返回输出文本"""
    cmd = [
        "python", sim_file,
        "--schedule", input_file,
        "--mu", str(mu),
        "--runs", str(runs),
        "--seed", str(seed)
    ]
    print(f"Running simulation: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("仿真运行错误：", result.stderr)
    return result.stdout


def parse_wait_time(output_text: str) -> float:
    """从仿真输出文本中提取平均等待时间（小时）"""
    pattern = re.compile(r"Average total waiting time per week\s*=\s*([\d.]+)\s*hours")
    match = pattern.search(output_text)
    if match:
        return float(match.group(1))
    else:
        print("未找到平均等待时间信息，输出如下：\n", output_text)
        return 0.0


def compute_schedule_stats(json_path: str):
    """计算医生总上班时间、借调医生数量"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    doctors = data.get("doctors", [])
    total_hours = 0
    borrowed_count = 0

    for doc in doctors:
        if doc.get("origin") == "borrowed":
            borrowed_count += 1
        shifts = doc.get("shifts", [])
        for s in shifts:
            start = int(s["start"])
            end = int(s["end"])
            total_hours += (end - start)

    return total_hours, borrowed_count


def compute_objective(wait_time, total_hours, borrowed_count, c_borrow=20.0):
    """计算题目中的综合目标函数"""
    objective = wait_time + 1.3 * total_hours + c_borrow * borrowed_count
    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="排班 JSON 文件路径（input.json）")
    parser.add_argument("--sim", type=str, required=True, help="simulation.py 文件路径")
    parser.add_argument("--mu", type=float, default=6.0)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    # Step 1: 调用仿真
    output_text = run_simulation(args.sim, args.input, mu=args.mu, runs=args.runs, seed=args.seed)

    # Step 2: 解析仿真输出
    avg_wait = parse_wait_time(output_text)
    print(f"平均总等待时间（小时）: {avg_wait:.4f}")

    # Step 3: 计算排班信息
    total_hours, borrowed_count = compute_schedule_stats(args.input)
    print(f"医生总上班时间: {total_hours} 小时")
    print(f"借调医生数量: {borrowed_count} 人")

    # Step 4: 计算目标函数
    objective = compute_objective(avg_wait, total_hours, borrowed_count)
    print(f"\n目标函数值 = {objective:.4f}")
    print(f"（公式：等待时间 + 1.3 × {total_hours} + 20 × {borrowed_count}）")


if __name__ == "__main__":
    main()
