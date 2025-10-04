import json
from er_benchmark import simulate_and_score
from gurobi_two_stage import optimize_schedule   # 你之前写的第一阶段求解器

def compute_objective(score, c_borrow=20.0):
    """
    根据模拟结果计算目标值：
    等待时间 + 1.3 * 总工时 + 借调成本
    """
    wait = score.get("avg_wait_time", 0.0)
    staff_hours = score.get("staff_hours", 0.0)
    borrowed = score.get("borrowed_doctors", 0.0)

    return wait + 1.3 * staff_hours + c_borrow * borrowed

def optimize_with_feedback(arrival_rates, mu, num_doctors=11, c_borrow=20.0, max_iter=5):
    """
    两阶段优化：第一阶段生成可行解，第二阶段结合仿真反馈优化目标
    """
    # -------------------------
    # 第一阶段：生成可行解
    # -------------------------
    schedule = optimize_schedule(arrival_rates, mu, num_doctors, c_borrow)
    best_score = simulate_and_score(schedule)
    best_obj = compute_objective(best_score, c_borrow)

    print("Initial score:", best_score)
    print("Initial objective:", best_obj)

    best_schedule = schedule

    # -------------------------
    # 第二阶段：迭代优化
    # -------------------------
    for it in range(max_iter):
        # 用第一阶段方法重新求解，但在目标中考虑 feedback
        # 简单做法：把 c_borrow 调整成一个动态值
        dynamic_c_borrow = c_borrow + it * 5   # 每次加大对借调的惩罚

        new_schedule = optimize_schedule(arrival_rates, mu, num_doctors, dynamic_c_borrow)
        new_score = simulate_and_score(new_schedule)
        new_obj = compute_objective(new_score, c_borrow)

        print(f"[Iter {it+1}] Score:", new_score)
        print(f"[Iter {it+1}] Objective:", new_obj)

        if new_obj < best_obj:
            best_schedule = new_schedule
            best_score = new_score
            best_obj = new_obj
            print(f"--> Improved solution found at iter {it+1}")

    return best_schedule, best_score, best_obj


# -------------------------
# 使用示例
# -------------------------
if __name__ == "__main__":
    # 示例：一周168小时的到达率
    arrival_rates = [5.0] * (7*24)
    mu = 6.0

    schedule, score, obj = optimize_with_feedback(
        arrival_rates, mu,
        num_doctors=11,
        c_borrow=20.0,
        max_iter=5
    )

    # 输出结果
    with open("solution_final.json", "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2)

    print("Final best score:", score)
    print("Final best objective:", obj)
    print("Final schedule exported to solution_final.json")
