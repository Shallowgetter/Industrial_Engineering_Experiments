"""
主程序 - 急诊医生排班优化
使用模拟退火算法优化医生排班，调用验证器和仿真器
"""

import numpy as np
import random
import math
import copy
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# 导入验证器和仿真器
from validator import MatrixConstraintValidator, convert_doctors_to_matrices, print_matrices
from simulator import Doctor, run_simulation_multiple

# ========== 模拟退火算法部分 ==========
class EmergencySchedulingSA:
    def __init__(self, arrival_rates: List[float], mu: float,
                 num_internal_doctors: int = 11, max_borrowed_doctors: int = 7,
                 num_days: int = 7, c_borrow: float = 20.0, seed: int = 42):
        """
        初始化模拟退火算法

        参数:
        arrival_rates: 168小时的到达率列表
        mu: 医生服务速率
        num_internal_doctors: 内部医生数量
        max_borrowed_doctors: 最大借调医生数量
        num_days: 排班天数 (7天)
        c_borrow: 借调医生成本
        seed: 随机种子
        """
        # 设置随机种子
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.arrival_rates = arrival_rates
        self.mu = mu
        self.num_internal_doctors = num_internal_doctors
        self.max_borrowed_doctors = max_borrowed_doctors
        self.num_doctors = num_internal_doctors + max_borrowed_doctors
        self.num_days = num_days
        self.c_borrow = c_borrow

        # 记录哪些医生是借调医生
        self.borrowed_doctors = list(range(num_internal_doctors, self.num_doctors))

        # 班次编码定义
        self.shift_codes = {
            # 白班编码: (开始时间, 时长)
            "AA": (7, 3), "AB": (7, 4), "AC": (7, 5), "AD": (7, 6), "AE": (7, 7), "AF": (7, 8),
            "BA": (8, 3), "BB": (8, 4), "BC": (8, 5), "BD": (8, 6), "BE": (8, 7), "BF": (8, 8),
            "CA": (9, 3), "CB": (9, 4), "CC": (9, 5), "CD": (9, 6), "CE": (9, 7), "CF": (9, 8),
            "DA": (10, 3), "DB": (10, 4), "DC": (10, 5), "DD": (10, 6), "DE": (10, 7), "DF": (10, 8),
            "EA": (11, 3), "EB": (11, 4), "EC": (11, 5), "ED": (11, 6), "EE": (11, 7), "EF": (11, 8),
            "FA": (12, 3), "FB": (12, 4), "FC": (12, 5), "FD": (12, 6), "FE": (12, 7), "FF": (12, 8),
            "GA": (13, 3), "GB": (13, 4), "GC": (13, 5), "GD": (13, 6), "GE": (13, 7), "GF": (13, 8),
            "HA": (14, 3), "HB": (14, 4), "HC": (14, 5), "HD": (14, 6), "HE": (14, 7), "HF": (14, 8),
            "IA": (15, 3), "IB": (15, 4), "IC": (15, 5), "ID": (15, 6), "IE": (15, 7), "IF": (15, 8),
            "JA": (16, 3), "JB": (16, 4), "JC": (16, 5), "JD": (16, 6), "JE": (16, 7), "JF": (16, 8),
            "KA": (17, 3), "KB": (17, 4), "KC": (17, 5), "KD": (17, 6), "KE": (17, 7), "KF": (17, 8),
            "LA": (18, 3), "LB": (18, 4), "LC": (18, 5), "LD": (18, 6), "LE": (18, 7), "LF": (18, 8),
            "MA": (19, 3), "MB": (19, 4), "MC": (19, 5), "MD": (19, 6), "ME": (19, 7), "MF": (19, 8),
            "NA": (20, 3), "NB": (20, 4), "NC": (20, 5), "ND": (20, 6), "NE": (20, 7), "NF": (20, 8),
            "OA": (21, 3), "OB": (21, 4), "OC": (21, 5), "OD": (21, 6), "OE": (21, 7), "OF": (21, 8),
            "PA": (22, 3), "PB": (22, 4), "PC": (22, 5), "PD": (22, 6), "PE": (22, 7), "PF": (22, 8),
            "QA": (23, 3), "QB": (23, 4), "QC": (23, 5), "QD": (23, 6), "QE": (23, 7), "QF": (23, 8),
            # 特殊班次
            "YY": (0, 7),  # 夜班
            "XX": (None, 0)  # 无班次
        }

        # 初始化验证器
        self.validator = MatrixConstraintValidator()

        # 记录最佳解
        self.best_solution = None
        self.best_objective = float('inf')
        self.best_borrowed_count = 0

        # 记录已删除的不活跃借调医生
        self.removed_borrowed_doctors = set()

    def create_initial_solution(self):
        """创建初始可行解"""
        # 重置随机种子以确保一致性
        random.seed(self.seed)
        np.random.seed(self.seed)

        first_shifts = np.full((self.num_doctors, self.num_days), "XX", dtype=object)
        second_shifts = np.full((self.num_doctors, self.num_days), "XX", dtype=object)

        # 确保借调医生在初始解中有排班
        for doctor in self.borrowed_doctors:
            # 跳过已删除的借调医生
            if doctor in self.removed_borrowed_doctors:
                continue

            for day in range(self.num_days):
                if random.random() < 0.8:  # 借调医生有80%概率有班次
                    day_shifts = [code for code in self.shift_codes.keys()
                                 if code not in ["YY", "XX"] and code[0] in "ABCDEFGHIJKLMNOPQ"]
                    first_shifts[doctor, day] = random.choice(day_shifts)

        # 为所有医生随机生成排班
        for doctor in range(self.num_doctors):
            # 跳过已删除的借调医生
            if doctor in self.removed_borrowed_doctors:
                continue

            for day in range(self.num_days):
                if doctor in self.borrowed_doctors and first_shifts[doctor, day] != "XX":
                    continue

                if random.random() < 0.7:  # 70%概率有班次
                    if random.random() < 0.2:  # 20%概率夜班
                        first_shifts[doctor, day] = "YY"
                    else:
                        day_shifts = [code for code in self.shift_codes.keys()
                                     if code not in ["YY", "XX"] and code[0] in "ABCDEFGHIJKLMNOPQ"]
                        first_shifts[doctor, day] = random.choice(day_shifts)

                # 如果第一个班次不是夜班，可能有第二个班次
                if first_shifts[doctor, day] != "YY" and first_shifts[doctor, day] != "XX" and random.random() < 0.3:
                    first_shift_info = self.shift_codes[first_shifts[doctor, day]]
                    first_end_time = first_shift_info[0] + first_shift_info[1]

                    available_second_shifts = []
                    for code, (start, duration) in self.shift_codes.items():
                        if code not in ["YY", "XX"] and start >= first_end_time + 2:
                            available_second_shifts.append(code)

                    if available_second_shifts:
                        second_shifts[doctor, day] = random.choice(available_second_shifts)

        return first_shifts, second_shifts

    def load_initial_solution_from_json(self, json_file_path: str):
        """
        从JSON文件加载初始可行解

        参数:
        json_file_path: JSON文件路径，包含doctors字段，每个医生有shifts列表
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        doctors_data = data.get("doctors", [])
        first_shifts, second_shifts, is_borrowed_list = convert_doctors_to_matrices(doctors_data, self.num_days)

        # 更新借调医生列表
        self.borrowed_doctors = [i for i, is_borrowed in enumerate(is_borrowed_list) if is_borrowed]
        self.num_internal_doctors = len(is_borrowed_list) - len(self.borrowed_doctors)
        self.num_doctors = len(is_borrowed_list)

        # 转换为numpy数组
        first_shifts_np = np.array(first_shifts)
        second_shifts_np = np.array(second_shifts)

        return first_shifts_np, second_shifts_np, is_borrowed_list

    def encode_solution(self, first_shifts, second_shifts):
        """将两个矩阵编码为每个医生的排班序列"""
        doctor_schedules = []
        for doctor in range(self.num_doctors):
            # 跳过已删除的借调医生
            if doctor in self.removed_borrowed_doctors:
                continue

            schedule = []
            for day in range(self.num_days):
                schedule.append(first_shifts[doctor, day])
                schedule.append(second_shifts[doctor, day])
            doctor_schedules.append(schedule)
        return doctor_schedules

    def decode_solution(self, doctor_schedules):
        """将医生的排班序列解码为两个矩阵"""
        first_shifts = np.full((self.num_doctors, self.num_days), "XX", dtype=object)
        second_shifts = np.full((self.num_doctors, self.num_days), "XX", dtype=object)

        for doctor_idx, schedule in enumerate(doctor_schedules):
            # 跳过已删除的借调医生
            if doctor_idx in self.removed_borrowed_doctors:
                continue

            for day in range(self.num_days):
                first_shifts[doctor_idx, day] = schedule[day*2]
                second_shifts[doctor_idx, day] = schedule[day*2+1]

        return first_shifts, second_shifts

    def convert_to_doctor_objects(self, first_shifts, second_shifts):
        """将排班矩阵转换为Doctor对象列表"""
        doctors = []
        for doctor_idx in range(self.num_doctors):
            # 跳过已删除的借调医生
            if doctor_idx in self.removed_borrowed_doctors:
                continue

            if doctor_idx < self.num_internal_doctors:
                doctor_id = f"D{doctor_idx+1}"
            else:
                doctor_id = f"B{doctor_idx+1-self.num_internal_doctors}"

            shifts = []
            for day in range(self.num_days):
                day_start_hour = day * 24

                # 处理第一个班次
                first_shift = first_shifts[doctor_idx, day]
                if first_shift == "YY":  # 夜班
                    shifts.append((day_start_hour + 0, day_start_hour + 7))
                elif first_shift != "XX":  # 白班
                    start_time, duration = self.shift_codes[first_shift]
                    shifts.append((day_start_hour + start_time, day_start_hour + start_time + duration))

                # 处理第二个班次
                second_shift = second_shifts[doctor_idx, day]
                if second_shift != "XX":  # 白班
                    start_time, duration = self.shift_codes[second_shift]
                    shifts.append((day_start_hour + start_time, day_start_hour + start_time + duration))

            doctors.append(Doctor(doctor_id, shifts))

        return doctors

    def is_feasible(self, first_shifts, second_shifts):
        """
        验证解是否可行 - 使用外部验证器

        返回:
        (是否可行, 违规列表)
        """
        # 转换为列表格式以兼容验证器
        first_shifts_list = first_shifts.tolist()
        second_shifts_list = second_shifts.tolist()

        # 使用验证器检查矩阵约束
        return self.validator.validate_matrix_constraints(first_shifts_list, second_shifts_list)

    def count_borrowed_doctors_with_shifts(self, first_shifts, second_shifts):
        """计算实际有排班的借调医生数量"""
        count = 0
        for doctor in self.borrowed_doctors:
            # 跳过已删除的借调医生
            if doctor in self.removed_borrowed_doctors:
                continue

            has_shift = False
            for day in range(self.num_days):
                if first_shifts[doctor, day] != "XX" or second_shifts[doctor, day] != "XX":
                    has_shift = True
                    break
            if has_shift:
                count += 1
        return count

    def neighborhood_swap_same_doctor(self, doctor_schedules, doctor_idx, day1, day2):
        """邻域操作1: 同一个医生内部的班次交换"""
        new_schedules = copy.deepcopy(doctor_schedules)

        new_schedules[doctor_idx][day1*2], new_schedules[doctor_idx][day2*2] = \
            new_schedules[doctor_idx][day2*2], new_schedules[doctor_idx][day1*2]
        new_schedules[doctor_idx][day1*2+1], new_schedules[doctor_idx][day2*2+1] = \
            new_schedules[doctor_idx][day2*2+1], new_schedules[doctor_idx][day1*2+1]

        return new_schedules

    def neighborhood_swap_different_doctors(self, doctor_schedules, doctor1, doctor2, day):
        """邻域操作2: 不同医生之间的班次交换"""
        new_schedules = copy.deepcopy(doctor_schedules)

        new_schedules[doctor1][day*2], new_schedules[doctor2][day*2] = \
            new_schedules[doctor2][day*2], new_schedules[doctor1][day*2]
        new_schedules[doctor1][day*2+1], new_schedules[doctor2][day*2+1] = \
            new_schedules[doctor2][day*2+1], new_schedules[doctor1][day*2+1]

        return new_schedules

    def neighborhood_modify_shift(self, doctor_schedules, doctor_idx, day, shift_to_modify):
        """邻域操作3: 修改单个班次"""
        new_schedules = copy.deepcopy(doctor_schedules)

        current_shift = new_schedules[doctor_idx][day*2 + shift_to_modify]

        if shift_to_modify == 0:  # 修改第一个班次
            # 根据当前班次类型选择可能的修改
            if current_shift == "YY":  # 当前是夜班
                # 夜班可以改为白班或休息
                options = ["XX"] + [code for code in self.shift_codes.keys()
                                  if code not in ["YY", "XX"] and code[0] in "ABCDEFGHIJKLMNOPQ"]
                new_shift = random.choice(options)
            elif current_shift == "XX":  # 当前是休息
                # 休息可以改为夜班或白班
                options = ["YY"] + [code for code in self.shift_codes.keys()
                                  if code not in ["YY", "XX"] and code[0] in "ABCDEFGHIJKLMNOPQ"]
                new_shift = random.choice(options)
            else:  # 当前是白班
                # 白班可以改为其他白班、夜班或休息
                options = ["XX", "YY"] + [code for code in self.shift_codes.keys()
                                        if code not in ["YY", "XX"] and code[0] in "ABCDEFGHIJKLMNOPQ" and code != current_shift]
                new_shift = random.choice(options)
        else:  # 修改第二个班次
            if current_shift == "XX":  # 当前是休息
                # 休息可以改为白班，但需要检查第一个班次
                first_shift = new_schedules[doctor_idx][day*2]
                if first_shift != "XX" and first_shift != "YY":
                    first_shift_info = self.shift_codes[first_shift]
                    first_end_time = first_shift_info[0] + first_shift_info[1]

                    available_shifts = []
                    for code, (start, duration) in self.shift_codes.items():
                        if code not in ["YY", "XX"] and start >= first_end_time + 2:
                            available_shifts.append(code)

                    if available_shifts:
                        new_shift = random.choice(available_shifts)
                    else:
                        new_shift = "XX"  # 没有可用班次，保持休息
                else:
                    new_shift = "XX"  # 第一个班次是夜班或休息，不能安排第二个班次
            else:  # 当前是白班
                # 白班可以改为其他白班或休息
                first_shift = new_schedules[doctor_idx][day*2]
                if first_shift != "XX" and first_shift != "YY":
                    first_shift_info = self.shift_codes[first_shift]
                    first_end_time = first_shift_info[0] + first_shift_info[1]

                    available_shifts = ["XX"]  # 可以改为休息
                    for code, (start, duration) in self.shift_codes.items():
                        if code not in ["YY", "XX"] and start >= first_end_time + 2 and code != current_shift:
                            available_shifts.append(code)

                    new_shift = random.choice(available_shifts)
                else:
                    new_shift = "XX"  # 第一个班次是夜班或休息，第二个班次必须为休息

        new_schedules[doctor_idx][day*2 + shift_to_modify] = new_shift
        return new_schedules

    def calculate_total_work_hours(self, first_shifts, second_shifts):
        """计算医生总上班时间"""
        total_work_hours = 0
        for doctor in range(self.num_doctors):
            # 跳过已删除的借调医生
            if doctor in self.removed_borrowed_doctors:
                continue

            for day in range(self.num_days):
                first_shift = first_shifts[doctor, day]
                second_shift = second_shifts[doctor, day]

                if first_shift != "XX":
                    total_work_hours += self.shift_codes[first_shift][1]
                if second_shift != "XX":
                    total_work_hours += self.shift_codes[second_shift][1]

        return total_work_hours

    def calculate_objective(self, first_shifts, second_shifts, n_simulation_runs=5):
        """计算目标函数值"""
        # 首先验证解是否可行
        is_feasible, violations = self.is_feasible(first_shifts, second_shifts)

        if not is_feasible:
            # 如果不可行，返回一个很大的目标函数值
            return float('inf'), 0

        # 转换为Doctor对象
        doctors = self.convert_to_doctor_objects(first_shifts, second_shifts)

        # 运行仿真计算等待时间
        waiting_time = run_simulation_multiple(
            self.arrival_rates, doctors, self.mu,
            n_runs=n_simulation_runs, seed=self.seed
        )

        # 计算医生总上班时间
        total_work_hours = self.calculate_total_work_hours(first_shifts, second_shifts)

        # 计算借调成本 - 只考虑活跃的借调医生
        borrowed_doctors_count = self.count_borrowed_doctors_with_shifts(first_shifts, second_shifts)
        borrow_cost = max(0, borrowed_doctors_count) * self.c_borrow

        # 目标函数值
        objective = waiting_time + total_work_hours * 1.3 + borrow_cost

        return objective, borrowed_doctors_count

    def remove_inactive_borrowed_doctors(self, first_shifts, second_shifts):
        """删除不活跃的借调医生（七天都休息的）"""
        doctors_to_remove = set()

        for doctor in self.borrowed_doctors:
            # 跳过已删除的借调医生
            if doctor in self.removed_borrowed_doctors:
                continue

            all_rest = True
            for day in range(self.num_days):
                if first_shifts[doctor, day] != "XX" or second_shifts[doctor, day] != "XX":
                    all_rest = False
                    break
            if all_rest:
                doctors_to_remove.add(doctor)

        # 删除不活跃的借调医生
        if doctors_to_remove:
            for doctor in doctors_to_remove:
                self.removed_borrowed_doctors.add(doctor)
                print(f"删除不活跃借调医生 {doctor}")

            # 更新借调医生列表
            self.borrowed_doctors = [d for d in self.borrowed_doctors if d not in doctors_to_remove]
            print(f"剩余借调医生数量: {len(self.borrowed_doctors)}")

    def systematic_neighborhood_search(self, doctor_schedules, current_temp, n_simulation_runs=5):
        """
        基于当前解遍历所有元素点，循环使用邻域操作

        参数:
        doctor_schedules: 当前解
        current_temp: 当前温度
        n_simulation_runs: 仿真运行次数

        返回:
        (新解, 是否接受, 新目标值)
        """
        # 解码当前解
        first_shifts, second_shifts = self.decode_solution(doctor_schedules)

        # 删除不活跃的借调医生
        self.remove_inactive_borrowed_doctors(first_shifts, second_shifts)

        best_neighbor = copy.deepcopy(doctor_schedules)
        best_objective = self.best_objective
        accepted = False

        # 从后往前遍历所有医生和天数
        for doctor_idx in range(self.num_doctors-1, -1, -1):
            # 跳过已删除的借调医生
            if doctor_idx in self.removed_borrowed_doctors:
                continue

            for day_idx in range(self.num_days-1, -1, -1):
                # 循环使用三种邻域操作
                for operation in range(3):
                    new_schedules = None

                    if operation == 0:  # 同一个医生内部的班次交换
                        # 随机选择另一个天进行交换
                        other_day = random.choice([d for d in range(self.num_days) if d != day_idx])
                        new_schedules = self.neighborhood_swap_same_doctor(
                            doctor_schedules, doctor_idx, day_idx, other_day
                        )

                    elif operation == 1:  # 不同医生之间的班次交换
                        # 随机选择另一个医生进行交换，跳过已删除的借调医生
                        available_doctors = [d for d in range(self.num_doctors)
                                           if d != doctor_idx and d not in self.removed_borrowed_doctors]
                        if available_doctors:
                            other_doctor = random.choice(available_doctors)
                            new_schedules = self.neighborhood_swap_different_doctors(
                                doctor_schedules, doctor_idx, other_doctor, day_idx
                            )

                    elif operation == 2:  # 修改单个班次
                        # 随机选择修改第一个或第二个班次
                        shift_to_modify = random.choice([0, 1])
                        new_schedules = self.neighborhood_modify_shift(
                            doctor_schedules, doctor_idx, day_idx, shift_to_modify
                        )

                    # 如果生成了新解，则验证可行性
                    if new_schedules is not None:
                        # 解码并验证可行性
                        new_first_shifts, new_second_shifts = self.decode_solution(new_schedules)
                        is_feasible, violations = self.is_feasible(new_first_shifts, new_second_shifts)

                        if is_feasible:
                            # 计算新解的目标函数值
                            new_objective, new_borrowed_count = self.calculate_objective(
                                new_first_shifts, new_second_shifts, n_simulation_runs
                            )

                            # 判断是否接受新解
                            delta_e = new_objective - self.best_objective

                            if delta_e < 0 or random.random() < math.exp(-delta_e / current_temp):
                                # 接受新解
                                best_neighbor = new_schedules
                                best_objective = new_objective
                                accepted = True

                                # 如果找到更好的解，立即返回
                                if delta_e < 0:
                                    return best_neighbor, accepted, best_objective

        return best_neighbor, accepted, best_objective

    def simulated_annealing(self, max_iterations=1000, n_simulation_runs=5, initial_solution_file=None):
        """模拟退火主算法 - 改进版本"""
        # 重置随机种子
        random.seed(self.seed)
        np.random.seed(self.seed)

        # 重置已删除的借调医生集合
        self.removed_borrowed_doctors = set()

        # 创建初始解
        if initial_solution_file:
            print(f"从文件加载初始解: {initial_solution_file}")
            first_shifts, second_shifts, is_borrowed_list = self.load_initial_solution_from_json(initial_solution_file)
            # 保存借调医生信息
            self.is_borrowed_list = is_borrowed_list
        else:
            print("使用随机初始解")
            first_shifts, second_shifts = self.create_initial_solution()
            self.is_borrowed_list = [False] * self.num_doctors
            for i in self.borrowed_doctors:
                if i < len(self.is_borrowed_list):
                    self.is_borrowed_list[i] = True

        doctor_schedules = self.encode_solution(first_shifts, second_shifts)

        # 计算初始目标函数值
        current_objective, current_borrowed_count = self.calculate_objective(
            first_shifts, second_shifts, n_simulation_runs
        )
        self.best_solution = copy.deepcopy(doctor_schedules)
        self.best_objective = current_objective
        self.best_borrowed_count = current_borrowed_count

        print(f"初始解目标函数值: {current_objective:.2f}, 借调医生数: {current_borrowed_count}")

        # 改进的温度参数
        initial_temp = 5000  # 提高初始温度
        current_temp = initial_temp
        min_temp = 0.1  # 降低最小温度
        cooling_rate = 0.995  # 使用乘法冷却，更慢的冷却速率

        iteration = 0
        consecutive_rejections = 0  # 记录连续拒绝次数
        max_consecutive_rejections = 50  # 最大连续拒绝次数

        while current_temp > min_temp and iteration < max_iterations:
            # 使用系统性的邻域搜索
            new_schedules, accepted, new_objective = self.systematic_neighborhood_search(
                doctor_schedules, current_temp, n_simulation_runs
            )

            # 额外检查：确保新解可行
            if accepted:
                new_first_shifts, new_second_shifts = self.decode_solution(new_schedules)
                is_feasible, violations = self.is_feasible(new_first_shifts, new_second_shifts)

                if not is_feasible:
                    # 如果新解实际上不可行，拒绝它
                    accepted = False
                    new_objective = float('inf')
                    print(f"迭代 {iteration}: 发现不可行解被错误接受，已拒绝")
                else:
                    # 双重检查目标函数值
                    verified_objective, _ = self.calculate_objective(
                        new_first_shifts, new_second_shifts, n_simulation_runs
                    )
                    if abs(new_objective - verified_objective) > 1e-6:
                        print(f"迭代 {iteration}: 目标函数值不一致 {new_objective:.2f} vs {verified_objective:.2f}")
                        new_objective = verified_objective

            if accepted:
                # 接受新解
                doctor_schedules = new_schedules
                current_objective = new_objective
                consecutive_rejections = 0  # 重置连续拒绝计数

                # 更新最佳解
                if current_objective < self.best_objective:
                    self.best_solution = copy.deepcopy(doctor_schedules)
                    self.best_objective = current_objective
                    # 更新借调医生数量
                    first_shifts, second_shifts = self.decode_solution(doctor_schedules)
                    self.best_borrowed_count = self.count_borrowed_doctors_with_shifts(first_shifts, second_shifts)

            else:
                consecutive_rejections += 1

            # 每10次迭代输出当前状态
            if iteration % 10 == 0:
                # 计算当前解的借调医生数量
                current_first_shifts, current_second_shifts = self.decode_solution(doctor_schedules)
                current_borrowed_count = self.count_borrowed_doctors_with_shifts(current_first_shifts, current_second_shifts)
                print(f"迭代 {iteration}: 当前温度 = {current_temp:.2f}, 当前目标函数值 = {current_objective:.2f}, 借调医生数 = {current_borrowed_count}")

            # 如果连续拒绝太多次，提前终止
            if consecutive_rejections >= max_consecutive_rejections:
                print(f"连续拒绝 {consecutive_rejections} 次，提前终止")
                break

            # 降温
            current_temp *= cooling_rate
            iteration += 1

        # 最终验证最佳解
        if self.best_solution is not None:
            best_first_shifts, best_second_shifts = self.decode_solution(self.best_solution)
            is_feasible, violations = self.is_feasible(best_first_shifts, best_second_shifts)
            if not is_feasible:
                print("警告: 最佳解在最终验证中不可行")
                self.best_objective = float('inf')
            else:
                # 重新计算最佳解的目标函数值
                verified_objective, verified_borrowed_count = self.calculate_objective(
                    best_first_shifts, best_second_shifts, n_simulation_runs
                )
                if abs(self.best_objective - verified_objective) > 1e-6:
                    print(f"最佳解目标函数值修正: {self.best_objective:.2f} -> {verified_objective:.2f}")
                    self.best_objective = verified_objective
                    self.best_borrowed_count = verified_borrowed_count

        # 输出最终结果
        final_first_shifts, final_second_shifts = self.decode_solution(doctor_schedules)
        final_borrowed_count = self.count_borrowed_doctors_with_shifts(final_first_shifts, final_second_shifts)
        print(f"最终迭代 {iteration}: 当前目标函数值 = {current_objective:.2f}, 借调医生数 = {final_borrowed_count}")

        print(f"优化完成，最佳目标函数值: {self.best_objective:.2f}")
        return self.best_solution, self.best_objective, self.best_borrowed_count

    def print_solution(self, doctor_schedules):
        """打印排班结果"""
        first_shifts, second_shifts = self.decode_solution(doctor_schedules)

        print("=" * 80)
        print("医生排班结果:")
        print("=" * 80)

        days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

        # 打印第一个矩阵
        print("\n第一个班次矩阵 (第一个班次):")
        print("医生\\日期", end="\t")
        for day in range(self.num_days):
            print(days[day], end="\t")
        print()

        for doctor in range(self.num_doctors):
            # 跳过已删除的借调医生
            if doctor in self.removed_borrowed_doctors:
                continue

            doctor_type = "借调" if doctor in self.borrowed_doctors else "内部"
            print(f"医生{doctor+1}({doctor_type})", end="\t")
            for day in range(self.num_days):
                print(first_shifts[doctor, day], end="\t")
            print()

        # 打印第二个矩阵
        print("\n第二个班次矩阵 (第二个班次):")
        print("医生\\日期", end="\t")
        for day in range(self.num_days):
            print(days[day], end="\t")
        print()

        for doctor in range(self.num_doctors):
            # 跳过已删除的借调医生
            if doctor in self.removed_borrowed_doctors:
                continue

            doctor_type = "借调" if doctor in self.borrowed_doctors else "内部"
            print(f"医生{doctor+1}({doctor_type})", end="\t")
            for day in range(self.num_days):
                print(second_shifts[doctor, day], end="\t")
            print()

        # 打印详细的排班解释
        print("\n" + "=" * 80)
        print("详细排班解释:")
        print("=" * 80)

        for doctor in range(self.num_doctors):
            # 跳过已删除的借调医生
            if doctor in self.removed_borrowed_doctors:
                continue

            doctor_type = "借调" if doctor in self.borrowed_doctors else "内部"
            print(f"\n医生 {doctor+1} ({doctor_type}):")
            for day in range(self.num_days):
                first_shift = first_shifts[doctor, day]
                second_shift = second_shifts[doctor, day]

                shift_str = ""
                if first_shift != "XX":
                    if first_shift == "YY":
                        shift_str += "夜班(0:00-7:00)"
                    else:
                        start, duration = self.shift_codes[first_shift]
                        shift_str += f"白班1({start}:00-{start+duration}:00)"

                if second_shift != "XX":
                    if shift_str:
                        shift_str += " + "
                    start, duration = self.shift_codes[second_shift]
                    shift_str += f"白班2({start}:00-{start+duration}:00)"

                if not shift_str:
                    shift_str = "休息"

                print(f"  {days[day]}: {shift_str}")

    def save_solution_as_json(self, doctor_schedules, file_path=None):
        """
        将最优解保存为JSON格式

        参数:
        doctor_schedules: 医生排班序列
        file_path: 保存文件路径，如果为None则只返回JSON字符串
        """
        first_shifts, second_shifts = self.decode_solution(doctor_schedules)

        # 构建JSON数据结构
        result = {
            "metadata": {
                "optimization_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "objective_value": self.best_objective,
                "borrowed_doctors_count": self.best_borrowed_count,
                "arrival_rates": self.arrival_rates,
                "mu": self.mu,
                "c_borrow": self.c_borrow
            },
            "doctors": []
        }

        # 构建医生排班数据
        for doctor_idx in range(self.num_doctors):
            # 跳过已删除的借调医生
            if doctor_idx in self.removed_borrowed_doctors:
                continue

            # 确定医生ID和类型
            if doctor_idx < self.num_internal_doctors:
                doctor_id = f"D{doctor_idx+1}"
                origin = "internal"
            else:
                doctor_id = f"B{doctor_idx+1-self.num_internal_doctors}"
                origin = "borrowed"

            # 构建班次列表
            shifts = []
            for day in range(self.num_days):
                day_start_hour = day * 24

                # 处理第一个班次
                first_shift = first_shifts[doctor_idx, day]
                if first_shift == "YY":  # 夜班
                    shifts.append({
                        "start": day_start_hour + 0,
                        "end": day_start_hour + 7,
                        "tag": "night"
                    })
                elif first_shift != "XX":  # 白班
                    start_time, duration = self.shift_codes[first_shift]
                    shifts.append({
                        "start": day_start_hour + start_time,
                        "end": day_start_hour + start_time + duration,
                        "tag": "day"
                    })

                # 处理第二个班次
                second_shift = second_shifts[doctor_idx, day]
                if second_shift != "XX":  # 白班
                    start_time, duration = self.shift_codes[second_shift]
                    shifts.append({
                        "start": day_start_hour + start_time,
                        "end": day_start_hour + start_time + duration,
                        "tag": "day"
                    })

            # 添加医生数据
            result["doctors"].append({
                "id": doctor_id,
                "origin": origin,
                "shifts": shifts
            })

        # 转换为JSON字符串
        json_str = json.dumps(result, indent=2, ensure_ascii=False)

        # 保存到文件或返回字符串
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"最优解已保存到: {file_path}")

        return json_str

def load_data_from_excel(file_path: str):
    """从Excel文件加载数据"""
    # 读取第一个sheet的数据
    df = pd.read_excel(file_path, sheet_name='数据')

    # 提取到达率数据 (第6行开始，第2列到第25列，共7天)
    arrival_rates = []
    for day in range(7):  # day1到day7
        row_data = df.iloc[day + 5, 1:25].values  # 第6-12行，B-Y列
        arrival_rates.extend(row_data)

    # 提取服务速率 (从B2单元格)
    mu_cell = df.iloc[1, 1]  # B2单元格
    if isinstance(mu_cell, str) and 'μ:' in mu_cell:
        # 解析字符串格式 "医生服务速率 μ:6 人每小时"
        mu = float(mu_cell.split('μ:')[1].split(' ')[0])
    else:
        mu = 6.0  # 默认值

    return arrival_rates, mu

def load_data_from_json(file_path: str):
    """从JSON文件加载数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    arrival_rates = data.get("arrival_rates")
    mu = data.get("mu", 6.0)
    c_borrow = data.get("c_borrow", 20.0)

    return arrival_rates, mu, c_borrow

# 使用示例
if __name__ == "__main__":
    # 从文件加载数据
    initial_solution_file = "optimal_solution.json"  # 包含初始可行解的JSON文件

    try:
        # 尝试从JSON文件加载数据
        arrival_rates, mu, c_borrow = load_data_from_json(initial_solution_file)
        print(f"成功从JSON加载数据: 到达率长度={len(arrival_rates)}, 服务速率μ={mu}, 借调成本={c_borrow}")
    except Exception as e:
        print(f"从JSON加载数据失败: {e}")
        try:
            # 尝试从Excel加载数据
            arrival_rates, mu = load_data_from_excel('original_data.xlsx')
            c_borrow = 20.0
            print(f"成功从Excel加载数据: 到达率长度={len(arrival_rates)}, 服务速率μ={mu}")
        except Exception as e2:
            print(f"从Excel加载数据失败: {e2}")
            print("使用默认数据...")
            # 使用默认数据作为后备
            arrival_rates = [1.0] * 168
            mu = 6.0
            c_borrow = 20.0

    # 创建模拟退火算法实例
    sa = EmergencySchedulingSA(
        arrival_rates=arrival_rates,
        mu=mu,
        num_internal_doctors=11,
        max_borrowed_doctors=7,
        num_days=7,
        c_borrow=c_borrow,
        seed=42  # 设置随机种子
    )

    # 运行算法 (使用初始解文件)
    print("开始模拟退火优化...")
    best_solution, best_objective, best_borrowed_count = sa.simulated_annealing(
        max_iterations=500,  # 减少迭代次数，因为每次迭代做更多工作
        n_simulation_runs=3,  # 仿真次数
        initial_solution_file=initial_solution_file  # 使用初始解文件
    )

    # 打印结果
    print("\n" + "=" * 80)
    print("最终结果:")
    print("=" * 80)
    print(f"最佳目标函数值: {best_objective:.2f}")
    print(f"实际借调医生数量: {best_borrowed_count}")

    # 打印两个矩阵和详细排班
    sa.print_solution(best_solution)

    # 保存最优解为JSON格式
    print("\n" + "=" * 80)
    print("最优解JSON格式:")
    print("=" * 80)
    json_output = sa.save_solution_as_json(best_solution, "optimal_solution_1.json")
    print(json_output)