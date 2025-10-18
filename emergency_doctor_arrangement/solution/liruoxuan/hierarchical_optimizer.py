"""
分层优化框架 - 急诊医生排班优化
结合多种算法优势：贪心构造、禁忌搜索、遗传算法、变邻域搜索
"""

import numpy as np
import random
import copy
import json
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
from collections import deque
import heapq

from validator import MatrixConstraintValidator, convert_doctors_to_matrices
from simulator import Doctor, run_simulation_multiple


class Solution:
    """解的封装类"""
    def __init__(self, first_shifts, second_shifts, objective=float('inf'), 
                 borrowed_count=0, waiting_time=0, work_hours=0):
        self.first_shifts = first_shifts
        self.second_shifts = second_shifts
        self.objective = objective
        self.borrowed_count = borrowed_count
        self.waiting_time = waiting_time
        self.work_hours = work_hours
    
    def copy(self):
        return Solution(
            copy.deepcopy(self.first_shifts),
            copy.deepcopy(self.second_shifts),
            self.objective,
            self.borrowed_count,
            self.waiting_time,
            self.work_hours
        )


class HierarchicalOptimizer:
    """分层优化器主类"""
    
    def __init__(self, arrival_rates: List[float], mu: float,
                 num_internal_doctors: int = 11, max_borrowed_doctors: int = 28,
                 num_days: int = 7, c_borrow: float = 20.0, seed: int = 42):
        
        self.arrival_rates = arrival_rates
        self.mu = mu
        self.num_internal_doctors = num_internal_doctors
        self.max_borrowed_doctors = max_borrowed_doctors
        self.num_doctors = num_internal_doctors + max_borrowed_doctors
        self.num_days = num_days
        self.c_borrow = c_borrow
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 借调医生索引
        self.borrowed_doctors = list(range(num_internal_doctors, self.num_doctors))
        
        # 班次编码
        self.shift_codes = self._init_shift_codes()
        
        # 验证器
        self.validator = MatrixConstraintValidator()
        
        # 最佳解
        self.best_solution = None
        
        # 计算每小时需求
        self.hourly_demand = self._calculate_hourly_demand()
        
        # 性能统计
        self.stats = {
            'evaluations': 0,
            'cache_hits': 0,
            'feasible_solutions': 0
        }
        
        # 评估缓存
        self.evaluation_cache = {}
    
    def _init_shift_codes(self):
        """初始化班次编码"""
        codes = {
            "YY": (0, 7),  # 夜班
            "XX": (None, 0)  # 休息
        }
        # 白班编码
        for i, start in enumerate(range(7, 24)):
            letter = chr(65 + i)  # A-Q
            for j, duration in enumerate(range(3, 9)):
                code = letter + chr(65 + j)
                codes[code] = (start, duration)
        return codes
    
    def _calculate_hourly_demand(self):
        """计算每小时所需医生数量（基于到达率和服务率）"""
        demand = []
        for rate in self.arrival_rates:
            # 使用排队论公式估算需求：λ/μ + 安全系数
            min_doctors = max(1, int(np.ceil(rate / self.mu)))
            # 考虑变异性，增加安全系数
            safety_factor = 1.2 if rate > self.mu * 3 else 1.0
            demand.append(int(np.ceil(min_doctors * safety_factor)))
        return demand
    
    # ============ 第一层：智能初始化 ============
    
    def greedy_construction(self) -> Solution:
        """
        基于需求的贪心构造初始解
        策略：优先在高需求时段安排医生
        """
        print("【第一层】智能初始化 - 贪心构造...")
        
        first_shifts = np.full((self.num_doctors, self.num_days), "XX", dtype=object)
        second_shifts = np.full((self.num_doctors, self.num_days), "XX", dtype=object)
        
        # 计算每天每小时的需求缺口
        coverage = np.zeros(168)  # 当前覆盖的医生数
        
        # 为每个医生分配班次
        doctor_order = list(range(self.num_doctors))
        random.shuffle(doctor_order)
        
        for doctor in doctor_order:
            is_borrowed = doctor in self.borrowed_doctors
            
            # 为每一天分配班次
            for day in range(self.num_days):
                day_start = day * 24
                
                # 找到当天需求缺口最大的时段
                day_demand = []
                for hour in range(24):
                    global_hour = day_start + hour
                    gap = self.hourly_demand[global_hour] - coverage[global_hour]
                    day_demand.append((hour, gap))
                
                # 按需求缺口排序
                day_demand.sort(key=lambda x: x[1], reverse=True)
                
                # 尝试分配第一个班次
                if day_demand[0][1] > 0 or random.random() < 0.5:
                    # 选择合适的班次类型
                    peak_hour = day_demand[0][0]
                    
                    if peak_hour < 7:  # 夜间高峰，安排夜班
                        first_shifts[doctor, day] = "YY"
                        for h in range(7):
                            coverage[day_start + h] += 1
                    else:  # 白天高峰，安排白班
                        # 选择覆盖高峰时段的班次
                        best_shift = self._select_best_shift(peak_hour, day_demand)
                        if best_shift:
                            first_shifts[doctor, day] = best_shift
                            start, duration = self.shift_codes[best_shift]
                            for h in range(duration):
                                if start + h < 24:
                                    coverage[day_start + start + h] += 1
                            
                            # 尝试分配第二个班次
                            if not is_borrowed and random.random() < 0.2:
                                second_shift = self._select_second_shift(
                                    best_shift, day_demand, day_start, coverage
                                )
                                if second_shift:
                                    second_shifts[doctor, day] = second_shift
                                    start2, duration2 = self.shift_codes[second_shift]
                                    for h in range(duration2):
                                        if start2 + h < 24:
                                            coverage[day_start + start2 + h] += 1
        
        # 验证并评估
        solution = Solution(first_shifts, second_shifts)
        if self._is_feasible(solution):
            self._evaluate(solution)
            print(f"  ✓ 构造可行解: 目标值={solution.objective:.2f}, 借调={solution.borrowed_count}")
            return solution
        else:
            print("  ✗ 构造解不可行，使用随机解")
            return self._random_solution()
    
    def _select_best_shift(self, peak_hour, day_demand):
        """选择最佳班次以覆盖高峰时段"""
        best_shift = None
        best_coverage = 0
        
        for code, (start, duration) in self.shift_codes.items():
            if code in ["XX", "YY"] or start is None:
                continue
            if start <= peak_hour < start + duration:
                # 计算该班次覆盖的需求
                coverage = sum(gap for hour, gap in day_demand 
                             if start <= hour < start + duration and gap > 0)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_shift = code
        
        return best_shift or "AB"  # 默认班次
    
    def _select_second_shift(self, first_shift, day_demand, day_start, coverage):
        """选择第二个班次"""
        start1, duration1 = self.shift_codes[first_shift]
        end1 = start1 + duration1
        
        best_shift = None
        best_score = 0
        
        for code, (start, duration) in self.shift_codes.items():
            if code in ["XX", "YY"] or start is None:
                continue
            if start >= end1 + 2:  # 满足间隔要求
                # 计算覆盖价值
                score = 0
                for h in range(duration):
                    hour = start + h
                    if hour < 24:
                        global_hour = day_start + hour
                        gap = self.hourly_demand[global_hour] - coverage[global_hour]
                        score += max(0, gap)
                
                if score > best_score:
                    best_score = score
                    best_shift = code
        
        return best_shift if best_score > 0 else None
    
    def _random_solution(self) -> Solution:
        """生成随机可行解（作为后备）"""
        for _ in range(100):
            first_shifts = np.full((self.num_doctors, self.num_days), "XX", dtype=object)
            second_shifts = np.full((self.num_doctors, self.num_days), "XX", dtype=object)
            
            for doctor in range(self.num_doctors):
                for day in range(self.num_days):
                    if random.random() < 0.6:
                        if random.random() < 0.2:
                            first_shifts[doctor, day] = "YY"
                        else:
                            codes = [c for c in self.shift_codes.keys() 
                                   if c not in ["XX", "YY"]]
                            first_shifts[doctor, day] = random.choice(codes)
            
            solution = Solution(first_shifts, second_shifts)
            if self._is_feasible(solution):
                self._evaluate(solution)
                return solution
        
        # 如果还是无法生成，返回空解
        return Solution(
            np.full((self.num_doctors, self.num_days), "XX", dtype=object),
            np.full((self.num_doctors, self.num_days), "XX", dtype=object)
        )
    
    # ============ 第二层：禁忌搜索 ============
    
    def tabu_search(self, initial_solution: Solution, max_iterations: int = 100) -> Solution:
        """
        禁忌搜索算法
        避免循环，提高搜索效率
        """
        print(f"【第二层】禁忌搜索 (最多{max_iterations}次迭代)...")
        
        current = initial_solution.copy()
        best = current.copy()
        
        # 禁忌表：记录最近的移动
        tabu_list = deque(maxlen=20)
        
        # 频率记忆：记录每个位置被修改的次数
        frequency = np.zeros((self.num_doctors, self.num_days))
        
        no_improve_count = 0
        max_no_improve = 30
        
        for iteration in range(max_iterations):
            # 生成候选邻域解
            candidates = []
            
            # 策略1：修改单个班次
            for _ in range(20):
                neighbor = current.copy()
                doctor = random.randint(0, self.num_doctors - 1)
                day = random.randint(0, self.num_days - 1)
                
                # 避开高频率修改的位置
                if frequency[doctor, day] > 3:
                    continue
                
                move = (doctor, day, 'modify')
                
                # 检查是否在禁忌表中
                if move not in tabu_list:
                    self._modify_shift(neighbor, doctor, day)
                    if self._is_feasible(neighbor):
                        self._evaluate(neighbor, n_simulation_runs=2)
                        candidates.append((neighbor, move))
            
            # 策略2：交换两个医生的某天班次
            for _ in range(10):
                neighbor = current.copy()
                d1, d2 = random.sample(range(self.num_doctors), 2)
                day = random.randint(0, self.num_days - 1)
                
                move = (d1, d2, day, 'swap')
                
                if move not in tabu_list:
                    self._swap_shifts(neighbor, d1, d2, day)
                    if self._is_feasible(neighbor):
                        self._evaluate(neighbor, n_simulation_runs=2)
                        candidates.append((neighbor, move))
            
            if not candidates:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    print(f"  ⚠ 连续{no_improve_count}次无改进，提前结束")
                    break
                continue
            
            # 选择最佳候选
            candidates.sort(key=lambda x: x[0].objective)
            next_solution, move = candidates[0]
            
            # 更新
            current = next_solution
            tabu_list.append(move)
            
            # 更新频率
            if move[0] == 'modify':
                frequency[move[0], move[1]] += 1
            
            # 更新最佳解（特赦准则：即使在禁忌表中，如果比最佳解好也接受）
            if current.objective < best.objective:
                best = current.copy()
                no_improve_count = 0
                print(f"  ✓ 迭代{iteration}: 新最佳解={best.objective:.2f}, 借调={best.borrowed_count}")
            else:
                no_improve_count += 1
            
            if iteration % 20 == 0:
                print(f"  - 迭代{iteration}: 当前={current.objective:.2f}, 最佳={best.objective:.2f}")
        
        print(f"  ✓ 禁忌搜索完成: 最佳目标值={best.objective:.2f}")
        return best
    
    # ============ 第三层：遗传算法 ============
    
    def genetic_algorithm(self, initial_solution: Solution, 
                         population_size: int = 20, 
                         generations: int = 50) -> Solution:
        """
        遗传算法进行全局搜索
        """
        print(f"【第三层】遗传算法 (种群={population_size}, 代数={generations})...")
        
        # 初始化种群
        population = [initial_solution]
        
        # 生成多样化的初始种群
        for _ in range(population_size - 1):
            individual = self._mutate(initial_solution.copy(), mutation_rate=0.3)
            if self._is_feasible(individual):
                self._evaluate(individual, n_simulation_runs=2)
                population.append(individual)
            else:
                population.append(initial_solution.copy())
        
        best = min(population, key=lambda x: x.objective).copy()
        
        for generation in range(generations):
            # 评估种群
            population.sort(key=lambda x: x.objective)
            
            # 选择
            elite_size = max(2, population_size // 5)
            elite = population[:elite_size]
            
            # 交叉和变异生成新个体
            new_population = elite.copy()
            
            while len(new_population) < population_size:
                # 锦标赛选择
                parent1 = self._tournament_selection(population, k=3)
                parent2 = self._tournament_selection(population, k=3)
                
                # 交叉
                if random.random() < 0.7:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # 变异
                if random.random() < 0.3:
                    child = self._mutate(child, mutation_rate=0.1)
                
                # 修复并评估
                if self._is_feasible(child):
                    self._evaluate(child, n_simulation_runs=2)
                    new_population.append(child)
            
            population = new_population
            
            # 更新最佳解
            current_best = min(population, key=lambda x: x.objective)
            if current_best.objective < best.objective:
                best = current_best.copy()
                print(f"  ✓ 第{generation}代: 新最佳解={best.objective:.2f}, 借调={best.borrowed_count}")
            
            if generation % 10 == 0:
                avg_obj = np.mean([ind.objective for ind in population])
                print(f"  - 第{generation}代: 最佳={best.objective:.2f}, 平均={avg_obj:.2f}")
        
        print(f"  ✓ 遗传算法完成: 最佳目标值={best.objective:.2f}")
        return best
    
    def _tournament_selection(self, population, k=3):
        """锦标赛选择"""
        candidates = random.sample(population, min(k, len(population)))
        return min(candidates, key=lambda x: x.objective)
    
    def _crossover(self, parent1: Solution, parent2: Solution) -> Solution:
        """单点交叉"""
        child = parent1.copy()
        
        # 随机选择交叉点
        crossover_doctor = random.randint(0, self.num_doctors - 1)
        
        # 交换部分基因
        for doctor in range(crossover_doctor, self.num_doctors):
            child.first_shifts[doctor, :] = parent2.first_shifts[doctor, :].copy()
            child.second_shifts[doctor, :] = parent2.second_shifts[doctor, :].copy()
        
        return child
    
    def _mutate(self, solution: Solution, mutation_rate: float = 0.1) -> Solution:
        """变异操作"""
        mutated = solution.copy()
        
        for doctor in range(self.num_doctors):
            for day in range(self.num_days):
                if random.random() < mutation_rate:
                    self._modify_shift(mutated, doctor, day)
        
        return mutated
    
    # ============ 第四层：变邻域搜索 ============
    
    def variable_neighborhood_search(self, initial_solution: Solution, 
                                    max_iterations: int = 50) -> Solution:
        """
        变邻域搜索（VNS）
        系统地改变邻域结构，避免局部最优
        """
        print(f"【第四层】变邻域搜索 (最多{max_iterations}次迭代)...")
        
        current = initial_solution.copy()
        best = current.copy()
        
        # 定义邻域结构（从小到大）
        neighborhoods = [
            ('modify_one', 1),      # 修改1个班次
            ('modify_few', 3),      # 修改3个班次
            ('swap_day', 5),        # 交换天数
            ('swap_doctors', 5),    # 交换医生
            ('modify_many', 10),    # 修改多个班次
        ]
        
        k = 0  # 当前邻域索引
        iteration = 0
        
        while iteration < max_iterations and k < len(neighborhoods):
            # 在第k个邻域中搜索
            neighbor_type, perturbation_size = neighborhoods[k]
            
            neighbor = self._generate_neighborhood_solution(
                current, neighbor_type, perturbation_size
            )
            
            if self._is_feasible(neighbor):
                self._evaluate(neighbor, n_simulation_runs=3)
                
                # 局部搜索
                neighbor = self._local_search(neighbor, max_steps=10)
                
                if neighbor.objective < best.objective:
                    best = neighbor.copy()
                    current = neighbor.copy()
                    k = 0  # 回到第一个邻域
                    print(f"  ✓ 迭代{iteration}: 新最佳解={best.objective:.2f}, 邻域={neighbor_type}")
                else:
                    k += 1  # 尝试下一个邻域
            else:
                k += 1
            
            iteration += 1
            
            if iteration % 10 == 0:
                print(f"  - 迭代{iteration}: 当前邻域={neighborhoods[min(k, len(neighborhoods)-1)][0]}, 最佳={best.objective:.2f}")
        
        print(f"  ✓ 变邻域搜索完成: 最佳目标值={best.objective:.2f}")
        return best
    
    def _generate_neighborhood_solution(self, solution: Solution, 
                                       neighbor_type: str, 
                                       perturbation_size: int) -> Solution:
        """生成邻域解"""
        neighbor = solution.copy()
        
        if neighbor_type == 'modify_one':
            doctor = random.randint(0, self.num_doctors - 1)
            day = random.randint(0, self.num_days - 1)
            self._modify_shift(neighbor, doctor, day)
        
        elif neighbor_type == 'modify_few':
            for _ in range(perturbation_size):
                doctor = random.randint(0, self.num_doctors - 1)
                day = random.randint(0, self.num_days - 1)
                self._modify_shift(neighbor, doctor, day)
        
        elif neighbor_type == 'modify_many':
            for _ in range(perturbation_size):
                doctor = random.randint(0, self.num_doctors - 1)
                day = random.randint(0, self.num_days - 1)
                self._modify_shift(neighbor, doctor, day)
        
        elif neighbor_type == 'swap_day':
            for _ in range(perturbation_size):
                doctor = random.randint(0, self.num_doctors - 1)
                d1, d2 = random.sample(range(self.num_days), 2)
                # 交换两天的班次
                neighbor.first_shifts[doctor, [d1, d2]] = neighbor.first_shifts[doctor, [d2, d1]]
                neighbor.second_shifts[doctor, [d1, d2]] = neighbor.second_shifts[doctor, [d2, d1]]
        
        elif neighbor_type == 'swap_doctors':
            for _ in range(perturbation_size):
                doc1, doc2 = random.sample(range(self.num_doctors), 2)
                day = random.randint(0, self.num_days - 1)
                self._swap_shifts(neighbor, doc1, doc2, day)
        
        return neighbor
    
    def _local_search(self, solution: Solution, max_steps: int = 10) -> Solution:
        """局部搜索改进"""
        current = solution.copy()
        
        for _ in range(max_steps):
            improved = False
            
            # 尝试所有单个修改
            for doctor in range(self.num_doctors):
                for day in range(self.num_days):
                    neighbor = current.copy()
                    self._modify_shift(neighbor, doctor, day)
                    
                    if self._is_feasible(neighbor):
                        self._evaluate(neighbor, n_simulation_runs=1)
                        
                        if neighbor.objective < current.objective:
                            current = neighbor
                            improved = True
                            break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return current
    
    # ============ 辅助方法 ============
    
    def _modify_shift(self, solution: Solution, doctor: int, day: int):
        """修改单个班次"""
        # 随机选择新班次
        if random.random() < 0.3:
            solution.first_shifts[doctor, day] = "XX"
        elif random.random() < 0.2:
            solution.first_shifts[doctor, day] = "YY"
            solution.second_shifts[doctor, day] = "XX"
        else:
            codes = [c for c in self.shift_codes.keys() if c not in ["XX", "YY"]]
            solution.first_shifts[doctor, day] = random.choice(codes)
            
            # 可能修改第二个班次
            if random.random() < 0.2:
                solution.second_shifts[doctor, day] = random.choice(codes + ["XX"])
            else:
                solution.second_shifts[doctor, day] = "XX"
    
    def _swap_shifts(self, solution: Solution, doctor1: int, doctor2: int, day: int):
        """交换两个医生某天的班次"""
        solution.first_shifts[[doctor1, doctor2], day] = \
            solution.first_shifts[[doctor2, doctor1], day]
        solution.second_shifts[[doctor1, doctor2], day] = \
            solution.second_shifts[[doctor2, doctor1], day]
    
    def _is_feasible(self, solution: Solution) -> bool:
        """检查解是否可行"""
        first_shifts_list = solution.first_shifts.tolist()
        second_shifts_list = solution.second_shifts.tolist()
        is_feasible, _ = self.validator.validate_matrix_constraints(
            first_shifts_list, second_shifts_list
        )
        if is_feasible:
            self.stats['feasible_solutions'] += 1
        return is_feasible
    
    def _evaluate(self, solution: Solution, n_simulation_runs: int = 3):
        """评估解的目标函数值"""
        # 检查缓存
        cache_key = self._solution_hash(solution)
        if cache_key in self.evaluation_cache:
            cached = self.evaluation_cache[cache_key]
            solution.objective = cached['objective']
            solution.borrowed_count = cached['borrowed_count']
            solution.waiting_time = cached['waiting_time']
            solution.work_hours = cached['work_hours']
            self.stats['cache_hits'] += 1
            return
        
        self.stats['evaluations'] += 1
        
        # 转换为Doctor对象
        doctors = self._convert_to_doctors(solution)
        
        # 运行仿真
        waiting_time = run_simulation_multiple(
            self.arrival_rates, doctors, self.mu,
            n_runs=n_simulation_runs, seed=self.seed
        )
        
        # 计算工作时间
        work_hours = self._calculate_work_hours(solution)
        
        # 计算借调医生数
        borrowed_count = self._count_borrowed_doctors(solution)
        
        # 目标函数
        objective = waiting_time + work_hours * 1.3 + borrowed_count * self.c_borrow
        
        solution.objective = objective
        solution.borrowed_count = borrowed_count
        solution.waiting_time = waiting_time
        solution.work_hours = work_hours
        
        # 缓存结果
        self.evaluation_cache[cache_key] = {
            'objective': objective,
            'borrowed_count': borrowed_count,
            'waiting_time': waiting_time,
            'work_hours': work_hours
        }
    
    def _solution_hash(self, solution: Solution) -> str:
        """生成解的哈希值用于缓存"""
        return hash(solution.first_shifts.tobytes() + solution.second_shifts.tobytes())
    
    def _convert_to_doctors(self, solution: Solution) -> List[Doctor]:
        """转换为Doctor对象列表"""
        doctors = []
        for doctor_idx in range(self.num_doctors):
            if doctor_idx < self.num_internal_doctors:
                doctor_id = f"D{doctor_idx+1}"
            else:
                doctor_id = f"B{doctor_idx+1-self.num_internal_doctors}"
            
            shifts = []
            for day in range(self.num_days):
                day_start_hour = day * 24
                
                # 第一个班次
                first_shift = solution.first_shifts[doctor_idx, day]
                if first_shift == "YY":
                    shifts.append((day_start_hour, day_start_hour + 7))
                elif first_shift != "XX":
                    start, duration = self.shift_codes[first_shift]
                    shifts.append((day_start_hour + start, day_start_hour + start + duration))
                
                # 第二个班次
                second_shift = solution.second_shifts[doctor_idx, day]
                if second_shift != "XX":
                    start, duration = self.shift_codes[second_shift]
                    shifts.append((day_start_hour + start, day_start_hour + start + duration))
            
            doctors.append(Doctor(doctor_id, shifts))
        
        return doctors
    
    def _calculate_work_hours(self, solution: Solution) -> float:
        """计算总工作时间"""
        total_hours = 0
        for doctor in range(self.num_doctors):
            for day in range(self.num_days):
                for shift in [solution.first_shifts[doctor, day], 
                             solution.second_shifts[doctor, day]]:
                    if shift != "XX":
                        total_hours += self.shift_codes[shift][1]
        return total_hours
    
    def _count_borrowed_doctors(self, solution: Solution) -> int:
        """计算活跃借调医生数"""
        count = 0
        for doctor in self.borrowed_doctors:
            has_shift = False
            for day in range(self.num_days):
                if (solution.first_shifts[doctor, day] != "XX" or 
                    solution.second_shifts[doctor, day] != "XX"):
                    has_shift = True
                    break
            if has_shift:
                count += 1
        return count
    
    # ============ 主优化流程 ============
    
    def optimize(self, strategy: str = 'full', initial_solution_file: str = None):
        """
        主优化函数
        
        参数:
        strategy: 'full' - 完整四层优化
                 'fast' - 快速两层优化
                 'quality' - 高质量三层优化
        """
        print("=" * 80)
        print("分层优化框架启动")
        print("=" * 80)
        print(f"策略: {strategy}")
        print(f"医生: 内部{self.num_internal_doctors}人, 借调{self.max_borrowed_doctors}人")
        print(f"需求: {len(self.arrival_rates)}小时, 平均到达率={np.mean(self.arrival_rates):.2f}")
        print("=" * 80)
        
        # 第一层：初始化
        if initial_solution_file:
            print(f"从文件加载初始解: {initial_solution_file}")
            solution = self._load_solution_from_file(initial_solution_file)
        else:
            solution = self.greedy_construction()
        
        self.best_solution = solution.copy()
        print(f"\n初始解: 目标值={solution.objective:.2f}, 借调={solution.borrowed_count}\n")
        
        # 第二层：禁忌搜索
        if strategy in ['full', 'fast', 'quality']:
            solution = self.tabu_search(solution, max_iterations=100 if strategy == 'full' else 50)
            if solution.objective < self.best_solution.objective:
                self.best_solution = solution.copy()
        
        # 第三层：遗传算法
        if strategy in ['full', 'quality']:
            solution = self.genetic_algorithm(
                solution, 
                population_size=20 if strategy == 'full' else 15,
                generations=50 if strategy == 'full' else 30
            )
            if solution.objective < self.best_solution.objective:
                self.best_solution = solution.copy()
        
        # 第四层：变邻域搜索
        if strategy == 'full':
            solution = self.variable_neighborhood_search(solution, max_iterations=50)
            if solution.objective < self.best_solution.objective:
                self.best_solution = solution.copy()
        
        # 最终精细优化
        print("\n【最终优化】精细局部搜索...")
        final_solution = self._local_search(self.best_solution, max_steps=20)
        if final_solution.objective < self.best_solution.objective:
            self.best_solution = final_solution
        
        # 输出结果
        print("\n" + "=" * 80)
        print("优化完成！")
        print("=" * 80)
        print(f"最佳目标值: {self.best_solution.objective:.2f}")
        print(f"  - 等待时间: {self.best_solution.waiting_time:.2f}")
        print(f"  - 工作时间: {self.best_solution.work_hours:.2f}")
        print(f"  - 借调医生: {self.best_solution.borrowed_count}人")
        print(f"  - 借调成本: {self.best_solution.borrowed_count * self.c_borrow:.2f}")
        print("\n性能统计:")
        print(f"  - 总评估次数: {self.stats['evaluations']}")
        print(f"  - 缓存命中: {self.stats['cache_hits']}")
        print(f"  - 可行解数量: {self.stats['feasible_solutions']}")
        print("=" * 80)
        
        return self.best_solution
    
    def _load_solution_from_file(self, file_path: str) -> Solution:
        """从JSON文件加载初始解"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        doctors_data = data.get("doctors", [])
        first_shifts, second_shifts, is_borrowed_list = convert_doctors_to_matrices(
            doctors_data, self.num_days
        )
        
        # 更新医生配置
        self.borrowed_doctors = [i for i, is_borrowed in enumerate(is_borrowed_list) 
                                if is_borrowed]
        self.num_internal_doctors = len(is_borrowed_list) - len(self.borrowed_doctors)
        self.num_doctors = len(is_borrowed_list)
        
        solution = Solution(np.array(first_shifts), np.array(second_shifts))
        if self._is_feasible(solution):
            self._evaluate(solution)
        
        return solution
    
    def save_solution(self, file_path: str):
        """保存最优解到JSON文件"""
        if self.best_solution is None:
            print("没有可保存的解")
            return
        
        result = {
            "metadata": {
                "optimization_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "objective_value": float(self.best_solution.objective),
                "waiting_time": float(self.best_solution.waiting_time),
                "work_hours": float(self.best_solution.work_hours),
                "borrowed_doctors_count": int(self.best_solution.borrowed_count),
                "borrow_cost": float(self.best_solution.borrowed_count * self.c_borrow),
                "evaluations": self.stats['evaluations'],
                "cache_hits": self.stats['cache_hits']
            },
            "arrival_rates": self.arrival_rates,
            "mu": self.mu,
            "c_borrow": self.c_borrow,
            "seed": self.seed,
            "include_overtime_in_cost": False,
            "doctors": []
        }
        
        # 构建医生排班数据
        for doctor_idx in range(self.num_doctors):
            if doctor_idx < self.num_internal_doctors:
                doctor_id = f"D{doctor_idx+1}"
                origin = "internal"
            else:
                doctor_id = f"B{doctor_idx+1-self.num_internal_doctors}"
                origin = "borrowed"
            
            shifts = []
            for day in range(self.num_days):
                day_start_hour = day * 24
                
                # 第一个班次
                first_shift = self.best_solution.first_shifts[doctor_idx, day]
                if first_shift == "YY":
                    shifts.append({
                        "start": day_start_hour,
                        "end": day_start_hour + 7,
                        "tag": "night"
                    })
                elif first_shift != "XX":
                    start, duration = self.shift_codes[first_shift]
                    shifts.append({
                        "start": day_start_hour + start,
                        "end": day_start_hour + start + duration,
                        "tag": "day"
                    })
                
                # 第二个班次
                second_shift = self.best_solution.second_shifts[doctor_idx, day]
                if second_shift != "XX":
                    start, duration = self.shift_codes[second_shift]
                    shifts.append({
                        "start": day_start_hour + start,
                        "end": day_start_hour + start + duration,
                        "tag": "day"
                    })
            
            if shifts:  # 只保存有班次的医生
                result["doctors"].append({
                    "id": doctor_id,
                    "origin": origin,
                    "shifts": shifts
                })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, indent=2, ensure_ascii=False, fp=f)
        
        print(f"\n最优解已保存到: {file_path}")


def load_data_from_json(file_path: str):
    """从JSON文件加载数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    arrival_rates = data.get("arrival_rates")
    mu = data.get("mu", 6.0)
    c_borrow = data.get("c_borrow", 20.0)
    seed = data.get("seed", 42)
    
    return arrival_rates, mu, c_borrow, seed


if __name__ == "__main__":
    # 加载数据
    input_file = "optimal_solution.json"
    
    try:
        arrival_rates, mu, c_borrow, seed = load_data_from_json(input_file)
        print(f"✓ 成功加载数据: 到达率={len(arrival_rates)}小时, μ={mu}, 借调成本={c_borrow}")
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        exit(1)
    
    # 创建优化器
    optimizer = HierarchicalOptimizer(
        arrival_rates=arrival_rates,
        mu=mu,
        num_internal_doctors=11,
        max_borrowed_doctors=28,
        num_days=7,
        c_borrow=c_borrow,
        seed=seed
    )
    
    # 执行优化
    # strategy选项：'full' - 完整优化(最慢但最好), 'quality' - 高质量, 'fast' - 快速
    best_solution = optimizer.optimize(
        strategy='quality',  # 推荐使用quality平衡速度和质量
        initial_solution_file=input_file  # 可选：使用已有解作为起点
    )
    
    # 保存结果
    optimizer.save_solution("hierarchical_optimal_solution.json")

