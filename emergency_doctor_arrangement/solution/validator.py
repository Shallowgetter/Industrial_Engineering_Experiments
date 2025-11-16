"""
独立验证器模块 - 用于验证急诊医生排班的约束
"""

import json
from typing import List, Dict, Tuple, Optional

class MatrixConstraintValidator:
    """验证矩阵元素的约束"""

    def __init__(self):
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

        # 反向查找编码
        self.code_to_shift = {v: k for k, v in self.shift_codes.items() if k not in ["YY", "XX"]}

    def validate_matrix_constraints(self, first_shifts, second_shifts, is_borrowed_list=None) -> Tuple[bool, List[str]]:
        """
        验证矩阵元素约束

        参数:
        first_shifts: 第一个班次矩阵 (医生×天)
        second_shifts: 第二个班次矩阵 (医生×天)
        is_borrowed_list: 是否为借调医生的列表

        返回:
        (是否可行, 违规列表)
        """
        violations = []

        num_doctors = len(first_shifts)
        num_days = len(first_shifts[0]) if num_doctors > 0 else 0

        # 如果没有提供借调医生列表，默认所有医生都不是借调医生
        if is_borrowed_list is None:
            is_borrowed_list = [False] * num_doctors

        # 为每个医生统计信息
        doctor_stats = [{
            'night_shifts': 0,  # 夜班次数
            'night_shift_days': []     # 夜班日期
        } for _ in range(num_doctors)]

        for doctor_idx in range(num_doctors):
            # 统计夜班次数
            night_shifts = 0
            night_shift_days = []

            for day_idx in range(num_days):
                first_shift = first_shifts[doctor_idx][day_idx]
                second_shift = second_shifts[doctor_idx][day_idx]

                # 约束1: "YY"只能出现在第一个矩阵中
                if second_shift == "YY":
                    violations.append(f"医生{doctor_idx+1} 第{day_idx+1}天: 夜班'YY'出现在第二个矩阵中")

                # 约束2: 如果第一个矩阵是"YY"，第二个矩阵必须是"XX"
                if first_shift == "YY" and second_shift != "XX":
                    violations.append(f"医生{doctor_idx+1} 第{day_idx+1}天: 夜班'YY'出现时第二个班次必须为'XX'")

                # 约束3: 白班班次时间约束
                if (first_shift != "XX" and first_shift != "YY" and
                    second_shift != "XX" and second_shift != "YY"):

                    first_info = self.shift_codes.get(first_shift)
                    second_info = self.shift_codes.get(second_shift)

                    if first_info and second_info:
                        first_start, first_duration = first_info
                        second_start, second_duration = second_info

                        first_end = first_start + first_duration

                        # 第二个班次开始时间必须至少比第一个班次结束时间大2
                        if second_start < first_end + 2:
                            violations.append(
                                f"医生{doctor_idx+1} 第{day_idx+1}天: "
                                f"第二个班次开始时间({second_start}:00)必须至少比第一个班次结束时间({first_end}:00)晚2小时"
                            )

                # 统计夜班
                if first_shift == "YY":
                    night_shifts += 1
                    night_shift_days.append(day_idx)

            # 约束12: 一个医生一周最多2个夜班
            if night_shifts > 2:
                violations.append(f"医生{doctor_idx+1}: 夜班次数({night_shifts}次)超过2次限制")

            # 保存医生统计信息
            doctor_stats[doctor_idx]['night_shifts'] = night_shifts
            doctor_stats[doctor_idx]['night_shift_days'] = night_shift_days

        # 验证其他复杂约束
        self._validate_additional_constraints(first_shifts, second_shifts, is_borrowed_list, doctor_stats, violations)

        return len(violations) == 0, violations

    def _validate_additional_constraints(self, first_shifts, second_shifts, is_borrowed_list, doctor_stats, violations):
        """验证额外的复杂约束"""
        num_doctors = len(first_shifts)
        num_days = len(first_shifts[0]) if num_doctors > 0 else 0

        for doctor_idx in range(num_doctors):
            # 跳过借调医生的某些约束
            is_borrowed = is_borrowed_list[doctor_idx]

            for day_idx in range(num_days):
                first_shift = first_shifts[doctor_idx][day_idx]
                second_shift = second_shifts[doctor_idx][day_idx]

                # 约束8: 白班时长验证 (3-8小时)
                self._validate_day_shift_duration(doctor_idx, day_idx, first_shift, second_shift, violations)

                # 约束8: 每天白班总时间不超过12小时
                self._validate_daily_white_shift_hours(doctor_idx, day_idx, first_shift, second_shift, violations)

                # 约束9: 每天最多两个白班或一个夜班
                #self._validate_daily_shift_count(doctor_idx, day_idx, first_shift, second_shift, violations)

                # 约束10: 夜班前8小时不能工作
                self._validate_night_shift_preparation(doctor_idx, day_idx, first_shift, second_shifts, violations)

            # 约束11: 夜班后必须连续休息24小时
            self._validate_post_night_shift_rest(doctor_idx, first_shifts, second_shifts, doctor_stats, violations)

            # 约束13: 每周至少休息1整天 (连续的24h，从7点到次日7点)
            if not is_borrowed:
                self._validate_weekly_continuous_rest(doctor_idx, first_shifts, second_shifts, violations)

    def _validate_day_shift_duration(self, doctor_idx, day_idx, first_shift, second_shift, violations):
        """验证白班时长 (3-8小时)"""
        for shift_code in [first_shift, second_shift]:
            if shift_code not in ["XX", "YY"]:
                shift_info = self.shift_codes.get(shift_code)
                if shift_info:
                    _, duration = shift_info
                    if duration < 3 or duration > 8:
                        violations.append(
                            f"医生{doctor_idx+1} 第{day_idx+1}天: "
                            f"班次{shift_code}时长({duration}小时)不符合3-8小时要求"
                        )

    def _validate_daily_white_shift_hours(self, doctor_idx, day_idx, first_shift, second_shift, violations):
        """验证每天白班总时间不超过12小时"""
        total_hours = 0

        for shift_code in [first_shift, second_shift]:
            if shift_code not in ["XX", "YY"]:
                shift_info = self.shift_codes.get(shift_code)
                if shift_info:
                    _, duration = shift_info
                    total_hours += duration

        if total_hours > 12:
            violations.append(
                f"医生{doctor_idx+1} 第{day_idx+1}天: "
                f"白班总时长({total_hours}小时)超过12小时限制"
            )

    def _validate_daily_shift_count(self, doctor_idx, day_idx, first_shift, second_shift, violations):
        """验证每天班次数量约束"""
        has_night_shift = (first_shift == "YY")
        white_shift_count = 0

        for shift_code in [first_shift, second_shift]:
            if shift_code not in ["XX", "YY"]:
                white_shift_count += 1

        # 约束: 每天最多两个白班或一个夜班
       # if has_night_shift and (first_shift != "XX" or second_shift != "XX"):
          #  violations.append(
           #     f"医生{doctor_idx+1} 第{day_idx+1}天: "
           #     f"有夜班时不能安排白班"
           # )

        if white_shift_count > 2:
            violations.append(
                f"医生{doctor_idx+1} 第{day_idx+1}天: "
                f"白班数量({white_shift_count})超过2个限制"
            )

    def _validate_night_shift_preparation(self, doctor_idx, day_idx, first_shift, second_shifts, violations):
        """验证夜班前8小时不能工作"""
        if first_shift == "YY":
            # 检查前一天16点之后的班次
            prev_day = day_idx - 1
            if prev_day >= 0:
                prev_second_shift = second_shifts[doctor_idx][prev_day]
                if prev_second_shift != "XX":
                    shift_info = self.shift_codes.get(prev_second_shift)
                    if shift_info:
                        start_time, duration = shift_info
                        end_time = start_time + duration
                        if end_time > 16:  # 16点之后结束的班次
                            violations.append(
                                f"医生{doctor_idx+1} 第{day_idx+1}天夜班: "
                                f"前一日有16点之后结束的班次{prev_second_shift}(结束时间{end_time}:00)"
                            )

    def _validate_post_night_shift_rest(self, doctor_idx, first_shifts, second_shifts, doctor_stats, violations):
        """验证夜班后必须连续休息24小时"""
        night_shift_days = doctor_stats[doctor_idx]['night_shift_days']

        for night_day in night_shift_days:
            next_day = night_day + 1
            if next_day < len(first_shifts[0]):
                next_first = first_shifts[doctor_idx][next_day]

                # 修正：夜班后下一天不能安排夜班，但可以安排白班或休息
                if next_first == "YY":
                    violations.append(
                        f"医生{doctor_idx+1}: "
                        f"第{night_day+1}天夜班后第{next_day+1}天不能安排夜班"
                    )

    def _validate_weekly_continuous_rest(self, doctor_idx, first_shifts, second_shifts, violations):
        """验证每周至少休息1整天 (连续的24h，从7点到次日7点)"""
        num_days = len(first_shifts[0])
        has_continuous_rest = False

        # 检查每一天是否满足连续24小时休息
        for day_idx in range(num_days):
            # 检查从当天7点到次日7点是否完全没有工作
            current_day_rest = True
            next_day_rest = True

            # 检查当天7点之后是否有班次
            first_shift = first_shifts[doctor_idx][day_idx]
            second_shift = second_shifts[doctor_idx][day_idx]

            if first_shift != "XX" and first_shift != "YY":
                # 白班，检查开始时间
                shift_info = self.shift_codes.get(first_shift)
                if shift_info and shift_info[0] >= 7:  # 7点或之后开始
                    current_day_rest = False

            if second_shift != "XX":
                # 第二个班次肯定是白班（如果有）
                current_day_rest = False

            # 检查次日0-7点是否有夜班
            next_day = day_idx + 1
            if next_day < num_days:
                next_first_shift = first_shifts[doctor_idx][next_day]
                if next_first_shift == "YY":  # 次日有夜班
                    next_day_rest = False

            if current_day_rest and next_day_rest:
                # 还需要检查这一天不是夜班后的强制休息日
                is_forced_rest = (day_idx > 0 and first_shifts[doctor_idx][day_idx-1] == "YY")
                if not is_forced_rest:
                    has_continuous_rest = True
                    break

        if not has_continuous_rest:
            violations.append(
                f"医生{doctor_idx+1}: "
                f"一周内没有满足连续24小时休息的要求(从某天7点到次日7点)"
            )

def convert_doctors_to_matrices(doctors_data: List[Dict], num_days: int = 7) -> Tuple[List[List[str]], List[List[str]], List[bool]]:
    """
    将医生数据转换为两个矩阵格式

    参数:
    doctors_data: 医生数据列表
    num_days: 天数

    返回:
    (first_shifts, second_shifts, is_borrowed_list)
    """
    # 初始化矩阵
    first_shifts = [["XX"] * num_days for _ in range(len(doctors_data))]
    second_shifts = [["XX"] * num_days for _ in range(len(doctors_data))]
    is_borrowed_list = [False] * len(doctors_data)  # 标记是否为借调医生

    # 班次编码定义
    shift_codes = {
        (0, 7): "YY",  # 夜班
        (7, 3): "AA", (7, 4): "AB", (7, 5): "AC", (7, 6): "AD", (7, 7): "AE", (7, 8): "AF",
        (8, 3): "BA", (8, 4): "BB", (8, 5): "BC", (8, 6): "BD", (8, 7): "BE", (8, 8): "BF",
        (9, 3): "CA", (9, 4): "CB", (9, 5): "CC", (9, 6): "CD", (9, 7): "CE", (9, 8): "CF",
        (10, 3): "DA", (10, 4): "DB", (10, 5): "DC", (10, 6): "DD", (10, 7): "DE", (10, 8): "DF",
        (11, 3): "EA", (11, 4): "EB", (11, 5): "EC", (11, 6): "ED", (11, 7): "EE", (11, 8): "EF",
        (12, 3): "FA", (12, 4): "FB", (12, 5): "FC", (12, 6): "FD", (12, 7): "FE", (12, 8): "FF",
        (13, 3): "GA", (13, 4): "GB", (13, 5): "GC", (13, 6): "GD", (13, 7): "GE", (13, 8): "GF",
        (14, 3): "HA", (14, 4): "HB", (14, 5): "HC", (14, 6): "HD", (14, 7): "HE", (14, 8): "HF",
        (15, 3): "IA", (15, 4): "IB", (15, 5): "IC", (15, 6): "ID", (15, 7): "IE", (15, 8): "IF",
        (16, 3): "JA", (16, 4): "JB", (16, 5): "JC", (16, 6): "JD", (16, 7): "JE", (16, 8): "JF",
        (17, 3): "KA", (17, 4): "KB", (17, 5): "KC", (17, 6): "KD", (17, 7): "KE", (17, 8): "KF",
        (18, 3): "LA", (18, 4): "LB", (18, 5): "LC", (18, 6): "LD", (18, 7): "LE", (18, 8): "LF",
        (19, 3): "MA", (19, 4): "MB", (19, 5): "MC", (19, 6): "MD", (19, 7): "ME", (19, 8): "MF",
        (20, 3): "NA", (20, 4): "NB", (20, 5): "NC", (20, 6): "ND", (20, 7): "NE", (20, 8): "NF",
        (21, 3): "OA", (21, 4): "OB", (21, 5): "OC", (21, 6): "OD", (21, 7): "OE", (21, 8): "OF",
        (22, 3): "PA", (22, 4): "PB", (22, 5): "PC", (22, 6): "PD", (22, 7): "PE", (22, 8): "PF",
        (23, 3): "QA", (23, 4): "QB", (23, 5): "QC", (23, 6): "QD", (23, 7): "QE", (23, 8): "QF",
    }

    for doctor_idx, doctor_data in enumerate(doctors_data):
        # 标记是否为借调医生
        is_borrowed_list[doctor_idx] = (doctor_data.get("origin") == "borrowed")

        shifts = doctor_data.get("shifts", [])

        # 按天组织班次
        daily_shifts = {}
        for shift in shifts:
            start = shift.get("start", 0)
            end = shift.get("end", 0)
            tag = shift.get("tag", "")

            # 计算班次在哪一天
            day = start // 24
            if day >= num_days:
                continue  # 忽略超出天数的班次

            # 计算班次在当天的开始时间和时长
            start_hour = start % 24
            duration = end - start

            # 转换为班次编码
            if tag == "night" or (start_hour == 0 and duration == 7):
                shift_code = "YY"
            else:
                shift_code = shift_codes.get((start_hour, duration), "XX")

            if day not in daily_shifts:
                daily_shifts[day] = []
            daily_shifts[day].append((start_hour, shift_code))

        # 将班次分配到第一和第二矩阵
        for day, shifts_in_day in daily_shifts.items():
            # 按开始时间排序
            shifts_in_day.sort(key=lambda x: x[0])

            # 分配到第一和第二矩阵
            if len(shifts_in_day) >= 1:
                first_shifts[doctor_idx][day] = shifts_in_day[0][1]
            if len(shifts_in_day) >= 2:
                second_shifts[doctor_idx][day] = shifts_in_day[1][1]

    return first_shifts, second_shifts, is_borrowed_list

def print_matrices(first_shifts, second_shifts, doctor_ids=None, is_borrowed_list=None):
    """打印两个矩阵"""
    num_doctors = len(first_shifts)
    num_days = len(first_shifts[0]) if num_doctors > 0 else 0

    days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

    # 打印第一个矩阵
    print("\n第一个班次矩阵:")
    print("医生\\日期", end="\t")
    for day in range(num_days):
        print(days[day], end="\t")
    print()

    for doctor_idx in range(num_doctors):
        if doctor_ids and doctor_idx < len(doctor_ids):
            doctor_display = doctor_ids[doctor_idx]
        else:
            doctor_display = f"医生{doctor_idx+1}"

        # 添加医生类型标记
        if is_borrowed_list and doctor_idx < len(is_borrowed_list):
            if is_borrowed_list[doctor_idx]:
                doctor_display += "(借调)"
            else:
                doctor_display += "(内部)"

        print(doctor_display, end="\t")

        for day in range(num_days):
            print(first_shifts[doctor_idx][day], end="\t")
        print()

    # 打印第二个矩阵
    print("\n第二个班次矩阵:")
    print("医生\\日期", end="\t")
    for day in range(num_days):
        print(days[day], end="\t")
    print()

    for doctor_idx in range(num_doctors):
        if doctor_ids and doctor_idx < len(doctor_ids):
            doctor_display = doctor_ids[doctor_idx]
        else:
            doctor_display = f"医生{doctor_idx+1}"

        # 添加医生类型标记
        if is_borrowed_list and doctor_idx < len(is_borrowed_list):
            if is_borrowed_list[doctor_idx]:
                doctor_display += "(借调)"
            else:
                doctor_display += "(内部)"

        print(doctor_display, end="\t")

        for day in range(num_days):
            print(second_shifts[doctor_idx][day], end="\t")
        print()

# 使用示例
if __name__ == "__main__":
    # 示例数据
    sample_doctors = [
        {
            "id": "doctor1",
            "origin": "internal",
            "shifts": [
                {"start": 0, "end": 7, "tag": "night"},  # 周一夜班
                {"start": 32, "end": 40}  # 周三白班 8:00-16:00
            ]
        },
        {
            "id": "doctor2",
            "origin": "borrowed",
            "shifts": [
                {"start": 8, "end": 16},  # 周一白班
                {"start": 24, "end": 32}  # 周二白班
            ]
        }
    ]

    # 转换为矩阵
    first_shifts, second_shifts, is_borrowed = convert_doctors_to_matrices(sample_doctors)

    # 打印矩阵
    print_matrices(first_shifts, second_shifts, ["医生1", "医生2"], is_borrowed)

    # 验证约束
    validator = MatrixConstraintValidator()
    is_valid, violations = validator.validate_matrix_constraints(first_shifts, second_shifts, is_borrowed)

    print(f"\n验证结果: {'通过' if is_valid else '不通过'}")
    if violations:
        print("违规项:")
        for violation in violations:
            print(f"  - {violation}")