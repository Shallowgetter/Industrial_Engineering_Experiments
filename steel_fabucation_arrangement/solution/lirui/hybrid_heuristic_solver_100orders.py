"""针对钢结构生产调度问题的混合启发式 + 局部搜索求解器。

本模块实现了一个面向大规模实例（例如包含 100 个订单的情况）的两阶段策略，
方法包括：

1. 通过贪心派工规则构造多个初始可行排程；
2. 使用模拟退火风格的局部搜索对最佳初始解进行改进，邻域操作包括交换、插入和 2-opt。

"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

try:
    # 复用现有 CP-SAT 实现中的数据读取辅助函数。
    from .cp_sat_10orders import read_excel_data
except ImportError:  # pragma: no cover - 脚本模式下的回退方案
    from cp_sat_10orders import read_excel_data


JobId = int
Stage = int
BaseId = int

BASES: Tuple[BaseId, BaseId] = (0, 1)
STAGES: Tuple[Stage, Stage, Stage] = (1, 2, 3)
DEFAULT_TRANS_TIME: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 450), (450, 0))


@dataclass(frozen=True)
class ScheduleResult:
    solution: Dict[JobId, Dict[str, int]]
    makespan: int
    job_sequence: List[JobId]


def _jobs_from_durations(durations: Dict[Tuple[JobId, Stage, BaseId], int]) -> List[JobId]:
    jobs = sorted({job for job, _, _ in durations.keys()})
    if not jobs:
        raise ValueError("durations 字典为空，无法构造调度方案")
    return jobs


def _precompute_job_features(
    durations: Dict[Tuple[JobId, Stage, BaseId], int],
    transport: Dict[Tuple[JobId, BaseId], int],
) -> Dict[JobId, Dict[str, float]]:
    """计算启发式排序规则所需的快速指标."""
    features: Dict[JobId, Dict[str, float]] = {}
    for job in _jobs_from_durations(durations):
        best_stage = {
            stage: min(durations[(job, stage, b)] for b in BASES)
            for stage in STAGES
        }
        total_best = sum(best_stage.values())
        stage3_gap = abs(durations[(job, 3, 0)] - durations[(job, 3, 1)])
        transport_best = min(transport[(job, b)] for b in BASES)
        features[job] = {
            "sum_best": total_best,
            "stage3_best": best_stage[3],
            "stage3_gap": stage3_gap,
            "transport_best": transport_best,
        }
    return features


def construct_schedule(
    job_sequence: Sequence[JobId],
    durations: Dict[Tuple[JobId, Stage, BaseId], int],
    transport: Dict[Tuple[JobId, BaseId], int],
    trans_time: Sequence[Sequence[int]] = DEFAULT_TRANS_TIME,
) -> ScheduleResult:
    """根据给定的作业访问顺序构造可行排程。

    对每个订单，在满足机器可用性、阶段间运输延迟以及出厂即时调度规则的前提下，
    贪心选择每个阶段的最佳基地。
    """
    machine_available = {(stage, base): 0 for stage in STAGES for base in BASES}
    transport_available = {base: 0 for base in BASES}

    solution: Dict[JobId, Dict[str, int]] = {}

    for job in job_sequence:
        job_plan: Dict[str, int] = {}
        prev_base: int | None = None
        prev_end = 0

        for stage in STAGES:
            best_choice: Tuple[int, int, int] | None = None  # （结束时间, 开始时间, 基地）
            for base in BASES:
                travel = 0 if prev_base is None else trans_time[prev_base][base]
                start = max(machine_available[(stage, base)], prev_end + travel)
                end = start + durations[(job, stage, base)]
                candidate = (end, start, base)
                if best_choice is None or candidate < best_choice:
                    best_choice = candidate

            assert best_choice is not None  # 供类型检查器使用
            end, start, base = best_choice
            machine_available[(stage, base)] = end
            job_plan[f"s{stage}_base"] = base
            job_plan[f"s{stage}_st"] = start
            job_plan[f"s{stage}_ed"] = end
            prev_base = base
            prev_end = end

        # 冷轧（阶段 3）结束后立即发运
        dispatch_base = job_plan["s3_base"]
        tr_start = max(prev_end, transport_available[dispatch_base])
        tr_end = tr_start + transport[(job, dispatch_base)]
        transport_available[dispatch_base] = tr_end

        job_plan["tr_base"] = dispatch_base
        job_plan["tr_st"] = tr_start
        job_plan["tr_ed"] = tr_end

        solution[job] = job_plan

    makespan = max(plan["tr_ed"] for plan in solution.values())
    return ScheduleResult(solution=solution, makespan=makespan, job_sequence=list(job_sequence))


def _generate_initial_sequences(
    durations: Dict[Tuple[JobId, Stage, BaseId], int],
    transport: Dict[Tuple[JobId, BaseId], int],
    random_starts: int,
    rng: random.Random,
) -> List[List[JobId]]:
    features = _precompute_job_features(durations, transport)
    jobs = _jobs_from_durations(durations)

    sequences: List[List[JobId]] = []

    # 按原始顺序安排（as-is）
    sequences.append(jobs.copy())

    # 按总加工时间从短到长排序
    sequences.append(sorted(jobs, key=lambda j: features[j]["sum_best"]))

    # 关注冷轧阶段（瓶颈）
    sequences.append(sorted(jobs, key=lambda j: features[j]["stage3_best"]))

    # 优先处理冷轧阶段差异大的订单以平滑运输
    sequences.append(sorted(jobs, key=lambda j: -features[j]["stage3_gap"]))

    # 优先安排运输较轻的订单
    sequences.append(sorted(jobs, key=lambda j: features[j]["transport_best"]))

    # 随机重启
    random_starts = max(0, random_starts)
    for _ in range(random_starts):
        seq = jobs.copy()
        rng.shuffle(seq)
        sequences.append(seq)

    # 在保留顺序的前提下去重
    unique_sequences: List[List[JobId]] = []
    seen = set()
    for seq in sequences:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            unique_sequences.append(seq)
    return unique_sequences


def _random_neighbor(sequence: Sequence[JobId], rng: random.Random) -> List[JobId]:
    if len(sequence) < 2:
        return list(sequence)

    move = rng.choice(["swap", "insert", "two_opt"])
    seq = list(sequence)
    i, j = rng.sample(range(len(seq)), 2)
    if i > j:
        i, j = j, i

    if move == "swap":
        seq[i], seq[j] = seq[j], seq[i]
    elif move == "insert":
        job = seq.pop(j)
        seq.insert(i, job)
    else:  # 2-opt 操作：反转区间
        seq[i:j + 1] = reversed(seq[i:j + 1])
    return seq


def _simulated_annealing(
    initial_sequence: Sequence[JobId],
    durations: Dict[Tuple[JobId, Stage, BaseId], int],
    transport: Dict[Tuple[JobId, BaseId], int],
    trans_time: Sequence[Sequence[int]],
    rng: random.Random,
    time_budget: float,
    max_iterations: int | None = None,
) -> ScheduleResult:
    start_time = time.time()
    cache: Dict[Tuple[JobId, ...], ScheduleResult] = {}

    def evaluate(sequence: Sequence[JobId]) -> ScheduleResult:
        key = tuple(sequence)
        if key not in cache:
            cache[key] = construct_schedule(sequence, durations, transport, trans_time)
        return cache[key]

    current = evaluate(initial_sequence)
    best = current

    if max_iterations is None:
        max_iterations = max(1000, len(initial_sequence) * 40)

    # 温度与当前完工期成比例
    temperature = max(1.0, current.makespan * 0.1)
    min_temperature = 1.0
    cooling = 0.995

    iterations = 0
    while iterations < max_iterations and (time.time() - start_time) < time_budget:
        neighbor_seq = _random_neighbor(current.job_sequence, rng)
        neighbor = evaluate(neighbor_seq)
        delta = neighbor.makespan - current.makespan

        accept = False
        if delta < 0:
            accept = True
        else:
            probability = math.exp(-delta / max(temperature, 1e-6))
            if rng.random() < probability:
                accept = True

        if accept:
            current = ScheduleResult(
                solution=neighbor.solution,
                makespan=neighbor.makespan,
                job_sequence=neighbor_seq,
            )
            if neighbor.makespan < best.makespan:
                best = ScheduleResult(
                    solution=neighbor.solution,
                    makespan=neighbor.makespan,
                    job_sequence=neighbor_seq,
                )

        temperature = max(min_temperature, temperature * cooling)
        iterations += 1

    return best


def solve_hybrid(
    durations: Dict[Tuple[JobId, Stage, BaseId], int],
    transport: Dict[Tuple[JobId, BaseId], int],
    trans_time: Sequence[Sequence[int]] = DEFAULT_TRANS_TIME,
    *,
    time_limit_s: float = 60.0,
    random_seed: int | None = None,
    random_restarts: int | None = None,
) -> ScheduleResult:
    """使用混合启发式与局部搜索求解该调度问题."""
    rng = random.Random(random_seed)
    jobs = _jobs_from_durations(durations)
    if random_restarts is None:
        random_restarts = max(5, len(jobs) // 10)

    initial_sequences = _generate_initial_sequences(durations, transport, random_restarts, rng)

    best_initial: ScheduleResult | None = None
    for seq in initial_sequences:
        candidate = construct_schedule(seq, durations, transport, trans_time)
        if best_initial is None or candidate.makespan < best_initial.makespan:
            best_initial = candidate

    assert best_initial is not None  # guaranteed with non-empty durations

    elapsed = 0.0
    start_time = time.time()
    remaining = time_limit_s - elapsed
    if remaining <= 0:
        return best_initial

    # 预留剩余时间的 20% 用于分散化重启
    sa_budget = max(0.0, remaining * 0.8)
    if sa_budget <= 0:
        return best_initial

    best_overall = best_initial

    # 主模拟退火过程
    sa_result = _simulated_annealing(
        best_initial.job_sequence,
        durations,
        transport,
        trans_time,
        rng,
        time_budget=sa_budget,
    )
    if sa_result.makespan < best_overall.makespan:
        best_overall = sa_result

    # 使用剩余时间进行多样化重启
    remaining_time = time_limit_s - (time.time() - start_time)
    restart_sequences = initial_sequences.copy()
    rng.shuffle(restart_sequences)
    for seq in restart_sequences:
        if remaining_time <= 0:
            break
        if list(seq) == best_overall.job_sequence:
            continue
        local_budget = remaining_time / 2  # 使用剩余时间的一半以保留后续空间
        local_result = _simulated_annealing(
            seq,
            durations,
            transport,
            trans_time,
            rng,
            time_budget=max(1.0, min(remaining_time, local_budget)),
        )
        if local_result.makespan < best_overall.makespan:
            best_overall = local_result
        remaining_time = time_limit_s - (time.time() - start_time)

    return best_overall


def solution_to_dataframe(solution: Dict[JobId, Dict[str, int]]) -> pd.DataFrame:
    rows = []
    for job, plan in sorted(solution.items()):
        rows.append({
            "job": job + 1,
            "s1_base": plan["s1_base"] + 1,
            "s1_st": plan["s1_st"],
            "s1_ed": plan["s1_ed"],
            "s2_base": plan["s2_base"] + 1,
            "s2_st": plan["s2_st"],
            "s2_ed": plan["s2_ed"],
            "s3_base": plan["s3_base"] + 1,
            "s3_st": plan["s3_st"],
            "s3_ed": plan["s3_ed"],
            "tr_base": plan["tr_base"] + 1,
            "tr_st": plan["tr_st"],
            "tr_ed": plan["tr_ed"],
        })
    return pd.DataFrame(rows)


def plot_gantt_from_solution(
    xlsx_path: str | Path,
    sheet_name: int | str = 0,
    col_map: Dict[str, str] | None = None,
    figsize: Tuple[int, int] = (14, 6),
    title: str | None = None,
    show_transport: bool = True,
    savepath: str | Path | None = None,
    font_size: int = 10,
):
    """根据排程结果绘制甘特图。"""

    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if col_map:
        df = df.rename(columns=col_map)

    expected = [
        "job",
        "s1_base",
        "s1_st",
        "s1_ed",
        "s2_base",
        "s2_st",
        "s2_ed",
        "s3_base",
        "s3_st",
        "s3_ed",
        "tr_base",
        "tr_st",
        "tr_ed",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"缺少预期列: {missing}. 请检查 Excel 列名或传入 col_map 映射。")

    def normalize_base_col(col: str) -> pd.Series:
        series = df[col].dropna()
        if not series.empty and series.isin([1, 2]).all():
            return df[col].astype(int) - 1
        return df[col].astype(int)

    df["s1_base"] = normalize_base_col("s1_base")
    df["s2_base"] = normalize_base_col("s2_base")
    df["s3_base"] = normalize_base_col("s3_base")
    df["tr_base"] = normalize_base_col("tr_base")

    bases = sorted({
        int(b)
        for b in df[["s1_base", "s2_base", "s3_base", "tr_base"]].values.flatten()
        if not pd.isna(b)
    })

    stage_names = {1: "炼铁", 2: "热轧", 3: "冷轧", "tr": "运输"}
    tracks: List[Tuple[int, int | str]] = []
    for b in bases:
        tracks.extend([(b, 1), (b, 2), (b, 3), (b, "tr")])

    track_labels = [f"基地{b + 1} {stage_names[s]}" for (b, s) in tracks]
    y_pos = list(range(len(tracks)))[::-1]
    track_index = {tracks[i]: y_pos[i] for i in range(len(tracks))}

    color_map = {1: "#4CAF50", 2: "#2196F3", 3: "#FFD54F", "tr": "#9CCC65"}

    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_title(title or "甘特图：各基地各阶段排程", fontsize=font_size + 2)
    ax.set_xlabel("时间", fontsize=font_size)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(track_labels, fontsize=font_size)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    time_min = df[[c for c in df.columns if c.endswith("_st")]].min().min()
    time_max = df[[c for c in df.columns if c.endswith("_ed")]].max().max()
    if pd.isna(time_min):
        time_min = 0
    if pd.isna(time_max):
        time_max = 0
    padding = max(1, int((time_max - time_min) * 0.02))
    ax.set_xlim(max(0, time_min - padding), time_max + padding)

    for _, row in df.iterrows():
        job_label = int(row["job"])

        b = int(row["s1_base"])
        st = float(row["s1_st"])
        ed = float(row["s1_ed"])
        y = track_index[(b, 1)]
        ax.barh(y, ed - st, left=st, height=0.6, color=color_map[1], edgecolor="k", alpha=0.9)
        ax.text(st + 0.03 * (time_max - time_min + 1), y, f"J{job_label}", va="center", ha="left", fontsize=max(6, font_size - 1), color="k")

        b = int(row["s2_base"])
        st = float(row["s2_st"])
        ed = float(row["s2_ed"])
        y = track_index[(b, 2)]
        ax.barh(y, ed - st, left=st, height=0.6, color=color_map[2], edgecolor="k", alpha=0.9)
        ax.text(st + 0.03 * (time_max - time_min + 1), y, f"J{job_label}", va="center", ha="left", fontsize=max(6, font_size - 1), color="k")

        b = int(row["s3_base"])
        st = float(row["s3_st"])
        ed = float(row["s3_ed"])
        y = track_index[(b, 3)]
        ax.barh(y, ed - st, left=st, height=0.6, color=color_map[3], edgecolor="k", alpha=0.9)
        ax.text(st + 0.03 * (time_max - time_min + 1), y, f"J{job_label}", va="center", ha="left", fontsize=max(6, font_size - 1), color="k")

        if show_transport:
            b = int(row["tr_base"])
            st = float(row["tr_st"])
            ed = float(row["tr_ed"])
            y = track_index[(b, "tr")]
            ax.barh(y, ed - st, left=st, height=0.6, color=color_map["tr"], edgecolor="k", alpha=0.8)
            ax.text(st + 0.03 * (time_max - time_min + 1), y, f"J{job_label}", va="center", ha="left", fontsize=max(6, font_size - 1), color="k")

    patches = [
        mpatches.Patch(facecolor=color_map[1], edgecolor="k", label="炼铁"),
        mpatches.Patch(facecolor=color_map[2], edgecolor="k", label="热轧"),
        mpatches.Patch(facecolor=color_map[3], edgecolor="k", label="冷轧"),
    ]
    if show_transport:
        patches.append(mpatches.Patch(facecolor=color_map["tr"], edgecolor="k", label="运输"))
    ax.legend(handles=patches, loc="upper right", fontsize=max(8, font_size - 2))

    plt.tight_layout()
    if savepath:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=200)
        print(f"甘特图已保存到 {savepath}")
    plt.show()
    plt.close()


def main(
    xlsx_path: str,
    sheet_idx: int | str = 0,
    output_path: str | None = None,
    time_limit_s: float = 60.0,
    random_seed: int | None = 42,
    *,
    plot: bool = True,
    plot_savepath: str | Path | None = None,
    plot_show_transport: bool = True,
    plot_title: str | None = None,
) -> pd.DataFrame:
    df, n, durations, transport = read_excel_data(xlsx_path, sheet_idx)
    print(f"读取数据完成，共 {n} 个订单。开始混合启发式求解……")

    result = solve_hybrid(
        durations,
        transport,
        DEFAULT_TRANS_TIME,
        time_limit_s=time_limit_s,
        random_seed=random_seed,
    )

    print(
        "混合启发式求解完成: makespan =",
        result.makespan,
        "(序列长度=",
        len(result.job_sequence),
        ")",
    )

    df_solution = solution_to_dataframe(result.solution)
    if output_path is None:
        sheet_str = str(sheet_idx).replace('/', '_')
        output_path = (
            Path("steel_fabucation_arrangement")
            / "solution"
            / "lirui"
            / f"schedule_sol_sheet_{sheet_str}_hybrid.xlsx"
        )
    else:
        output_path = Path(output_path)

    df_solution.to_excel(output_path, index=False)
    print("排产结果已导出到:", output_path)

    if plot:
        if plot_savepath is None:
            plot_savepath = output_path.with_suffix(".png")
        if plot_title is None:
            plot_title = f"混合启发式甘特图 (makespan={result.makespan})"
        plot_gantt_from_solution(
            output_path,
            sheet_name=0,
            title=plot_title,
            show_transport=plot_show_transport,
            savepath=plot_savepath,
        )

    return df_solution


if __name__ == "__main__":
    default_xlsx = "steel_fabucation_arrangement/data/original_data_v6.xlsx"
    main(default_xlsx, sheet_idx=0, time_limit_s=120.0, random_seed=42)
