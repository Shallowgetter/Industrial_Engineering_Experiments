import pandas as pd
import sys
import os

from ortools.sat.python import cp_model

def solve_exact(durations, transport, trans_time, time_limit_s=30):
    B = [0,1]
    S = [1,2,3]
    n = len({k[0] for k in durations.keys()})
    # 时间上界
    sum_all = sum(durations[(i,s,b)] for i in range(n) for s in S for b in B)
    sum_tr = sum(max(transport[(i,b)] for b in B) for i in range(n))
    horizon = sum_all + sum_tr + 1000

    model = cp_model.CpModel()

    # 变量：分配、时间、interval
    # x[(i,s,b)]：订单 i 的阶段 s 是否在基地 b 执行（presence / assignment）
    # st/ed：对应 interval 的开始/结束时间
    # itv：可选 interval（presence = x）
    # tr_*：运输的开始/结束/interval（presence = x[(i,3,b)]）
    x = {}
    st = {}; ed = {}; itv = {}
    tr_st = {}; tr_ed = {}; tr_itv = {}
    for i in range(n):
        for s in S:
            for b in B:
                x[(i,s,b)] = model.NewBoolVar(f"x_i{i}_s{s}_b{b}")
                st[(i,s,b)] = model.NewIntVar(0, horizon, f"st_i{i}_s{s}_b{b}")
                ed[(i,s,b)] = model.NewIntVar(0, horizon, f"ed_i{i}_s{s}_b{b}")
                dur = durations[(i,s,b)]
                itv[(i,s,b)] = model.NewOptionalIntervalVar(st[(i,s,b)], dur, ed[(i,s,b)], x[(i,s,b)],
                                                           f"itv_i{i}_s{s}_b{b}")
        for b in B:
            tr_st[(i,b)] = model.NewIntVar(0, horizon, f"trst_i{i}_b{b}")
            tr_ed[(i,b)] = model.NewIntVar(0, horizon, f"tred_i{i}_b{b}")
            tr_itv[(i,b)] = model.NewOptionalIntervalVar(tr_st[(i,b)], transport[(i,b)], tr_ed[(i,b)], x[(i,3,b)],
                                                         f"tr_itv_i{i}_b{b}")

    # ready[(i,s,b)] 记录订单 i 在基地 b 的阶段 s (s>1) 最早可开工时间，实现“立即生产”
    ready = {}
    for i in range(n):
        for b in B:
            for s in [2,3]:
                ready[(i,s,b)] = model.NewIntVar(0, horizon, f"ready_i{i}_s{s}_b{b}")
                # 未在该基地执行时将 ready 固定为 0，避免污染比较约束
                model.Add(ready[(i,s,b)] == 0).OnlyEnforceIf(x[(i,s,b)].Not())
                model.Add(ready[(i,s,b)] <= st[(i,s,b)]).OnlyEnforceIf(x[(i,s,b)])

    # 将 ready 与前序阶段结束+转运时间绑定
    for i in range(n):
        for b in B:
            for b_prev in B:
                model.Add(ready[(i,2,b)] == ed[(i,1,b_prev)] + trans_time[b_prev][b]).OnlyEnforceIf([x[(i,1,b_prev)], x[(i,2,b)]])
                model.Add(ready[(i,3,b)] == ed[(i,2,b_prev)] + trans_time[b_prev][b]).OnlyEnforceIf([x[(i,2,b_prev)], x[(i,3,b)]])

    # 约束：每个阶段恰好在一个基地执行
    for i in range(n):
        for s in S:
            model.Add(sum(x[(i,s,b)] for b in B) == 1)

    # 约束：工序内部前后依赖（含跨基地运输时间）
    # S1 -> S2, S2 -> S3；运输开始 >= 冷轧结束
    for i in range(n):
        # S1->S2
        for b1 in B:
            for b2 in B:
                model.Add(ed[(i,1,b1)] + trans_time[b1][b2] <= st[(i,2,b2)]).OnlyEnforceIf([x[(i,1,b1)], x[(i,2,b2)]])
        # S2->S3
        for b1 in B:
            for b2 in B:
                model.Add(ed[(i,2,b1)] + trans_time[b1][b2] <= st[(i,3,b2)]).OnlyEnforceIf([x[(i,2,b1)], x[(i,3,b2)]])
        # 运输不能早于冷轧结束
        for b in B:
            model.Add(tr_st[(i,b)] >= ed[(i,3,b)]).OnlyEnforceIf(x[(i,3,b)])

    # 约束：每个产线（基地+阶段）一次只能做一个任务
    # 使用 AddNoOverlap 保证同资源上的 interval 不重叠
    for b in B:
        for s in S:
            model.AddNoOverlap([itv[(i,s,b)] for i in range(n)])
    # 运输资源在每个基地也互斥
    for b in B:
        model.AddNoOverlap([tr_itv[(i,b)] for i in range(n)])

    # 阶段2“立即生产”：构造顺序与“紧邻后继”变量，确保资源无空闲
    order_s2 = {}
    succ_s2 = {}
    first_s2 = {}
    for b in B:
        for i in range(n):
            for j in range(i+1, n):
                pres_pair = [x[(i,2,b)], x[(j,2,b)]]
                o_ij = model.NewBoolVar(f"ord_s2_b{b}_i{i}_j{j}")
                o_ji = model.NewBoolVar(f"ord_s2_b{b}_i{j}_j{i}")
                order_s2[(b,i,j)] = o_ij
                order_s2[(b,j,i)] = o_ji
                model.Add(o_ij + o_ji == 1).OnlyEnforceIf(pres_pair)
                model.Add(st[(j,2,b)] >= ed[(i,2,b)]).OnlyEnforceIf([o_ij] + pres_pair)
                model.Add(st[(i,2,b)] >= ed[(j,2,b)]).OnlyEnforceIf([o_ji] + pres_pair)

                succ_ij = model.NewBoolVar(f"succ_s2_b{b}_i{i}_j{j}")
                succ_ji = model.NewBoolVar(f"succ_s2_b{b}_i{j}_j{i}")
                succ_s2[(b,i,j)] = succ_ij
                succ_s2[(b,j,i)] = succ_ji
                model.Add(succ_ij <= o_ij)
                model.Add(succ_ji <= o_ji)
                model.Add(succ_ij + succ_ji <= 1)
                for lit in pres_pair:
                    model.Add(succ_ij <= lit)
                    model.Add(succ_ji <= lit)

                # 立即生产：若 i 为 j 的紧邻前驱，则 st_j == max(ed_i, ready_j)
                ready_le_ij = model.NewBoolVar(f"ready_le_s2_b{b}_i{i}_j{j}")
                model.Add(ready[(j,2,b)] <= ed[(i,2,b)]).OnlyEnforceIf([ready_le_ij] + pres_pair)
                model.Add(ready[(j,2,b)] >= ed[(i,2,b)] + 1).OnlyEnforceIf([ready_le_ij.Not()] + pres_pair)
                model.Add(st[(j,2,b)] == ed[(i,2,b)]).OnlyEnforceIf([succ_ij, ready_le_ij])
                model.Add(st[(j,2,b)] == ready[(j,2,b)]).OnlyEnforceIf([succ_ij, ready_le_ij.Not()])

                ready_le_ji = model.NewBoolVar(f"ready_le_s2_b{b}_i{j}_j{i}")
                model.Add(ready[(i,2,b)] <= ed[(j,2,b)]).OnlyEnforceIf([ready_le_ji] + pres_pair)
                model.Add(ready[(i,2,b)] >= ed[(j,2,b)] + 1).OnlyEnforceIf([ready_le_ji.Not()] + pres_pair)
                model.Add(st[(i,2,b)] == ed[(j,2,b)]).OnlyEnforceIf([succ_ji, ready_le_ji])
                model.Add(st[(i,2,b)] == ready[(i,2,b)]).OnlyEnforceIf([succ_ji, ready_le_ji.Not()])

        for j in range(n):
            preds = [succ_s2[(b,i,j)] for i in range(n) if i != j and (b,i,j) in succ_s2]
            first = model.NewBoolVar(f"first_s2_b{b}_j{j}")
            first_s2[(b,j)] = first
            if preds:
                model.Add(sum(preds) + first == x[(j,2,b)])
            else:
                model.Add(first == x[(j,2,b)])
            model.Add(first <= x[(j,2,b)])
            model.Add(st[(j,2,b)] == ready[(j,2,b)]).OnlyEnforceIf([first, x[(j,2,b)]])

        for i in range(n):
            succs = [succ_s2[(b,i,j)] for j in range(n) if i != j and (b,i,j) in succ_s2]
            if succs:
                model.Add(sum(succs) <= x[(i,2,b)])

    # 顺序相关：冷轧顺序与运输顺序一致，并实现“立即生产 + 即时发车”逻辑
    # 为冷轧在每个基地的每对订单建立顺序/紧邻变量，并使用 g_ij 描述运输首发条件
    order = {}
    succ_cold = {}
    first_cold = {}
    for b in B:
        for i in range(n):
            for j in range(i+1, n):
                pres_pair = [x[(i,3,b)], x[(j,3,b)]]
                o_ij = model.NewBoolVar(f"ord_b{b}_i{i}_j{j}")
                o_ji = model.NewBoolVar(f"ord_b{b}_i{j}_j{i}")
                order[(b,i,j)] = o_ij
                order[(b,j,i)] = o_ji
                model.Add(o_ij + o_ji == 1).OnlyEnforceIf(pres_pair)
                model.Add(ed[(i,3,b)] <= ed[(j,3,b)]).OnlyEnforceIf([o_ij] + pres_pair)
                model.Add(ed[(j,3,b)] <= ed[(i,3,b)]).OnlyEnforceIf([o_ji] + pres_pair)

                succ_ij = model.NewBoolVar(f"succ_s3_b{b}_i{i}_j{j}")
                succ_ji = model.NewBoolVar(f"succ_s3_b{b}_i{j}_j{i}")
                succ_cold[(b,i,j)] = succ_ij
                succ_cold[(b,j,i)] = succ_ji
                model.Add(succ_ij <= o_ij)
                model.Add(succ_ji <= o_ji)
                model.Add(succ_ij + succ_ji <= 1)
                for lit in pres_pair:
                    model.Add(succ_ij <= lit)
                    model.Add(succ_ji <= lit)

                ready_le_ij_cold = model.NewBoolVar(f"ready_le_s3_b{b}_i{i}_j{j}")
                model.Add(ready[(j,3,b)] <= ed[(i,3,b)]).OnlyEnforceIf([ready_le_ij_cold] + pres_pair)
                model.Add(ready[(j,3,b)] >= ed[(i,3,b)] + 1).OnlyEnforceIf([ready_le_ij_cold.Not()] + pres_pair)
                model.Add(st[(j,3,b)] == ed[(i,3,b)]).OnlyEnforceIf([succ_ij, ready_le_ij_cold])
                model.Add(st[(j,3,b)] == ready[(j,3,b)]).OnlyEnforceIf([succ_ij, ready_le_ij_cold.Not()])

                ready_le_ji_cold = model.NewBoolVar(f"ready_le_s3_b{b}_i{j}_j{i}")
                model.Add(ready[(i,3,b)] <= ed[(j,3,b)]).OnlyEnforceIf([ready_le_ji_cold] + pres_pair)
                model.Add(ready[(i,3,b)] >= ed[(j,3,b)] + 1).OnlyEnforceIf([ready_le_ji_cold.Not()] + pres_pair)
                model.Add(st[(i,3,b)] == ed[(j,3,b)]).OnlyEnforceIf([succ_ji, ready_le_ji_cold])
                model.Add(st[(i,3,b)] == ready[(i,3,b)]).OnlyEnforceIf([succ_ji, ready_le_ji_cold.Not()])

                # g_ij 用于判断 ed_j >= tr_ed_i，从而分支 tr_start_j 的取值
                g_ij = model.NewBoolVar(f"g_b{b}_i{i}_j{j}")
                model.Add(ed[(j,3,b)] >= tr_ed[(i,b)]).OnlyEnforceIf(g_ij)
                model.Add(ed[(j,3,b)] <= tr_ed[(i,b)] - 1).OnlyEnforceIf(g_ij.Not())

                # 若 i 在 j 之前且 ed_j >= tr_ed_i => tr_start_j == ed_j
                model.Add(tr_st[(j,b)] == ed[(j,3,b)]).OnlyEnforceIf([o_ij, g_ij, x[(j,3,b)]])
                # 若 i 在 j 之前且 ed_j < tr_ed_i  => tr_start_j == tr_end_i（须等前车运输结束）
                model.Add(tr_st[(j,b)] == tr_ed[(i,b)]).OnlyEnforceIf([o_ij, g_ij.Not(), x[(j,3,b)], x[(i,3,b)]])

                # 对称地为 j 在 i 之前的情形建立相同逻辑
                g_ji = model.NewBoolVar(f"g_b{b}_i{j}_j{i}")
                model.Add(ed[(i,3,b)] >= tr_ed[(j,b)]).OnlyEnforceIf(g_ji)
                model.Add(ed[(i,3,b)] <= tr_ed[(j,b)] - 1).OnlyEnforceIf(g_ji.Not())
                model.Add(tr_st[(i,b)] == ed[(i,3,b)]).OnlyEnforceIf([o_ji, g_ji, x[(i,3,b)]])
                model.Add(tr_st[(i,b)] == tr_ed[(j,b)]).OnlyEnforceIf([o_ji, g_ji.Not(), x[(i,3,b)], x[(j,3,b)]])

        for j in range(n):
            preds = [succ_cold[(b,i,j)] for i in range(n) if i != j and (b,i,j) in succ_cold]
            first = model.NewBoolVar(f"first_s3_b{b}_j{j}")
            first_cold[(b,j)] = first
            if preds:
                model.Add(sum(preds) + first == x[(j,3,b)])
            else:
                model.Add(first == x[(j,3,b)])
            model.Add(first <= x[(j,3,b)])
            model.Add(st[(j,3,b)] == ready[(j,3,b)]).OnlyEnforceIf([first, x[(j,3,b)]])

        for i in range(n):
            succs = [succ_cold[(b,i,j)] for j in range(n) if i != j and (b,i,j) in succ_cold]
            if succs:
                model.Add(sum(succs) <= x[(i,3,b)])

    # 强制 S1 与 S2 在同一基地的相对顺序一致
    # 对每对订单在同一基地同时做 S1 和 S2 时，创建 order 布尔并绑定 S1 与 S2 的先后
    for b in B:
        for i in range(n):
            for j in range(i+1, n):
                # 仅在两订单在该基地同时做 S1 和 S2 时生效
                pres_s1s2 = [x[(i,1,b)], x[(j,1,b)], x[(i,2,b)], x[(j,2,b)]]
                o12_ij = model.NewBoolVar(f"ord12_b{b}_i{i}_j{j}")
                o12_ji = model.NewBoolVar(f"ord12_b{b}_i{j}_j{i}")
                model.Add(o12_ij + o12_ji == 1).OnlyEnforceIf(pres_s1s2)
                # 若 o12_ij 为真，则 i 在 S1 和 S2 上均先于 j
                model.Add(ed[(i,1,b)] <= st[(j,1,b)]).OnlyEnforceIf([o12_ij] + pres_s1s2)
                model.Add(ed[(i,2,b)] <= st[(j,2,b)]).OnlyEnforceIf([o12_ij] + pres_s1s2)
                # 对称
                model.Add(ed[(j,1,b)] <= st[(i,1,b)]).OnlyEnforceIf([o12_ji] + pres_s1s2)
                model.Add(ed[(j,2,b)] <= st[(i,2,b)]).OnlyEnforceIf([o12_ji] + pres_s1s2)

    # 强制“首个下线的订单必须立刻发车”——复用 first_cold 指示器
    for b in B:
        for j in range(n):
            first = first_cold[(b,j)]
            model.Add(tr_st[(j,b)] == ed[(j,3,b)]).OnlyEnforceIf([first, x[(j,3,b)]])

    # 目标：最小化所有订单运输完成时间的最大值（makespan）
    makespan = model.NewIntVar(0, horizon, "makespan")
    for i in range(n):
        for b in B:
            model.Add(makespan >= tr_ed[(i,b)]).OnlyEnforceIf(x[(i,3,b)])
    model.Minimize(makespan)

    # 求解参数与求解
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = max(1, os.cpu_count() - 1 or 1)
    print("Starting Exact solve (may be heavy for n>20)... n =", n)
    solver.parameters.log_search_progress = True
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("Exact solve status:", solver.StatusName(status), "makespan=", solver.Value(makespan))
        sol = {}
        for i in range(n):
            sol[i] = {}
            for s in S:
                for b in B:
                    if solver.Value(x[(i,s,b)]) == 1:
                        sol[i][f"s{s}_base"] = b
                        sol[i][f"s{s}_st"] = solver.Value(st[(i,s,b)])
                        sol[i][f"s{s}_ed"] = solver.Value(ed[(i,s,b)])
            for b in B:
                if solver.Value(x[(i,3,b)]) == 1:
                    sol[i]['tr_base'] = b
                    sol[i]['tr_st'] = solver.Value(tr_st[(i,b)])
                    sol[i]['tr_ed'] = solver.Value(tr_ed[(i,b)])
        return True, sol, solver.Value(makespan)
    else:
        print("Exact solver status:", solver.StatusName(status))
        return False, None, None

    
def read_excel_data(path, sheet_name=0):
    df = pd.read_excel(path, sheet_name=sheet_name)
    print(df.head)
    df = df.reset_index(drop=True)
    n = len(df)
    durations = {}
    transport = {}
    for i in range(n):
        durations[(i,1,0)] = int(df.iloc[i]['基地1炼铁时间'])
        durations[(i,1,1)] = int(df.iloc[i]['基地2炼铁时间'])
        durations[(i,2,0)] = int(df.iloc[i]['基地1热轧时间'])
        durations[(i,2,1)] = int(df.iloc[i]['基地2热轧时间'])
        durations[(i,3,0)] = int(df.iloc[i]['基地1冷轧时间'])
        durations[(i,3,1)] = int(df.iloc[i]['基地2冷轧时间'])
        transport[(i,0)] = int(df.iloc[i]['基地1运输时间'])
        transport[(i,1)] = int(df.iloc[i]['基地2运输时间'])
    return df, n, durations, transport

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_gantt_from_solution(xlsx_path, sheet_name=0,
                             col_map=None,
                             figsize=(14, 6),
                             title=None,
                             show_transport=True,
                             savepath=None,
                             font_size=10):
    """
    Read schedule solution from an Excel and draw a Gantt chart.
    Expected columns (default):
      'job', 
      's1_base','s1_st','s1_ed',
      's2_base','s2_st','s2_ed',
      's3_base','s3_st','s3_ed',
      'tr_base','tr_st','tr_ed'
    - xlsx_path: file path to excel
    - sheet_name: sheet index or name
    - col_map: optional dict to map your column names to expected ones,
               e.g. {'订单号':'job','基地1炼铁时间':...}  (only map the output columns)
    - figsize: figure size
    - show_transport: whether to draw transport intervals
    - savepath: if provided, save the figure to this path (png/pdf)
    """
    # read
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    # normalize column names if mapping provided
    if col_map:
        df = df.rename(columns=col_map)

    # verify required columns
    expected = ['job',
                's1_base','s1_st','s1_ed',
                's2_base','s2_st','s2_ed',
                's3_base','s3_st','s3_ed',
                'tr_base','tr_st','tr_ed']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"缺少预期列: {missing}. 请检查 Excel 列名或传入 col_map 映射。")

    # convert bases to int (0 or 1) if they are 1-based, try to detect 1-based
    # we support either (0,1) or (1,2). If values are 1 or 2, convert to 0/1.
    def normalize_base_col(col):
        if df[col].dropna().isin([1,2]).all():
            return df[col].astype(int) - 1
        else:
            return df[col].astype(int)

    df['s1_base'] = normalize_base_col('s1_base')
    df['s2_base'] = normalize_base_col('s2_base')
    df['s3_base'] = normalize_base_col('s3_base')
    df['tr_base'] = normalize_base_col('tr_base')

    # resources per base
    bases = sorted(list({int(b) for b in df[['s1_base','s2_base','s3_base','tr_base']].values.flatten() if not pd.isna(b)}))
    # define track order: for each base, tracks = ['炼铁','热轧','冷轧','运输']
    stage_names = {1: '炼铁', 2: '热轧', 3: '冷轧', 'tr': '运输'}
    tracks = []
    for b in bases:
        tracks.extend([(b,1), (b,2), (b,3), (b,'tr')])

    # create y positions
    track_labels = [f"基地{b+1} {stage_names[s]}" for (b,s) in tracks]
    y_pos = list(range(len(tracks)))[::-1]  # reverse so base1 top, base2 below (optional)

    track_index = {tracks[i]: y_pos[i] for i in range(len(tracks))}

    # colors for stages
    color_map = {1: '#4CAF50', 2: '#2196F3', 3: '#FFD54F', 'tr': '#9CCC65'}  # green, blue, yellow, light green

    # create figure
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_title(title or "甘特图：各基地各阶段排程", fontsize=font_size+2)
    ax.set_xlabel("时间", fontsize=font_size)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(track_labels, fontsize=font_size)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    # find global time span
    time_min = df[[c for c in df.columns if c.endswith('_st')]].min().min()
    time_max = df[[c for c in df.columns if c.endswith('_ed')]].max().max()
    if pd.isna(time_min): time_min = 0
    if pd.isna(time_max): time_max = 0
    padding = max(1, int((time_max - time_min) * 0.02))
    ax.set_xlim(max(0, time_min - padding), time_max + padding)

    # draw bars for each job & each stage
    for _, row in df.iterrows():
        job_label = int(row['job'])
        # stage 1
        b = int(row['s1_base'])
        st = float(row['s1_st']); ed = float(row['s1_ed'])
        y = track_index[(b,1)]
        ax.barh(y, ed - st, left=st, height=0.6, color=color_map[1], edgecolor='k', alpha=0.9)
        ax.text(st + 0.03*(time_max-time_min+1), y, f"J{job_label}", va='center', ha='left', fontsize=max(6,font_size-1), color='k')

        # stage 2
        b = int(row['s2_base'])
        st = float(row['s2_st']); ed = float(row['s2_ed'])
        y = track_index[(b,2)]
        ax.barh(y, ed - st, left=st, height=0.6, color=color_map[2], edgecolor='k', alpha=0.9)
        ax.text(st + 0.03*(time_max-time_min+1), y, f"J{job_label}", va='center', ha='left', fontsize=max(6,font_size-1), color='k')

        # stage 3
        b = int(row['s3_base'])
        st = float(row['s3_st']); ed = float(row['s3_ed'])
        y = track_index[(b,3)]
        ax.barh(y, ed - st, left=st, height=0.6, color=color_map[3], edgecolor='k', alpha=0.9)
        ax.text(st + 0.03*(time_max-time_min+1), y, f"J{job_label}", va='center', ha='left', fontsize=max(6,font_size-1), color='k')

        # transport
        if show_transport:
            b = int(row['tr_base'])
            st = float(row['tr_st']); ed = float(row['tr_ed'])
            y = track_index[(b,'tr')]
            ax.barh(y, ed - st, left=st, height=0.6, color=color_map['tr'], edgecolor='k', alpha=0.8)
            ax.text(st + 0.03*(time_max-time_min+1), y, f"J{job_label}", va='center', ha='left', fontsize=max(6,font_size-1), color='k')

    # legend
    patches = [mpatches.Patch(facecolor=color_map[1], edgecolor='k', label='炼铁'),
               mpatches.Patch(facecolor=color_map[2], edgecolor='k', label='热轧'),
               mpatches.Patch(facecolor=color_map[3], edgecolor='k', label='冷轧')]
    if show_transport:
        patches.append(mpatches.Patch(facecolor=color_map['tr'], edgecolor='k', label='运输'))
    ax.legend(handles=patches, loc='upper right', fontsize=max(8,font_size-2))

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
        print(f"甘特图已保存到 {savepath}")
    plt.show()


def main(xlsx_path, sheet_idx=0, mode='auto', time_limit_s=30):
    # mode in {'auto','exact','heuristic'}
    df, n, durations, transport = read_excel_data(xlsx_path, sheet_idx)
    # trans time between bases: 450 if different, 0 if same
    trans_time = [[0,450],[450,0]]

    if mode == 'auto':
        if n <= 20:
            mode_use = 'exact'
        else:
            mode_use = 'heuristic'
    else:
        mode_use = mode

    if mode_use == 'exact':
        ok, sol, mk = solve_exact(durations, transport, trans_time, time_limit_s=time_limit_s)

    if not ok:
        print("未获得可行解。")
        return

    rows = []
    for i in range(n):
        r = {
            'job': i+1,
            's1_base': sol[i]['s1_base']+1,
            's1_st': sol[i]['s1_st'],
            's1_ed': sol[i]['s1_ed'],
            's2_base': sol[i]['s2_base']+1,
            's2_st': sol[i]['s2_st'],
            's2_ed': sol[i]['s2_ed'],
            's3_base': sol[i]['s3_base']+1,
            's3_st': sol[i]['s3_st'],
            's3_ed': sol[i]['s3_ed'],
            'tr_base': sol[i]['tr_base']+1,
            'tr_st': sol[i]['tr_st'],
            'tr_ed': sol[i]['tr_ed']
        }
        rows.append(r)
    out_df = pd.DataFrame(rows)
    out_file = f"steel_fabucation_arrangement\solution\lirui\schedule_sol_sheet_{sheet_idx}_{mode_use}_instant.xlsx"
    out_df.to_excel(out_file, index=False)
    print("已导出到", out_file)
    print("makespan =", mk)
    return out_df

if __name__ == "__main__":
    xlsx_path = "steel_fabucation_arrangement\data\original_data_v5.xlsx"
    xlsx_path100 = 'steel_fabucation_arrangement\data\original_data_v6.xlsx'

    df10 = main(xlsx_path, sheet_idx='数据1', mode='exact', time_limit_s=60)

    plot_gantt_from_solution("steel_fabucation_arrangement\solution\lirui\schedule_sol_sheet_数据1_exact_instant.xlsx", sheet_name=0, savepath="steel_fabucation_arrangement\solution\lirui\gantt_10jobs_instant.png")