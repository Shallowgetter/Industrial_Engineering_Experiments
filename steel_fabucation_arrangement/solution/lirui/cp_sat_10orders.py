import pandas as pd
import sys
import os

from ortools.sat.python import cp_model

def solve_exact(durations, transport, trans_time, time_limit_s=30):
    B = [0,1]
    S = [1,2,3]
    n = len({k[0] for k in durations.keys()})
    # horizon upper bound
    sum_all = sum(durations[(i,s,b)] for i in range(n) for s in S for b in B)
    sum_tr = sum(max(transport[(i,b)] for b in B) for i in range(n))
    horizon = sum_all + sum_tr + 1000

    model = cp_model.CpModel()

    # variables
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

    # each stage executed once
    for i in range(n):
        for s in S:
            model.Add(sum(x[(i,s,b)] for b in B) == 1)

    # precedence within job
    for i in range(n):
        # S1->S2
        for b1 in B:
            for b2 in B:
                model.Add(ed[(i,1,b1)] + trans_time[b1][b2] <= st[(i,2,b2)]).OnlyEnforceIf([x[(i,1,b1)], x[(i,2,b2)]])
        # S2->S3
        for b1 in B:
            for b2 in B:
                model.Add(ed[(i,2,b1)] + trans_time[b1][b2] <= st[(i,3,b2)]).OnlyEnforceIf([x[(i,2,b1)], x[(i,3,b2)]])
        # transport start >= cold end
        for b in B:
            model.Add(tr_st[(i,b)] >= ed[(i,3,b)]).OnlyEnforceIf(x[(i,3,b)])

    # machine no-overlap per base & stage
    for b in B:
        for s in S:
            model.AddNoOverlap([itv[(i,s,b)] for i in range(n)])
    # transport resource per base no-overlap (redundant but useful)
    for b in B:
        model.AddNoOverlap([tr_itv[(i,b)] for i in range(n)])

    # order variables: enforce cold-order and transport-order equality and immediate dispatch
    order = {}  # order[(b,i,j)] when i before j on base b
    for b in B:
        for i in range(n):
            for j in range(i+1, n):
                pres_pair = [x[(i,3,b)], x[(j,3,b)]]
                o_ij = model.NewBoolVar(f"ord_b{b}_i{i}_j{j}")
                o_ji = model.NewBoolVar(f"ord_b{b}_i{j}_j{i}")
                order[(b,i,j)] = o_ij
                order[(b,j,i)] = o_ji
                # if both present, exactly one ordering
                model.Add(o_ij + o_ji == 1).OnlyEnforceIf(pres_pair)
                # link to cold end times
                model.Add(ed[(i,3,b)] <= ed[(j,3,b)]).OnlyEnforceIf([o_ij] + pres_pair)
                model.Add(ed[(j,3,b)] <= ed[(i,3,b)]).OnlyEnforceIf([o_ji] + pres_pair)

                # now immediate transport relation: need bool g_ij representing ed_j >= tr_ed_i
                g_ij = model.NewBoolVar(f"g_b{b}_i{i}_j{j}")
                # encode ed_j >= tr_ed_i  as two reified constraints
                model.Add(ed[(j,3,b)] >= tr_ed[(i,b)]).OnlyEnforceIf(g_ij)
                model.Add(ed[(j,3,b)] <= tr_ed[(i,b)] - 1).OnlyEnforceIf(g_ij.Not())

                # if i before j and g_ij true => tr_start_j == ed_j
                model.Add(tr_st[(j,b)] == ed[(j,3,b)]).OnlyEnforceIf([o_ij, g_ij, x[(j,3,b)]])
                # if i before j and g_ij false => tr_start_j == tr_end_i
                model.Add(tr_st[(j,b)] == tr_ed[(i,b)]).OnlyEnforceIf([o_ij, g_ij.Not(), x[(j,3,b)], x[(i,3,b)]])
                # symmetric: need g_ji for opposite comparison
                g_ji = model.NewBoolVar(f"g_b{b}_i{j}_j{i}")
                model.Add(ed[(i,3,b)] >= tr_ed[(j,b)]).OnlyEnforceIf(g_ji)
                model.Add(ed[(i,3,b)] <= tr_ed[(j,b)] - 1).OnlyEnforceIf(g_ji.Not())
                model.Add(tr_st[(i,b)] == ed[(i,3,b)]).OnlyEnforceIf([o_ji, g_ji, x[(i,3,b)]])
                model.Add(tr_st[(i,b)] == tr_ed[(j,b)]).OnlyEnforceIf([o_ji, g_ji.Not(), x[(i,3,b)], x[(j,3,b)]])

    # makespan and objective
    makespan = model.NewIntVar(0, horizon, "makespan")
    for i in range(n):
        for b in B:
            model.Add(makespan >= tr_ed[(i,b)]).OnlyEnforceIf(x[(i,3,b)])
    model.Minimize(makespan)

    # solver
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
            # find assigned bases and intervals
            sol[i] = {}
            for s in S:
                for b in B:
                    if solver.Value(x[(i,s,b)]) == 1:
                        sol[i][f"s{s}_base"] = b
                        sol[i][f"s{s}_st"] = solver.Value(st[(i,s,b)])
                        sol[i][f"s{s}_ed"] = solver.Value(ed[(i,s,b)])
            # transport
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

def solve_heuristic(durations, transport, trans_time, time_limit_s=30):
    B = [0,1]
    S = [1,2,3]
    n = len({k[0] for k in durations.keys()})
    sum_all = sum(durations[(i,s,b)] for i in range(n) for s in S for b in B)
    horizon = sum_all + 10000

    model = cp_model.CpModel()
    x = {}; st = {}; ed = {}; itv = {}
    for i in range(n):
        for s in S:
            for b in B:
                x[(i,s,b)] = model.NewBoolVar(f"x_i{i}_s{s}_b{b}")
                st[(i,s,b)] = model.NewIntVar(0, horizon, f"st_i{i}_s{s}_b{b}")
                ed[(i,s,b)] = model.NewIntVar(0, horizon, f"ed_i{i}_s{s}_b{b}")
                dur = durations[(i,s,b)]
                itv[(i,s,b)] = model.NewOptionalIntervalVar(st[(i,s,b)], dur, ed[(i,s,b)], x[(i,s,b)],
                                                           f"itv_i{i}_s{s}_b{b}")
    # each stage once
    for i in range(n):
        for s in S:
            model.Add(sum(x[(i,s,b)] for b in B) == 1)
    # precedence within job
    for i in range(n):
        for b1 in B:
            for b2 in B:
                model.Add(ed[(i,1,b1)] + trans_time[b1][b2] <= st[(i,2,b2)]).OnlyEnforceIf([x[(i,1,b1)], x[(i,2,b2)]])
                model.Add(ed[(i,2,b1)] + trans_time[b1][b2] <= st[(i,3,b2)]).OnlyEnforceIf([x[(i,2,b1)], x[(i,3,b2)]])
    # machine no-overlap
    for b in B:
        for s in S:
            model.AddNoOverlap([itv[(i,s,b)] for i in range(n)])
    # objective: minimize upper bound of cold end + transport, approximate by minimizing max cold_end + max transport times (proxy)
    # We'll instead minimize max cold_end + sum transport upper bound, or simply minimize max cold_end as a proxy
    makespan_prod = model.NewIntVar(0, horizon, "makespan_prod")
    for i in range(n):
        for b in B:
            # if assigned to base b for cold
            model.Add(makespan_prod >= ed[(i,3,b)]).OnlyEnforceIf(x[(i,3,b)])
    model.Minimize(makespan_prod)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = max(1, os.cpu_count() - 1 or 1)
    print("Starting Heuristic production solve n=", n)
    solver.parameters.log_search_progress = True
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("Heuristic production solve failed:", solver.StatusName(status))
        return False, None, None

    # extract cold end times and assigned base
    cold_end = {}  # cold_end[i] = (base, end_time)
    prod_solution = {}
    for i in range(n):
        prod_solution[i] = {}
        for s in S:
            for b in B:
                if solver.Value(x[(i,s,b)]) == 1:
                    prod_solution[i][f"s{s}_base"] = b
                    prod_solution[i][f"s{s}_st"] = solver.Value(st[(i,s,b)])
                    prod_solution[i][f"s{s}_ed"] = solver.Value(ed[(i,s,b)])
        # cold
        b_cold = prod_solution[i]['s3_base']
        cold_end[i] = (b_cold, prod_solution[i]['s3_ed'])

    # Now simulate transport per base using immediate dispatch rule:
    transports = {}  # per i: tr_start,tr_end, tr_base
    for b in B:
        # collect jobs assigned to b
        jobs = [i for i in range(n) if cold_end[i][0] == b]
        # sort by cold_end time ascending (if equal, tie by job id)
        jobs_sorted = sorted(jobs, key=lambda j: (cold_end[j][1], j))
        prev_end = -1
        for idx, j in enumerate(jobs_sorted):
            ce = cold_end[j][1]
            if idx == 0:
                tr_st = ce
                tr_ed = tr_st + transport[(j,b)]
            else:
                tr_st = max(ce, prev_end)
                tr_ed = tr_st + transport[(j,b)]
            transports[j] = {'tr_base': b, 'tr_st': tr_st, 'tr_ed': tr_ed}
            prev_end = tr_ed

    # compute final makespan
    final_makespan = max(transports[j]['tr_ed'] for j in transports)
    print("Heuristic final makespan (after transport sim) =", final_makespan)
    # combine solution
    sol = {}
    for i in range(n):
        sol[i] = {}
        for s in S:
            sol[i][f"s{s}_base"] = prod_solution[i][f"s{s}_base"]
            sol[i][f"s{s}_st"] = prod_solution[i][f"s{s}_st"]
            sol[i][f"s{s}_ed"] = prod_solution[i][f"s{s}_ed"]
        sol[i]['tr_base'] = transports[i]['tr_base']
        sol[i]['tr_st'] = transports[i]['tr_st']
        sol[i]['tr_ed'] = transports[i]['tr_ed']
    return True, sol, final_makespan

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
    else:
        ok, sol, mk = solve_heuristic(durations, transport, trans_time, time_limit_s=time_limit_s)

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
    out_file = f"schedule_sol_sheet_{sheet_idx}_{mode_use}.xlsx"
    out_df.to_excel(out_file, index=False)
    print("已导出到", out_file)
    print("makespan =", mk)
    return out_df

if __name__ == "__main__":
    xlsx_path = "steel_fabucation_arrangement\data\original_data_v5.xlsx"
    xlsx_path100 = 'steel_fabucation_arrangement\data\original_data_v5.xlsx'

    df10 = main(xlsx_path, sheet_idx='数据1', mode='exact', time_limit_s=60)

    plot_gantt_from_solution("schedule_sol_sheet_数据1_exact.xlsx", sheet_name=0, savepath="gantt_10jobs.png")

    df100 = main(xlsx_path100, sheet_idx='数据2', mode='heuristic', time_limit_s=120)