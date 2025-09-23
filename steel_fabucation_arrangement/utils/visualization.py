import pandas as pd

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
