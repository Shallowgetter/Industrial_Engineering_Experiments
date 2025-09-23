import pandas as pd

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