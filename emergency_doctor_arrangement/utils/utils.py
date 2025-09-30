import pandas as pd
import re

def load_lambdas(xlsx_path: str):
    raw = pd.read_excel(xlsx_path, sheet_name=" 数据", header=None)
    m = re.search(r'μ[:：]\s*([0-9]+(\.[0-9]+)?)', str(raw.iloc[1,0]))
    mu = float(m.group(1)) if m else 6.0
    hours = list(raw.iloc[4, 1:1+24].astype(str).values)
    days = raw.iloc[5:12, 0].astype(str).tolist()
    lambda_table = raw.iloc[5:12, 1:1+24].astype(float).values
    df = pd.DataFrame(lambda_table, index=days, columns=range(24)).reset_index().rename(columns={"index":"day"})
    return df, mu