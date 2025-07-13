# monitor_drift.py
import pandas as pd
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

# 1) carrega dados de referência (por ex. todo o histórico)
ref = pd.read_parquet("data/parquet/applicants/applicants.parquet")

# 2) carrega logs de entradas recentes (depois de algum período você coleta esses logs)
logs = pd.read_json("logs/predictions.jsonl", lines=True)

# 3) dashboard de drift
dash = Dashboard(tabs=[DataDriftTab()])
dash.calculate(ref, logs)
dash.save("monitoring/drift_report.html")
