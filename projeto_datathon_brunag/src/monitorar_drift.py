"""
monitor_drift.py

Gera relatório de data drift usando Evidently (DataDriftTab).
"""

import pandas as pd
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

def main() -> None:
    """Calcula e salva o dashboard de data drift."""
    # 1) Dados de referência
    ref = pd.read_parquet("data/parquet/applicants/applicants.parquet")
    # 2) Logs de predições recentes
    logs = pd.read_json("logs/predictions.jsonl", lines=True)

    # 3) Gera dashboard
    dash = Dashboard(tabs=[DataDriftTab()])
    dash.calculate(ref, logs)
    dash.save("monitoring/drift_report.html")
    print("✅ Drift report salvo em monitoring/drift_report.html")

if __name__ == "__main__":
    main()
