from __future__ import annotations

import subprocess
import sys

def run(cmd: list[str]) -> None:
    print(f"\n▶ Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

def main() -> None:
    run([sys.executable, "src/score.py"])
    run([sys.executable, "src/monitor_summary.py"])
    run([sys.executable, "src/model_metrics_export.py"])

    print("\nTableau exports complete. You can now connect Tableau to:")
    print("- outputs/predictions/customer_scored.csv")
    print("- outputs/metrics/feature_importance.csv")
    print("- outputs/metrics/monitor_summary.csv")
    print("- outputs/metrics/model_metrics.csv")

if __name__ == "__main__":
    main()
