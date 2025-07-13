import pandas as pd
import glob
import os

def analyze_pivots(reports_path='reports'):
    """
    Reads all pivot reports, computes per-symbol metrics, and outputs a summary CSV.
    """
    all_files = glob.glob(os.path.join(reports_path, "pivots_*.csv"))
    if not all_files:
        print("No pivot reports found.")
        return

    df_from_each_file = (pd.read_csv(f) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)

    if concatenated_df.empty:
        print("No pivot data found in reports.")
        return

    summary = concatenated_df.groupby('symbol').agg(
        average_confidence=('confidence', 'mean'),
        detection_frequency=('symbol', 'size')
    ).reset_index()

    summary_path = os.path.join(reports_path, "pivot_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Pivot analysis summary saved to {summary_path}")

if __name__ == "__main__":
    analyze_pivots()
