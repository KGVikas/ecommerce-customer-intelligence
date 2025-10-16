# src/features/rfm.py
"""
Compute RFM metrics + segmentation from orders_rfm.csv (robust path resolution).
This script resolves the project root from its own file location, so it doesn't
depend on current working directory. It also accepts optional CLI args to
override input/output paths.

Usage (from project root or anywhere):
  python -m src.features.rfm
  python -m src.features.rfm --input "data/processed/orders_rfm.csv" --output "data/processed/rfm.csv"
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

# Resolve project root relative to this file
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
DEFAULT_INPUT = DEFAULT_PROCESSED_DIR / "orders_rfm.csv"
DEFAULT_OUTPUT = DEFAULT_PROCESSED_DIR / "rfm.csv"


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'count',
        'payment_value': 'sum'
    }).reset_index()

    rfm.columns = ['customer_unique_id', 'Recency', 'Frequency', 'Monetary']
    rfm['Monetary'] = rfm['Monetary'].round(2)
    return rfm


def score_and_segment(rfm: pd.DataFrame) -> pd.DataFrame:
    quartiles = rfm[['Recency', 'Frequency', 'Monetary']].quantile([0.25, 0.5, 0.75]).to_dict()

    def r_score(x):
        if x <= quartiles['Recency'][0.25]:
            return 4
        elif x <= quartiles['Recency'][0.50]:
            return 3
        elif x <= quartiles['Recency'][0.75]:
            return 2
        else:
            return 1

    def fm_score(x, col):
        if x <= quartiles[col][0.25]:
            return 1
        elif x <= quartiles[col][0.50]:
            return 2
        elif x <= quartiles[col][0.75]:
            return 3
        else:
            return 4

    rfm['R_Score'] = rfm['Recency'].apply(r_score)
    rfm['F_Score'] = rfm['Frequency'].apply(lambda x: fm_score(x, 'Frequency'))
    rfm['M_Score'] = rfm['Monetary'].apply(lambda x: fm_score(x, 'Monetary'))
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    def segment_label(row):
        if row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
            return 'VIP High-Spender'
        elif row['F_Score'] >= 3 and row['M_Score'] <= 2:
            return 'Frequent Bargain Buyer'
        elif row['R_Score'] == 4 and row['F_Score'] <= 2:
            return 'New Potential'
        elif row['F_Score'] == 1 and row['M_Score'] >= 3:
            return 'One-Time Big Purchase'
        elif row['R_Score'] <= 2 and row['F_Score'] <= 2:
            return 'At Risk'
        else:
            return 'Opportunity Segment'

    rfm['Segment'] = rfm.apply(segment_label, axis=1)
    return rfm


def main():
    parser = argparse.ArgumentParser(description="Compute RFM from processed orders_rfm.csv")
    parser.add_argument("--input", help="Path to orders_rfm.csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", help="Path to write rfm.csv", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Helpful debug print so logs show exact locations
    print(f"PROJECT_DIR: {PROJECT_DIR}")
    print(f"Looking for orders_rfm at: {input_path.resolve()}")

    if not input_path.exists():
        raise FileNotFoundError(f"Path to {input_path} not found. Run ETL first. (expected: {input_path.resolve()})")

    # Read orders_rfm
    orders_rfm = pd.read_csv(input_path, parse_dates=['order_purchase_timestamp'])
    print(f"Loaded orders_rfm ({len(orders_rfm)} rows) from {input_path.resolve()}")

    rfm = compute_rfm(orders_rfm)
    rfm = score_and_segment(rfm)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(output_path, index=False)
    print(f"Saved RFM to: {output_path.resolve()} (customers: {len(rfm)})")


if __name__ == "__main__":
    main()
