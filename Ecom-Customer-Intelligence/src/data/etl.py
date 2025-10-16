# src/data/etl.py
"""
ETL script (data layer).
- Uses src/data/ingest.py to load raw Olist CSVs.
- Cleans and aggregates to produce analysis-ready files:
    data/processed/customers_clean.csv
    data/processed/orders_rfm.csv
    data/processed/items_agg.csv
    data/processed/payments_agg.csv
- Optionally runs RFM step via --rfm (calls src.features.rfm)
"""
from pathlib import Path
import argparse
import pandas as pd
import subprocess
import sys

# import ingest (assumes src/data/ingest.py exists)
from . import ingest

# Resolve project root reliably
PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    df = customers.drop_duplicates(subset=['customer_id']).copy()
    df['customer_unique_id'] = df['customer_unique_id'].astype(str).str.strip()
    return df[['customer_id', 'customer_unique_id', 'customer_state']]


def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    date_cols = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    for c in date_cols:
        if c in orders.columns:
            orders[c] = pd.to_datetime(orders[c], errors='coerce')
    # remove canceled/unavailable orders
    if 'order_status' in orders.columns:
        orders = orders[~orders['order_status'].isin(['canceled', 'unavailable'])]
    return orders


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    payments_agg = payments.groupby('order_id', as_index=False)['payment_value'].sum()
    payments_agg['payment_value'] = pd.to_numeric(payments_agg['payment_value'], errors='coerce').fillna(0.0)
    return payments_agg


def aggregate_items(items: pd.DataFrame) -> pd.DataFrame:
    if {'price', 'freight_value'}.issubset(items.columns):
        items_agg = items.groupby('order_id', as_index=False).agg({
            'price': 'sum',
            'freight_value': 'sum'
        }).rename(columns={'price': 'total_price', 'freight_value': 'total_freight'})
    elif 'price' in items.columns:
        items_agg = items.groupby('order_id', as_index=False)['price'].sum().rename(columns={'price': 'total_price'})
        items_agg['total_freight'] = 0.0
    else:
        items_agg = pd.DataFrame(columns=['order_id', 'total_price', 'total_freight'])
    # Ensure numeric
    for col in ['total_price', 'total_freight']:
        if col in items_agg.columns:
            items_agg[col] = pd.to_numeric(items_agg[col], errors='coerce').fillna(0.0)
    return items_agg


def build_orders_rfm(save: bool = True) -> pd.DataFrame:
    """Build the orders_rfm dataframe and save processed files if save=True."""
    # load raw tables via ingest (ingest resolves paths internally)
    customers = ingest.load_customers()
    orders = ingest.load_orders()
    payments = ingest.load_payments()
    items = ingest.load_order_items()

    # Clean and save customers_clean
    customers_clean = clean_customers(customers)
    customers_clean.to_csv(PROCESSED_DIR / "customers_clean.csv", index=False)
    print(f"Saved customers_clean ({len(customers_clean)} rows)")

    # Clean orders
    orders = clean_orders(orders)

    # For RFM: keep only delivered orders
    if 'order_status' in orders.columns:
        orders_rfm = orders.loc[orders['order_status'] == 'delivered', ['order_id', 'customer_id', 'order_purchase_timestamp']].copy()
    else:
        orders_rfm = orders[['order_id', 'customer_id', 'order_purchase_timestamp']].copy()

    # Payments aggregation and save
    payments_agg = aggregate_payments(payments)
    payments_agg.to_csv(PROCESSED_DIR / "payments_agg.csv", index=False)
    print(f"Saved payments_agg ({len(payments_agg)} rows)")

    # Merge payments to orders_rfm and drop orders w/o payment
    orders_rfm = orders_rfm.merge(payments_agg, on='order_id', how='left')
    orders_rfm = orders_rfm.dropna(subset=['payment_value']).copy()

    # Merge customer fields
    orders_rfm = orders_rfm.merge(customers_clean, on='customer_id', how='left')

    # Items aggregation and save
    items_agg = aggregate_items(items)
    items_agg.to_csv(PROCESSED_DIR / "items_agg.csv", index=False)
    print(f"Saved items_agg ({len(items_agg)} rows)")

    # Merge items aggregates
    orders_rfm = orders_rfm.merge(items_agg, on='order_id', how='left')

    # Fill NaNs for numeric totals
    for col in ['payment_value', 'total_price', 'total_freight']:
        if col in orders_rfm.columns:
            orders_rfm[col] = pd.to_numeric(orders_rfm[col], errors='coerce').fillna(0.0)

    # Cast convenient dtypes
    if 'customer_state' in orders_rfm.columns:
        orders_rfm['customer_state'] = orders_rfm['customer_state'].astype('category')
    for c in ['order_id', 'customer_id', 'customer_unique_id']:
        if c in orders_rfm.columns:
            orders_rfm[c] = orders_rfm[c].astype('string')

    # Save orders_rfm
    if save:
        out = PROCESSED_DIR / "orders_rfm.csv"
        orders_rfm.to_csv(out, index=False)
        print(f" Saved orders_rfm to: {out.resolve()}  (rows: {len(orders_rfm)})")

    return orders_rfm


def run_rfm_subprocess(input_path: Path, output_path: Path) -> int:
    """Run the rfm module as subprocess with cwd=PROJECT_DIR and explicit paths."""
    cmd = [
        sys.executable, "-m", "src.features.rfm",
        "--input", str(input_path),
        "--output", str(output_path)
    ]
    print("Running RFM subprocess with cmd:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False, cwd=PROJECT_DIR, capture_output=True, text=True)
    print("RFM returncode:", proc.returncode)
    if proc.stdout:
        print("RFM stdout:\n", proc.stdout)
    if proc.stderr:
        print("RFM stderr:\n", proc.stderr)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Run ETL to build processed Olist tables.")
    parser.add_argument("--save", action="store_true", help="Save processed outputs")
    parser.add_argument("--rfm", action="store_true", help="Also run RFM pipeline after ETL")
    args = parser.parse_args()

    # Build orders_rfm (and other processed files)
    print("Running ETL pipeline...")
    orders_rfm_df = build_orders_rfm(save=args.save)

    orders_rfm_path = PROCESSED_DIR / "orders_rfm.csv"
    print("ETL: expected orders_rfm path:", orders_rfm_path.resolve())
    if not orders_rfm_path.exists():
        print("ERROR: orders_rfm.csv not found after ETL:", orders_rfm_path.resolve())
    else:
        print("ETL check: orders_rfm.csv exists.")

    # Optionally run rfm step (calls src.features.rfm)
    if args.rfm:
        print("Running RFM pipeline...")
        # pass explicit paths to avoid ambiguity
        rfm_out = PROCESSED_DIR / "rfm.csv"
        rc = run_rfm_subprocess(orders_rfm_path, rfm_out)
        if rc != 0:
            print(f"RFM subprocess exited with code {rc}. See logs above.")
        else:
            print(f"RFM subprocess completed successfully. rfm saved at: {rfm_out.resolve()}")


if __name__ == "__main__":
    main()
