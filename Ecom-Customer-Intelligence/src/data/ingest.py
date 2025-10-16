import pandas as pd
from pathlib import Path

# Project root
PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR / "data" / "raw"

def load_orders():
    return pd.read_csv(RAW_DIR / "olist_orders_dataset.csv")

def load_customers():
    return pd.read_csv(RAW_DIR / "olist_customers_dataset.csv")

def load_order_items():
    return pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")

def load_payments():
    return pd.read_csv(RAW_DIR / "olist_order_payments_dataset.csv")
