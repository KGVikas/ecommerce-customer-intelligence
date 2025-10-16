

from pathlib import Path
import subprocess
import sys
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
import subprocess, sys

PROJECT_DIR = Path(__file__).resolve().parents[2]


# ---------- Paths ----------
PROJECT_DIR = Path(__file__).resolve().parents[2]  # project root
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RFM_PATH = PROCESSED_DIR / "rfm.csv"
ORDERS_RFM_PATH = PROCESSED_DIR / "orders_rfm.csv"

# ---------- Helpers (cached) ----------
@st.cache_data(ttl=600)
def load_rfm(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=False)

@st.cache_data(ttl=600)
def load_orders_rfm(path: Path) -> pd.DataFrame:
    # parse date column
    df = pd.read_csv(path, parse_dates=["order_purchase_timestamp"])
    return df

def run_pipeline(save: bool = True, rfm: bool = True) -> tuple[int, str]:
    """
    Run the ETL pipeline to rebuild processed files.
    Returns (returncode, combined_stdout_stderr).
    """
    # Build command: call etl module which can produce both orders_rfm and rfm
    cmd = [sys.executable, "-m", "src.data.etl"]
    if save:
        cmd.append("--save")
    if rfm:
        cmd.append("--rfm")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=PROJECT_DIR)
        output = proc.stdout + "\n" + proc.stderr
        return proc.returncode, output
    except Exception as e:
        return 1, f"Failed to run pipeline: {e}"

def safe_read(path: Path, loader_fn):
    try:
        return loader_fn(path)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error reading {path.name}: {e}")
        return None

# ---------- Streamlit ----------
st.set_page_config(page_title="Olist Customer Intelligence", layout="wide")
st.title("Olist Customer Intelligence — RFM Dashboard")

st.markdown(
    """
    This dashboard reads processed datasets produced by the ETL pipeline:
    `data/processed/orders_rfm.csv` and `data/processed/rfm.csv`.
    You can re-run the ETL+RFM pipeline from here if files are missing or you want fresh data.
    """
)

# Check files
rfm_exists = RFM_PATH.exists()
orders_exists = ORDERS_RFM_PATH.exists()

col1, col2 = st.columns([3, 1])
with col2:
    st.write("**Pipeline tools**")
    if st.button("Run ETL & RFM pipeline"):
        with st.spinner("Running ETL & RFM (this may take a minute)..."):
            code, out = run_pipeline(save=True, rfm=True)
        if code == 0:
            st.success("Pipeline finished successfully.")
        else:
            st.error("Pipeline returned non-zero exit code. See logs below.")
        st.code(out[:4000])  # show first chunk

    # Quick status
    st.write("**Processed files**")
    st.write(f"- rfm.csv: {'✅' if rfm_exists else '❌'}")
    st.write(f"- orders_rfm.csv: {'✅' if orders_exists else '❌'}")

# Load data (try)
rfm_df = safe_read(RFM_PATH, load_rfm) if rfm_exists else None
orders_df = safe_read(ORDERS_RFM_PATH, load_orders_rfm) if orders_exists else None

if rfm_df is None or orders_df is None:
    st.warning("Processed files missing. Run the ETL pipeline (right) or place processed CSVs in data/processed/.")
    # Download buttons 
    if orders_exists:
        st.success("Found orders_rfm.csv - you can continue with detailed analysis.")
    if rfm_exists:
        st.success("Found rfm.csv - you can continue with segment-level analysis.")
    st.stop()

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
# segments
segments = sorted(rfm_df["Segment"].unique().tolist())
selected_segments = st.sidebar.multiselect("Segments", options=segments, default=segments)

# states from orders (join)
orders_states = sorted(orders_df["customer_state"].astype(str).unique().tolist())
selected_states = st.sidebar.multiselect("Customer state", options=orders_states, default=[])

# date range
min_date = orders_df["order_purchase_timestamp"].min().date()
max_date = orders_df["order_purchase_timestamp"].max().date()
start_date, end_date = st.sidebar.date_input("Purchase date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Apply filters to orders and rfm
# Filter RFM by segments first
rfm_filtered = rfm_df[rfm_df["Segment"].isin(selected_segments)].copy()

# If state filter or date filter is set, derive customer IDs to keep
if selected_states or (start_date and end_date):
    mask = pd.Series([True] * len(orders_df))
    if selected_states:
        mask &= orders_df["customer_state"].astype(str).isin(selected_states)
    if start_date and end_date:
        mask &= orders_df["order_purchase_timestamp"].dt.date.between(start_date, end_date)
    cust_ids = orders_df.loc[mask, "customer_unique_id"].unique()
    rfm_filtered = rfm_filtered[rfm_filtered["customer_unique_id"].isin(cust_ids)]

# ---------- KPI row ----------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
total_customers = len(rfm_filtered)
total_revenue = rfm_filtered["Monetary"].sum()
avg_freq = rfm_filtered["Frequency"].mean() if total_customers else 0
avg_recency = rfm_filtered["Recency"].mean() if total_customers else 0

kpi1.metric("Customers (filtered)", f"{total_customers:,}")
kpi2.metric("Total Monetary ($)", f"{total_revenue:,.2f}")
kpi3.metric("Avg Frequency", f"{avg_freq:.2f}")
kpi4.metric("Avg Recency (days)", f"{avg_recency:.1f}")

# ---------- Charts ----------
# 1) Segment counts & revenue
seg_agg = rfm_filtered.groupby("Segment").agg({
    "customer_unique_id": "count",
    "Monetary": "sum"
}).rename(columns={"customer_unique_id": "CustomerCount", "Monetary": "Revenue"}).reset_index()
seg_agg = seg_agg.sort_values("Revenue", ascending=False)

col_a, col_b = st.columns([1, 1])
with col_a:
    st.subheader("Customers by Segment")
    fig_seg_count = px.bar(seg_agg, x="Segment", y="CustomerCount", hover_data=["Revenue"], labels={"CustomerCount": "Customers"}, orientation="v")
    st.plotly_chart(fig_seg_count, use_container_width=True)
with col_b:
    st.subheader("Revenue by Segment")
    fig_seg_rev = px.pie(seg_agg, names="Segment", values="Revenue", hole=0.4)
    st.plotly_chart(fig_seg_rev, use_container_width=True)

# 2) Recency vs Frequency scatter (size by Monetary)
st.subheader("Recency vs Frequency (size = Monetary)")
# Join Monetary into orders-level view if needed; here use rfm_filtered
fig = px.scatter(rfm_filtered, x="Recency", y="Frequency", size="Monetary", color="Segment",
                 hover_data=["customer_unique_id", "Monetary"], labels={"Recency":"Days since last purchase", "Frequency":"Order count"})
st.plotly_chart(fig, use_container_width=True)

# 3) Distributions
dist_col1, dist_col2, dist_col3 = st.columns(3)
with dist_col1:
    st.subheader("Recency distribution")
    fig_r = px.histogram(rfm_filtered, x="Recency", nbins=30)
    st.plotly_chart(fig_r, use_container_width=True)
with dist_col2:
    st.subheader("Frequency distribution")
    fig_f = px.histogram(rfm_filtered, x="Frequency", nbins=30)
    st.plotly_chart(fig_f, use_container_width=True)
with dist_col3:
    st.subheader("Monetary distribution")
    fig_m = px.histogram(rfm_filtered, x="Monetary", nbins=30)
    st.plotly_chart(fig_m, use_container_width=True)

# 4) Top customers table (by Monetary)
st.subheader("Top customers (by Monetary)")
top_n = st.slider("Top N customers", min_value=5, max_value=100, value=25)
top_customers = rfm_filtered.sort_values("Monetary", ascending=False).head(top_n)
st.dataframe(top_customers[["customer_unique_id", "Segment", "Recency", "Frequency", "Monetary"]].reset_index(drop=True))

# 5) Drilldown: Orders by state/time (using orders_df)
st.subheader("Orders trend (filtered)")
mask = pd.Series([True] * len(orders_df))
if selected_states:
    mask &= orders_df["customer_state"].astype(str).isin(selected_states)
mask &= orders_df["order_purchase_timestamp"].dt.date.between(start_date, end_date)
orders_time = orders_df.loc[mask].copy()
if orders_time.empty:
    st.info("No orders in selected filters/date range.")
else:
    orders_time["order_date"] = orders_time["order_purchase_timestamp"].dt.date
    orders_daily = orders_time.groupby("order_date").agg({"payment_value":"sum", "order_id":"count"}).rename(columns={"payment_value":"Revenue","order_id":"Orders"}).reset_index()
    fig_time = px.line(orders_daily, x="order_date", y=["Orders","Revenue"])
    st.plotly_chart(fig_time, use_container_width=True)

# ---------- Data export ----------
st.markdown("---")
st.subheader("Export filtered data")
@st.cache_data()
def get_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

if not rfm_filtered.empty:
    st.download_button("Download filtered RFM (CSV)", data=get_csv_bytes(rfm_filtered), file_name="rfm_filtered.csv", mime="text/csv")

if not orders_time.empty:
    st.download_button("Download filtered Orders (CSV)", data=get_csv_bytes(orders_time), file_name="orders_filtered.csv", mime="text/csv")

st.markdown("**Notes:** The dashboard reads processed CSVs from `data/processed/`. Use the 'Run ETL & RFM pipeline' button to rebuild processed files if needed.")
