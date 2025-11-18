# -------------------------------------------------------------
# Cashew AI Marketing Dashboard (Streamlit)
# FIX: Updated Ongoing Campaigns based on latest transaction date
# FIX: Fixed Weekly aggregation to use Week-Start Date for chronological sorting
# FIX: Resolved KeyError: 'loyalty_tier' by relying on its presence in sales data 
#      and simplifying the preprocessing (removed redundant merge).
# FIX: Derive 'category' from 'sku_description' by removing weight/size using a robust two-pass regex.
# FIX: Resolved KeyError: 'flavour' in Loyalty Tier Summary by using only 'category'.
# NEW: Replaced Customer Age Band chart with Loyalty Tier Summary Table (with shortened headers).
# NEW: Added Traffic Acquisition and Funnel Efficiency charts.
# NEW: Use event_name from events_amended for ROMS metric.
# UPDATE: Removed 'Unknown' Loyalty Tier row completely from the summary table.
# UPDATE: Ongoing/Upcoming campaigns now use event_name from events_amended.
# UPDATE: Removed 'Top Product Category' from the Loyalty Tier Summary Table.
# UPDATE: Added 'Quarterly' aggregation option.
# UPDATE: Renamed 'Top Payment Method' column to 'Top Online Payment Method'.
# OPTIMIZATION: Replaced dual time aggregation functions with a single, vectorized function 
#               to improve performance, especially for Quarterly aggregation.
# FIX: Quarterly X-axis label format is now 'XQ YYYY'.
# REMOVED: 'Top Online Payment Method' column from Loyalty Tier Summary table.
# -------------------------------------------------------------
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# -------------------- GLOBALS / CONFIG ------------------------
# Chart template + fonts (preserved from dashboard.py)
pio.templates.default = "plotly_white"
pio.templates["plotly_white"].layout.font.size = 16
pio.templates["plotly_white"].layout.title.font.size = 20
pio.templates["plotly_white"].layout.legend.font.size = 14
pio.templates["plotly_white"].layout.yaxis.title.font.size = 14

st.set_page_config(page_title="Cashew AI Marketing Dashboard", layout="wide")

# Date / period related global constants & regex (preserved)
WEEKLY_PERIOD_REGEX = re.compile(r"Wk(\d+)\s*([A-Za-z]{3})(\d{4})")
MONTHLY_FMT = "%Y-%m"
YEARLY_FMT = "%Y"

# Rolling window default (preserved)
DEFAULT_ROLLING_WINDOW = 3

# Campaign Data Structure (Simulated/Hardcoded for example - kept for compatibility with other sections if needed)
CAMPAIGN_DATA = pd.DataFrame([
    {'name': 'Summer Sales Blitz', 'start_date': '2025-07-01', 'end_date': '2025-08-31'},
    {'name': 'Q4 Holiday Push', 'start_date': '2025-11-01', 'end_date': '2025-12-15'},
    {'name': 'New Year Kickoff', 'start_date': '2026-01-01', 'end_date': '2026-01-31'},
    {'name': 'Early Bird Spring', 'start_date': '2025-10-01', 'end_date': '2025-11-15'},
    {'name': 'Black Friday Pre-Sale', 'start_date': '2025-11-16', 'end_date': '2025-11-20'},
])
CAMPAIGN_DATA['start_date'] = pd.to_datetime(CAMPAIGN_DATA['start_date'])
CAMPAIGN_DATA['end_date'] = pd.to_datetime(CAMPAIGN_DATA['end_date'])

# Define color map for consistency
LOYALTY_COLOR_MAP = {
    'Bronze': '#CD7F32', 'Silver': '#C0C0C0',
    'Gold': '#FFD700', 'Platinum': '#7a5195',
    'Unknown': '#808080'
}

# --- Function to determine most popular item for table ---
def get_most_popular(df, mode_col, count_col, is_sum=False):
    """Finds the most popular item in mode_col based on the sum/count of count_col."""
    if df.empty:
        return "N/A"
    
    if is_sum:
        # For product category (sum quantity)
        agg = df.groupby(mode_col)[count_col].sum().reset_index()
    else:
        # For channels/payment (count unique orders)
        agg = df.groupby(mode_col)[count_col].nunique().reset_index()
    
    if agg.empty or agg[count_col].sum() == 0:
        return "N/A"
        
    most_popular = agg.nlargest(1, count_col)
    
    # Format: "Item (Count)"
    return f"{most_popular[mode_col].iloc[0]} ({most_popular[count_col].iloc[0]:,.0f})"
# -------------------------------------------------------------

# -------------------- DATA LOADING ---------------------------
@st.cache_data
def load_data():
    # Load required dataframes
    sales = pd.read_csv("sales_transactions_expanded_modified.csv")
    customers = pd.read_csv("customers.csv")
    ecommerce = pd.read_csv("ecommerce_purchases.csv")
    # Load marketing events data
    events_amended = pd.read_csv("events_amended.csv")
    # NEW: Load traffic acquisition data
    traffic_data = pd.read_csv("traffic_acquisition.csv")
    
    # Keep legacy column name handling
    if "line_net_sales_sgd" in sales.columns:
        sales.rename(columns={"line_net_sales_sgd": "net_sales_sgd"}, inplace=True)

    return sales, customers, ecommerce, events_amended, traffic_data

try:
    # UPDATED: Unpack the new traffic_data
    sales, customers, ecommerce, events_amended, traffic_data = load_data()
except FileNotFoundError as e:
    st.error(f"⚠️ Missing file: {e.filename}. Please ensure all CSV files are in the same folder.")
    st.stop()

# -------------------- PREPROCESSING (centralised) -------------------
def safe_to_datetime(df, col):
    """Convert column to datetime if present and not already."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

# Convert all known datetime columns
safe_to_datetime(sales, 'order_datetime')
safe_to_datetime(ecommerce, 'order_datetime')
safe_to_datetime(customers, 'register_datetime')
safe_to_datetime(events_amended, 'start_date') 
safe_to_datetime(events_amended, 'end_date') # Ensure end_date is datetime for ROMS logic
safe_to_datetime(traffic_data, 'start_date') # NEW: Traffic data conversion


# NEW: Optimized function to create time periods and labels in a single, vectorized pass.
def create_time_periods(df, date_col, agg_option):
    """
    Calculates time grouping columns (time_period, period_dt, time_period_label) 
    in a single vectorized pass for efficiency.

    Adds:
      - time_period (string): 'Wk YYYY-MM-DD' or 'YYYY-MM' or 'YYYYQ1' or 'YYYY' (used for internal grouping)
      - period_dt (datetime): Period start date (used for sorting)
      - time_period_label (string): Human-readable label (used for plotting)
    """
    df = df.copy()
    date_series = df[date_col].dt
    period = None

    if agg_option == "Weekly":
        period = df[date_col].dt.to_period('W')
        df['period_dt'] = period.dt.start_time
        df['time_period'] = 'Wk ' + df['period_dt'].dt.strftime('%Y-%m-%d')
        df['time_period_label'] = 'Wk ' + df['period_dt'].dt.strftime('%b %d')

    elif agg_option == "Monthly":
        period = df[date_col].dt.to_period('M')
        df['period_dt'] = period.dt.start_time
        df['time_period'] = period.astype(str)
        df['time_period_label'] = df['period_dt'].dt.strftime('%b %Y')

    elif agg_option == "Quarterly":
        # Calculate Quarter Period
        period = df[date_col].dt.to_period('Q')
        df['period_dt'] = period.dt.start_time
        df['time_period'] = period.astype(str)
        
        # FIX: Create display label: 'XQ YYYY' (e.g., '1Q 2025')
        df['time_period_label'] = period.dt.quarter.astype(str) + 'Q ' + period.dt.qyear.astype(str)

    else:  # Yearly
        period = df[date_col].dt.to_period('Y')
        df['period_dt'] = period.dt.start_time
        df['time_period'] = period.astype(str)
        df['time_period_label'] = df['period_dt'].dt.strftime('%Y')
        
    return df.dropna(subset=['period_dt', 'time_period']).sort_values('period_dt')
# END NEW TIME AGGREGATION FUNCTION

# Sidebar: aggregation selection (preserved)
st.sidebar.header("📅 Time Aggregation")
agg_option = st.sidebar.radio("Select Time Granularity:", ["Weekly", "Monthly", "Quarterly", "Yearly"])
change_label = {"Weekly": "WoW", "Monthly": "MoM", "Quarterly": "QoQ", "Yearly": "YoY"}[agg_option]

# --- NEW: Rolling Window Slider ---
st.sidebar.markdown("---")
st.sidebar.header("📈 Chart Smoothing")
# Use a fixed max of 10 for display purposes
max_window = 10 
rolling_window = st.sidebar.slider(
    "Rolling Window Size (Periods)", 
    min_value=1, 
    max_value=max_window, 
    value=DEFAULT_ROLLING_WINDOW, 
    step=1,
    help=f"Applies a rolling mean of size 'n' to trend charts. Max {max_window}."
)
# -----------------------------------

# Apply the new, efficient time grouping function to all relevant frames
sales = create_time_periods(sales, 'order_datetime', agg_option)
events_amended = create_time_periods(events_amended, 'start_date', agg_option) 
traffic_data = create_time_periods(traffic_data, 'start_date', agg_option) 
ecommerce = create_time_periods(ecommerce, 'order_datetime', agg_option) 
customers = create_time_periods(customers, 'register_datetime', agg_option)


# NEW: Pre-calculate cumulative new customers (Needed for CLV calculation)
customers = customers.sort_values('register_datetime').reset_index(drop=True)
customers['cumulative_customers'] = customers.index + 1
# NOTE: The time periods are already assigned by create_time_periods above.

# --- FIX: Ensure loyalty_tier is cleaned and filled globally ---
# Based on user input, 'loyalty_tier' is present in sales_transactions_expanded.csv.
# We perform fillna here to handle any missing values.
sales['loyalty_tier'] = sales['loyalty_tier'].fillna('Unknown')
# -------------------------------------------------------------

# --- FIX: Derive Category from sku_description (More Robust Weight/Size Removal) ---
# As per user input, category is derived by removing the weight/size from sku_description.
# We apply two regex passes for robustness:
# 1. Remove common weight/unit patterns at the end (e.g., ' - 200g', ' 1.5L'). Note: Expanded unit length to 3 and added 'pc'.
# 2. Remove size/weight enclosed in parentheses at the end (e.g., ' (200g)').

WEIGHT_REGEX = r'\s*[-–]?\s*\d+(?:\.\d+)?\s*[a-zA-Z]{1,3}(?:g|oz|ml|kg|l|pc)\s*$' 
PARENTHETICAL_SUFFIX_REGEX = r'\s*\([^)]*\)\s*$'

# Pass 1: Remove common weight/unit suffix
sales['derived_category'] = sales['sku_description'].str.replace(WEIGHT_REGEX, '', regex=True)

# Pass 2: Remove parenthetical suffix (e.g. '(200g)'). Also strip whitespace after replacements.
sales['derived_category'] = sales['derived_category'].str.replace(PARENTHETICAL_SUFFIX_REGEX, '', regex=True).str.strip()

# Overwrite or create the 'category' column with the derived value, as it is needed downstream.
sales['category'] = sales['derived_category']
# ---------------------------------------------------

# --- Define 'channel_type' for all relevant DataFrames ---
sales['channel_type'] = sales['platform'].apply(lambda x: 'Offline' if pd.isna(x) or x == '' else 'Online')
ecommerce['channel_type'] = 'Online'

# FIX for ROMS chart: Derive 'channel_type' from 'event_name' in events_amended
if 'event_name' in events_amended.columns:
    events_amended['event_platform'] = events_amended['event_name'].str.split().str[0].fillna('')
    events_amended['channel_type'] = events_amended['event_platform'].apply(
        lambda x: 'Online' if x != '' and x not in ['Retail', 'Pop-up'] else 'Offline' # Adjusted logic to classify based on common offline terms if possible
    )
    if events_amended['event_platform'].eq('').any():
         st.sidebar.info("ℹ️ Some ROMS data is categorized as 'Offline' because the `event_name` was missing.")
else:
    events_amended['channel_type'] = 'Unclassified'
    st.sidebar.error("❌ The column 'event_name' is missing in `events_amended.csv`. Cannot calculate ROMS breakdown.")
# -----------------------------------------------------------------


# --- Campaign Filtering Logic (Dynamic Date, now using events_amended) ---
# Define "Ongoing" based on the latest transaction date in the sales data.
LATEST_TRANSACTION_DATE = sales['order_datetime'].max()
COMPARISON_DATE = LATEST_TRANSACTION_DATE.date() 

# Use events_amended for ongoing/upcoming events
# Rename 'event_name' to 'name' to align with the display logic below
events_for_status = events_amended.rename(columns={'event_name': 'name'}).copy()
# Filter out events with missing dates or names (Crucial for status checks)
events_for_status = events_for_status.dropna(subset=['start_date', 'end_date', 'name'])

# Filter ongoing events (start <= comparison date AND end >= comparison date)
ongoing_campaigns = events_for_status[
    (events_for_status['start_date'].dt.date <= COMPARISON_DATE) &
    (events_for_status['end_date'].dt.date >= COMPARISON_DATE)
].sort_values('end_date')

# Filter upcoming events (start > comparison date)
upcoming_campaigns = events_for_status[
    (events_for_status['start_date'].dt.date > COMPARISON_DATE)
].sort_values('start_date')
# -----------------------------------------------

# --- ROMS Calculation for Last Previous Campaign (Using events_amended) ---
# Find the latest finished campaign based on the end_date in the events_amended data.
events_amended_for_roms = events_amended.copy()
# Ensure we only consider events that have necessary info and are completed
finished_events = events_amended_for_roms[
    (events_amended_for_roms['end_date'].dt.date < COMPARISON_DATE) &
    events_amended_for_roms['return_on_mkt_spend'].notna() &
    events_amended_for_roms['event_name'].notna()
].copy()

last_previous_campaign = None
last_campaign_roms = None

if not finished_events.empty:
    # Identify the last finished *event* (latest end_date)
    last_previous_event = finished_events.sort_values('end_date', ascending=False).iloc[0]
    
    # We use the specific event's ROMS value, not a mean over a period.
    last_campaign_roms = last_previous_event['return_on_mkt_spend']
    
    # Store the necessary details for display
    last_previous_campaign = {
        'name': last_previous_event['event_name'],
        'end_date': last_previous_event['end_date'],
        'roms': last_campaign_roms
    }
# ----------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.header("🚀 Campaign Status")

# --- ROMS Display for Last Previous Campaign ---
if last_previous_campaign is not None and last_campaign_roms is not None:
    campaign_name = last_previous_campaign['name']
    
    # Determine box styling based on ROMS (Using paler colors)
    if last_campaign_roms >= 2.5:
        box_color = "#A5D6A7" # Pale Green
        text_color = "#000000" # Black for contrast
        roms_status = "Excellent"
    elif last_campaign_roms >= 2.0:
        box_color = "#FFE0B2" # Pale Amber
        text_color = "#000000" # Black for contrast
        roms_status = "Good"
    else: # ROMS < 2.0
        box_color = "#FFCDD2" # Pale Red
        text_color = "#000000" # Black for contrast
        roms_status = "Needs Improvement"
        
    roms_text = (
        f"**Last Event:** {campaign_name} "
        f"(Ended: {last_previous_campaign['end_date'].strftime('%d %b %Y')})"
    )
    roms_metric = f"**ROMS:** {last_campaign_roms:,.2f} ({roms_status})"

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<div style="background-color: {box_color}; padding: 10px; border-radius: 5px;">'
        f'<span style="color: {text_color}; font-weight: bold;">{roms_text}</span><br>'
        f'<span style="color: {text_color}; font-weight: bold;">{roms_metric}</span>'
        f'</div>', 
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")

elif last_previous_campaign is not None and last_campaign_roms is None:
    st.sidebar.info(f"Last Event: {last_previous_campaign['name']} found, but ROMS data is missing.")
else:
    st.sidebar.info("No completed marketing events found prior to the latest transaction date.")
# -----------------------------------------------------------------

# Ongoing Campaigns (Now using events_amended)
st.sidebar.markdown(f"**Ongoing Events/Campaigns (as of {COMPARISON_DATE.strftime('%d %b %Y')}):**")
if not ongoing_campaigns.empty:
    for _, row in ongoing_campaigns.iterrows():
        # 'name' is the renamed 'event_name'
        st.sidebar.markdown(f"- **{row['name']}** (Ends: {row['end_date'].strftime('%b %d')})")
else:
    st.sidebar.info("No ongoing event/campaign found.")

# Upcoming Campaigns (Now using events_amended)
st.sidebar.markdown("**Upcoming Events/Campaigns:**")
if not upcoming_campaigns.empty:
    for _, row in upcoming_campaigns.iterrows():
        # 'name' is the renamed 'event_name'
        st.sidebar.markdown(f"- **{row['name']}** (Starts: {row['start_date'].strftime('%b %d')})")
else:
    st.sidebar.info("No upcoming event/campaign found.")
    
st.sidebar.markdown("---")


# NEW: Merge sales and events_amended for ROMS calculation (used in KPI metric and Line chart)
# Grouping is now efficient due to the new time pre-processing.
sales_roms = sales.groupby(['time_period', 'period_dt'], as_index=False)['net_sales_sgd'].sum()
events_roms = events_amended.groupby(['time_period', 'period_dt'], as_index=False).agg(
    attributed_sales_sgd=('attributed_sales_sgd', 'sum'),
    return_on_mkt_spend=('return_on_mkt_spend', 'mean')
)
roms_data = sales_roms.merge(events_roms, on=['time_period', 'period_dt'], how='inner')
roms_data = roms_data.sort_values('period_dt')

# -------------------- DASHBOARD HEADER ---------------------------
st.title("Cashew AI Marketing Dashboard")

# -------------------- I. Marketing Top Line Metrics -----------------
st.subheader("I. Marketing Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

# --- 1. Total Sales (Period Revenue) ---
period_revenue = (
    sales.groupby(['time_period', 'period_dt'], as_index=False)
    .agg(net_sales_sgd=('net_sales_sgd', 'sum'))
    .sort_values('period_dt')
)
if len(period_revenue) < 2:
    current_sales_value = float(period_revenue['net_sales_sgd'].iloc[-1]) if len(period_revenue) else 0.0
    sales_change = 0.0
else:
    current_sales_value = period_revenue['net_sales_sgd'].iloc[-1]
    sales_change = period_revenue['net_sales_sgd'].pct_change().iloc[-1] * 100

col1.metric(f"Total Sales ({agg_option})", f"${current_sales_value:,.0f}", f"{sales_change:.2f}% {change_label}")

# --- 2. Average Value Per Order (AVPO) ---
period_orders = (
    sales.groupby(['time_period', 'period_dt'], as_index=False)
    .agg(
        total_sales=('net_sales_sgd', 'sum'),
        total_orders=('order_id', 'nunique')
    )
    .sort_values('period_dt')
)
period_orders['avpo'] = period_orders['total_sales'] / period_orders['total_orders']

if len(period_orders) < 2:
    current_avpo = float(period_orders['avpo'].iloc[-1]) if len(period_orders) else 0.0
    avpo_change = 0.0
else:
    current_avpo = period_orders['avpo'].iloc[-1]
    # Reverted to calculate change against previous period AVPO
    avpo_change = period_orders['avpo'].pct_change().iloc[-1] * 100

col2.metric(f"Avg Value Per Order ({agg_option})", f"${current_avpo:,.2f}", f"{avpo_change:.2f}% {change_label}")

# --- 3. Return on Marketing Spend (ROMS) ---
if not roms_data.empty and len(roms_data) >= 1:
    current_roms = roms_data['return_on_mkt_spend'].iloc[-1]
    if len(roms_data) >= 2:
        roms_change_abs = current_roms - roms_data['return_on_mkt_spend'].iloc[-2]
    else:
        roms_change_abs = 0.0
else:
    current_roms = 0.0
    roms_change_abs = 0.0

col3.metric(f"Return on Mkt Spend ({agg_option})", f"{current_roms:,.2f}", f"{roms_change_abs:+.2f} pts {change_label}")

# --- 4. New Customers Registered ---
period_new_customers = (
    customers.groupby(['time_period', 'period_dt'], as_index=False)
    .agg(new_customers=('customer_id', 'nunique'))
    .sort_values('period_dt')
)
if len(period_new_customers) < 2:
    current_new_cust = float(period_new_customers['new_customers'].iloc[-1]) if len(period_new_customers) else 0
    new_cust_change = 0.0
else:
    current_new_cust = period_new_customers['new_customers'].iloc[-1]
    new_cust_change = period_new_customers['new_customers'].pct_change().iloc[-1] * 100

col4.metric(f"New Customers ({agg_option})", f"{current_new_cust:,.0f}", f"{new_cust_change:.2f}% {change_label}")


# --- Revenue Bar Chart and Customer Lifetime Value (CLV) Line Chart (Combined) ---

# 1. Calculate cumulative revenue
period_revenue['cumulative_revenue'] = period_revenue['net_sales_sgd'].cumsum()

# 2. Get cumulative customers (max registered up to that period)
cust_max_by_period = customers.groupby(['time_period', 'period_dt'], as_index=False)['cumulative_customers'].max()

# 3. Merge data for CLV calculation
clv_data = period_revenue.merge(cust_max_by_period[['period_dt', 'cumulative_customers']], on='period_dt', how='inner')

# 4. Calculate CLV (Average Cumulative Revenue per Customer)
clv_data['clv_proxy'] = clv_data['cumulative_revenue'] / clv_data['cumulative_customers']

# 5. Merge CLV data and label into the final chart data structure
chart_data = period_revenue.merge(clv_data[['period_dt', 'clv_proxy']], on='period_dt', how='inner')
# Merge in the label from sales (use a distinct list of period_dt and label)
label_map = sales[['period_dt', 'time_period_label']].drop_duplicates()
chart_data = chart_data.merge(label_map, on='period_dt', how='left').dropna(subset=['time_period_label'])
chart_data = chart_data.sort_values('period_dt')

# Limit last 20 periods for weekly/monthly/quarterly
if agg_option.lower() in ['weekly', 'monthly', 'quarterly']:
    chart_data = chart_data.tail(20)

fig_revenue_customers = go.Figure()

# Revenue: Bar Chart (Primary Y-axis)
fig_revenue_customers.add_trace(
    go.Bar(
        x=chart_data['time_period_label'],
        y=chart_data['net_sales_sgd'],
        name='Total Revenue (SGD)',
        marker_color='#0F0C61', # Dark blue
        opacity=0.7
    )
)

# CLV: Line Chart (Secondary Y-axis - y2)
fig_revenue_customers.add_trace(
    go.Scatter(
        x=chart_data['time_period_label'],
        y=chart_data['clv_proxy'],
        mode='lines+markers',
        name='Customer Lifetime Value (ACR)',
        yaxis='y2',
        line=dict(color='#FF9800', width=3) # Orange/Gold
    )
)

# Layout styling with dual axes
fig_revenue_customers.update_layout(
    title="Total Revenue & Customer Lifetime Value (ACR)",
    title_x=0,
    title_y=0.95,
    xaxis=dict(tickangle=45, title='Period'),
    # Primary Y-axis for Revenue
    yaxis=dict(title="Total Revenue (SGD)"),
    # Secondary Y-axis for CLV (overlaying the primary)
    yaxis2=dict(
        title="Customer Lifetime Value (SGD)", 
        overlaying='y', 
        side='right', 
        showgrid=False
    ),
    # Shift legend to the right
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=50, b=40, l=40, r=40)
)

st.plotly_chart(fig_revenue_customers, use_container_width=True)


# -------------------- II. Marketing Performance Breakdown -----------------
st.subheader("II. Marketing Performance Breakdown")

# --- Common Period Filtering Setup for Section II Charts (Last 12 periods) ---
last_periods_dt = None
if agg_option.lower() in ['weekly', 'monthly', 'quarterly']:
    unique_periods = sales[['period_dt']].drop_duplicates().sort_values('period_dt')
    # Use sales to determine the 12 most recent periods
    last_periods_dt = unique_periods.tail(12)['period_dt']


# Set up 2 columns for the remaining charts
chart_col1, chart_col2 = st.columns([1, 1])


# --- 1. ROMS Line Chart (chart_col1) ---
# Aggregate ROMS by time_period and channel_type
events_roms_channel = events_amended.groupby(['time_period', 'period_dt', 'channel_type'], as_index=False).agg(
    return_on_mkt_spend=('return_on_mkt_spend', 'mean')
).sort_values('period_dt')

# Apply period limit if set
if last_periods_dt is not None:
    events_roms_channel = events_roms_channel[events_roms_channel['period_dt'].isin(last_periods_dt)]
    
# Add time_period_label for plotting
label_map = sales[['period_dt', 'time_period_label']].drop_duplicates()
events_roms_channel = events_roms_channel.merge(
    label_map,
    on='period_dt',
    how='left'
).dropna(subset=['time_period_label'])

fig_roms_channel = px.line(
    events_roms_channel,
    x='time_period_label',
    y='return_on_mkt_spend',
    color='channel_type',
    title="Return on Marketing Spend (ROMS)",
    color_discrete_map={'Online': '#2196F3', 'Offline': '#FF9800'}
)

# ADD INDUSTRY BENCHMARK LINE (2.5x)
fig_roms_channel.add_hline(
    y=2.5,
    line_dash="dash",
    line_color="red",
    annotation_text="Benchmark (2.5x)",
    annotation_position="top right",
    annotation_font_color="red",
    annotation_font_size=14
)

# Remove legend
fig_roms_channel.update_layout(
    title_x=0, title_y=0.95,
    yaxis_title="ROMS",
    xaxis=dict(tickangle=45, title='Period'),
    showlegend=False,
    margin=dict(t=50, b=40, l=40, r=40)
)
chart_col1.plotly_chart(fig_roms_channel, use_container_width=True)

# --- 2. Online Sales by Platform (Top 5) (chart_col2) ---
# Filter for online sales (platform is not NaN/empty)
online_sales_all = sales[sales['channel_type'] == 'Online'].copy()

# Aggregate online sales by platform and period
platform_sales = online_sales_all.groupby(['platform', 'time_period', 'period_dt'], as_index=False)['net_sales_sgd'].sum()

# Get the top 5 platforms across all time
top_5_platforms = platform_sales.groupby('platform')['net_sales_sgd'].sum().nlargest(5).index

# Filter data for only the top 5
platform_sales_top = platform_sales[platform_sales['platform'].isin(top_5_platforms)].sort_values('period_dt')

# Apply period limit if set
if last_periods_dt is not None:
    platform_sales_top = platform_sales_top[platform_sales_top['period_dt'].isin(last_periods_dt)]
    
# Add time_period_label for plotting
platform_sales_top = platform_sales_top.merge(
    label_map,
    on='period_dt',
    how='left'
).dropna(subset=['time_period_label'])

# FIX: Determine the order of platforms for stacking (largest total sales at the bottom)
platform_order = (
    platform_sales_top.groupby('platform')['net_sales_sgd']
    .sum()
    .sort_values(ascending=False)
    .index.tolist()
)

fig_platform_sales = px.bar(
    platform_sales_top,
    x='time_period_label',
    y='net_sales_sgd',
    color='platform',
    title="Online Sales by Platform (Top 5)",
    category_orders={"platform": platform_order} 
)
fig_platform_sales.update_layout(
    barmode='stack',
    title_x=0, title_y=0.95,
    yaxis_title="Online Sales (SGD)",
    xaxis=dict(tickangle=45, title='Period'),
    legend_title_text='Platform',
    margin=dict(t=50, b=40, l=40, r=40)
)
chart_col2.plotly_chart(fig_platform_sales, use_container_width=True)

# -------------------- III. Customer Profile Breakdown ---------------------
st.subheader("III. Sales/Orders by Customer Profile")
cust_chart_col1, cust_chart_col2 = st.columns([1, 1])

# --- Average Order Value (AOV) by Loyalty Tier Line Chart (NOW USING ROLLING WINDOW) ---
# 1. Prepare data
aov_loyalty = sales.copy()

# 2. Aggregate AOV across time and tier
aov_loyalty_agg = aov_loyalty.groupby(['time_period', 'period_dt', 'loyalty_tier'], as_index=False).agg(
    total_sales=('net_sales_sgd', 'sum'),
    total_orders=('order_id', 'nunique')
).sort_values('period_dt')

# Calculate AOV
aov_loyalty_agg['aov'] = aov_loyalty_agg['total_sales'] / aov_loyalty_agg['total_orders']

# 2.5 NEW: Apply rolling average *per loyalty tier*
aov_loyalty_agg['rolling_aov'] = aov_loyalty_agg.groupby('loyalty_tier')['aov'].rolling(
    window=rolling_window, 
    min_periods=1, 
    center=False
).mean().reset_index(level=0, drop=True)


# 3. Merge label and handle time window
aov_loyalty_agg = aov_loyalty_agg.merge(
    label_map,
    on='period_dt',
    how='left'
).dropna(subset=['time_period_label'])

# Limit last 6 periods for weekly/monthly/quarterly
if agg_option.lower() in ['weekly', 'monthly', 'quarterly']:
    # Get the latest 6 period_dt values
    recent_periods = aov_loyalty_agg['period_dt'].drop_duplicates().nlargest(6).tolist()
    aov_loyalty_agg = aov_loyalty_agg[aov_loyalty_agg['period_dt'].isin(recent_periods)]
    
aov_loyalty_agg = aov_loyalty_agg.sort_values(['period_dt', 'loyalty_tier'])

with cust_chart_col1:
    if not aov_loyalty_agg.empty and aov_loyalty_agg['total_orders'].sum() > 0:
        fig_aov_loyalty = px.line(
            aov_loyalty_agg,
            x='time_period_label',
            y='rolling_aov', # Use the rolling average for smoothing
            color='loyalty_tier',
            title=f"Avg Order Value (AOV) by Loyalty Tier (Rolling {rolling_window} {agg_option})",
            color_discrete_map=LOYALTY_COLOR_MAP,
            line_shape='spline',
            markers=True
        )
        
        fig_aov_loyalty.update_layout(
            title_x=0, title_y=0.95,
            yaxis_title="Average Order Value (SGD)",
            xaxis_title="Period",
            xaxis=dict(tickangle=45),
            margin=dict(t=50, b=40, l=40, r=40),
            legend_title_text='Loyalty Tier'
        )
        fig_aov_loyalty.update_traces(marker=dict(size=8))
        
        st.plotly_chart(fig_aov_loyalty, use_container_width=True, height=400)
    else:
        st.warning("Not enough data to calculate AOV trend by Loyalty Tier.")
# -----------------------------------------------------------------

# --- Loyalty Tier Summary Table (Removed 'Unknown' Tier, 'Top Product Category', and 'Top Online Payment Method') ---
with cust_chart_col2:
    
    latest_period_dt = sales['period_dt'].max()
    latest_period_label_for_chart = sales.loc[sales['period_dt'].idxmax(), 'time_period_label'] if not sales.empty else "Latest Period"
    
    st.markdown(f"**Loyalty Tier Performance Summary ({latest_period_label_for_chart})**")
    
    if latest_period_dt is None or sales.empty:
        st.info("No data available for the latest period.")
    else:
        # --- 1. Filter and Merge Data ---
        latest_sales = sales[sales['period_dt'] == latest_period_dt].copy()
        # latest_ecommerce = ecommerce[ecommerce['period_dt'] == latest_period_dt].copy() # ecommerce is no longer needed
        
        # Determine all tiers to report on
        all_tiers = sorted(sales['loyalty_tier'].unique())
        
        summary_rows = []

        for tier in all_tiers:
            # Only process tiers that are NOT 'Unknown'
            if tier == 'Unknown':
                continue
            
            # Filter sales data for the current tier
            tier_sales_data = latest_sales[latest_sales['loyalty_tier'] == tier].copy()
            
            # 1. Most Popular Offline Channel (by orders)
            offline_data = tier_sales_data[tier_sales_data['channel_type'] == 'Offline']
            offline_channel = get_most_popular(offline_data, 'channel', 'order_id', is_sum=False)

            # 2. Most Popular Online Channel (by orders)
            online_data = tier_sales_data[tier_sales_data['channel_type'] == 'Online']
            online_channel = get_most_popular(online_data, 'platform', 'order_id', is_sum=False)
            
            # 3. Most Popular Online Payment Method (by orders from ecommerce) - DROPPED
            # tier_orders = tier_sales_data['order_id'].unique()
            # tier_ecommerce = latest_ecommerce[latest_ecommerce['order_id'].isin(tier_orders)]
            # payment_method = get_most_popular(tier_ecommerce, 'payment_method', 'order_id', is_sum=False)

            summary_row = {
                'Tier': tier,
                'Top Offline Channel': offline_channel,
                'Top Online Channel': online_channel,
                # 'Top Online Payment Method': payment_method, # DROPPED
            }
            
            summary_rows.append(summary_row)
            
        summary_df = pd.DataFrame(summary_rows)
        # Sort Tiers for better presentation (no 'Unknown' tier included in order list)
        tier_order = ['Platinum', 'Gold', 'Silver', 'Bronze']
        summary_df['Tier'] = pd.Categorical(summary_df['Tier'], categories=tier_order, ordered=True)
        # Sort and remove any tiers not in the specified order
        summary_df = summary_df.sort_values('Tier').dropna(subset=['Tier'])
        
        # Display the table
        st.dataframe(
            summary_df, 
            hide_index=True, 
            use_container_width=True, 
            height=300 # Set a fixed height for visual consistency
        )

# -------------------- IV. Traffic Acquisition and Funnel Efficiency -----------------
st.subheader("IV. Traffic Acquisition and Funnel Efficiency")
traffic_col1, traffic_col2 = st.columns(2)

# --- Aggregate Traffic Data for Plotting ---
period_traffic = (
    traffic_data.groupby(['time_period', 'period_dt'], as_index=False)
    .agg(
        total_spend=('estimated_spend_sgd', 'sum'),
        total_clicks=('clicks', 'sum'),
        total_conversions=('conversions', 'sum')
    )
    .sort_values('period_dt')
)

# Calculate CVR (in percentage)
period_traffic['cvr'] = np.where(
    period_traffic['total_clicks'] > 0, 
    (period_traffic['total_conversions'] / period_traffic['total_clicks']) * 100, 
    0
)

# Add time_period_label for plotting and apply period limit
period_traffic = period_traffic.merge(
    label_map,
    on='period_dt',
    how='left'
).dropna(subset=['time_period_label'])

# last_periods_dt is defined in Section II and applies to Weekly/Monthly/Quarterly
if last_periods_dt is not None:
    period_traffic = period_traffic[period_traffic['period_dt'].isin(last_periods_dt)]
    
period_traffic = period_traffic.sort_values('period_dt')

# --- Chart 1: Total Ad Spend (traffic_col1) ---
# Apply rolling average to Ad Spend (Total Spend)
period_traffic['rolling_spend'] = period_traffic['total_spend'].rolling(
    window=rolling_window, 
    min_periods=1, 
    center=False
).mean()

fig_spend = px.line(
    period_traffic,
    x='time_period_label',
    y='rolling_spend',
    title=f"Total Ad Spend Trend (Rolling {rolling_window} {agg_option})",
    line_shape='spline',
    markers=True,
    color_discrete_sequence=['#4CAF50'] # Green
)
fig_spend.update_layout(
    title_x=0, title_y=0.95,
    yaxis_title="Ad Spend (SGD)",
    xaxis_title="Period",
    xaxis=dict(tickangle=45),
    margin=dict(t=50, b=40, l=40, r=40)
)
traffic_col1.plotly_chart(fig_spend, use_container_width=True)


# --- Chart 2: Conversion Rate (CVR) (traffic_col2) ---
# Apply rolling average to CVR
period_traffic['rolling_cvr'] = period_traffic['cvr'].rolling(
    window=rolling_window, 
    min_periods=1, 
    center=False
).mean()

fig_cvr = px.line(
    period_traffic,
    x='time_period_label',
    y='rolling_cvr',
    title=f"Conversion Rate (CVR) Trend (Rolling {rolling_window} {agg_option})",
    line_shape='spline',
    markers=True,
    color_discrete_sequence=['#FFC107'] # Amber/Yellow
)
fig_cvr.update_layout(
    title_x=0, title_y=0.95,
    yaxis_title="CVR (%)",
    xaxis_title="Period",
    xaxis=dict(tickangle=45),
    margin=dict(t=50, b=40, l=40, r=40)
)
traffic_col2.plotly_chart(fig_cvr, use_container_width=True)

st.success("✅ Marketing Dashboard Loaded Successfully")
