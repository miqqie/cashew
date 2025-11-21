import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import os
import json
# Import the OpenAI client library
from openai import OpenAI 

# --- LLM API INTEGRATION FUNCTION (USING st.secrets) ---

def generate_llm_summary_api_call(agg_option, change_label, metrics_data):
    """
    Generates a concise 100-word C-suite summary using the OpenAI API and st.secrets.
    
    Args:
        metrics_data (dict): Dictionary containing the latest calculated metrics.
    
    Returns:
        tuple: (llm_output, actionables_status, actionables_content)
               actionables_status is 'amber', 'green', or 'error'.
    """
    
    api_key = st.secrets.get("OPENAI_API_KEY") 
    
    # -------------------- START OF USER-REQUESTED MODIFICATION (Context) --------------------
    caution_message = ""
    is_revenue_lower = metrics_data['revenue_change'] < 0
    is_profit_lower = metrics_data['profit_change'] < 0
    
    # Apply caution only if Revenue or Profit are lower than the previous period (WoW/MoM/QoQ/YoY).
    if is_revenue_lower or is_profit_lower:
        caution_message = (
            f"**IMPORTANT CONTEXT:** The data provided is for the latest {agg_option} period, which may be incomplete. "
            "If Revenue or Profit show a negative trend, seamlessly integrate a note of caution, "
            "stating that the data should be interpreted with caution due to the incomplete period, and provide possible reasons "
            "for the lower figures (e.g., lower sales due to partial data, seasonality, or temporary market shifts). "
            "Ensure this caution is integrated into the 'Key Trends' bullets. Do NOT include this context section in your final output."
        )
    # -------------------- END OF USER-REQUESTED MODIFICATION (Context) --------------------
    
    # Fallback/Error Check
    if not api_key:
        error_content = f"""
## ⚠️ Configuration Error: OpenAI API Key Missing
* Please ensure your API key is correctly configured in **.streamlit/secrets.toml** under the key `OPENAI_API_KEY`.
* Current Data: Revenue Change: {metrics_data['revenue_change']:.1f}%, DSI: {metrics_data['dsi_value']} days.
"""
        return error_content, "error", "" # Return error and empty actionable

    client = OpenAI(api_key=api_key)
    
    # 1. Format the data into a clean, structured table for the LLM
    data_table = f"""
| Metric | Latest Value (K SGD) | Change ({change_label}) | Status |
|---|---|---|---|
| Revenue | {metrics_data['revenue_latest']/1000:,.1f}K | {metrics_data['revenue_change']:.1f}% | N/A |
| Profit | {metrics_data['profit_latest']/1000:,.1f}K | {metrics_data['profit_change']:.1f}% | N/A |
| Online Share | {metrics_data['online_share']:.1f}% | N/A | N/A |
| DSI | {metrics_data['dsi_value']} days | N/A | {metrics_data['dsi_status']} |
"""
    
    prompt = f"""
    Act as a C-suite financial analyst for an FMCG snack company in Singapore. The DSI target is 20-45 days.
    
    Your task is to analyze the data provided in the table below and write a summary for the executive team.
    
    **Constraint 1 (Structure):** The output must be formatted as exactly three Markdown lists.
        1. The first list, under "Key Trends", must contain **3 key trend bullet points**.
        2. The second list, under "Immediate Actionables Required", must contain either **1-2 bullet points** detailing immediate actions for critical issues (like DSI being 'Risk' or sharp drops in Profit/Revenue), OR a single sentence/bullet point stating **"No immediate actionables required at this time."**
        3. The third list, under "Growth Strategies", must contain **2 strategic action bullet points**.
    
    **Constraint 2 (Content - Growth Strategies):** The Growth Strategy points should link back to key trends and should focus on growth, profit, and inventory (DSI) and address external factors (e.g., health trends, supply chain, global competition, labor costs and other important trends) that Singapore FMCG snack companies face. If key trends in on yearly data, strategies should be longer term. for monthly and weekly, strategies should be more short term and set achieveable goals to reach in the short term. There should be sufficient variation in the strategies proposed. the strategies must not sound the same across time periods. Set benchmakr targets which should be different for the short and long term.
    
    **Constraint 3 (Formatting):** Do NOT combine the profit and revenue metrics into a single sentence. Use the 'K' suffix for all financial values, as the input table is in thousands of SGD.
    
    **Constraint 4 (Length):** The total word count must be under 150 words.
    
    **Constraint 5 (Headers):** You MUST use the following **three** headers, and ONLY these three, in the order specified: "**Key Trends**", "**Immediate Actionables Required**", and "**Growth Strategies**". Do not use any other headers or modify these three.
    
    **Data Table for {agg_option} Analysis:**
    {data_table}
    
    """
    
    final_prompt_content = caution_message + prompt
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a concise financial analyst for C-suite executives. You write strictly in markdown, following ALL constraints, especially the list structure, word count, and separation of metrics."},
                {"role": "user", "content": final_prompt_content}
            ],
            max_tokens=200, 
            temperature=0.1
        )
        llm_output = response.choices[0].message.content
        
        # -------------------- START OF USER-REQUESTED MODIFICATION (Extraction and Removal) --------------------
        # Regex to capture content between "Immediate Actionables Required" and the next header "Growth Strategies"
        # We capture the content and the header itself.
        match = re.search(r'\*\*Immediate Actionables Required\*\*(.*?)\n\*\*Growth Strategies\*\*', llm_output, re.DOTALL | re.IGNORECASE)
        
        actionables_content = ""
        actionables_status = "unknown"
        llm_output_clean = llm_output # Start with the original output
        
        if match:
            # Clean up the captured group (the actionables list)
            actionables_content = match.group(1).strip()
            
            # --- KEY CHANGE: REMOVE THE SECTION ENTIRELY FROM THE MAIN OUTPUT ---
            # Replace the entire block (header + content) with the next valid header
            llm_output_clean = re.sub(r'\*\*Immediate Actionables Required\*\*(.*?)\n\*\*Growth Strategies\*\*', 
                                      '\n\n**Growth Strategies**', # Replace the whole block with two newlines and the next header
                                      llm_output, 
                                      flags=re.DOTALL | re.IGNORECASE,
                                      count=1)
            
            # Determine status: Amber if specific actions, Green if 'No immediate actionables'
            if "no immediate actionables required at this time" in actionables_content.lower():
                actionables_status = "green"
            else:
                actionables_status = "amber"
                
            return llm_output_clean, actionables_status, actionables_content

        # Fallback if regex fails to find the section
        return llm_output, "unknown", ""
        # -------------------- END OF USER-REQUESTED MODIFICATION (Extraction and Removal) --------------------
        
    except Exception as e:
        error_content = f"""
## ⚠️ LLM Error: Summary Generation Failed
* Could not execute API call. Details: `{e}`
* Check network connection and model availability.
"""
        return error_content, "error", ""


# -------------------- GLOBALS / CONFIG ------------------------
# Chart template + fonts (preserved)
pio.templates.default = "plotly_white"
pio.templates["plotly_white"].layout.font.size = 16
pio.templates["plotly_white"].layout.title.font.size = 20
pio.templates["plotly_white"].layout.legend.font.size = 14
pio.templates["plotly_white"].layout.yaxis.title.font.size = 14

st.set_page_config(page_title="Cashew AI Dashboard", layout="wide")

# Date / period related global constants & regex
WEEKLY_PERIOD_REGEX = re.compile(r"Wk(\d+)\s*([A-Za-z]{3})(\d{4})") # expects 'Wk3 Sep2025' or similar
MONTHLY_FMT = "%Y-%m"  # format used when we store period as period.to_period('M').astype(str)
YEARLY_FMT = "%Y"

# Rolling window default (kept interactive in sidebar but provide default)
DEFAULT_ROLLING_WINDOW = 3

# -------------------- DATA LOADING (OPTIMIZED) ---------------------------
def safe_to_datetime(df, col):
    """Convert column to datetime if present and not already."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=False) # Ensure consistent parsing
    return df

@st.cache_data
def load_and_preprocess_data():
    sales = pd.read_csv("sales_transactions_expanded_modified.csv")
    sku = pd.read_csv("sku_master_expanded.csv")
    customers = pd.read_csv("customers.csv")
    traffic = pd.read_csv("traffic_acquisition.csv")
    events = pd.read_csv("events_amended.csv")
    ecommerce = pd.read_csv("ecommerce_purchases.csv")
    inventory_value = pd.read_csv("diminventoryvalue.csv")

    # Keep legacy column name handling
    if "line_net_sales_sgd" in sales.columns:
        sales.rename(columns={"line_net_sales_sgd": "net_sales_sgd"}, inplace=True)

    # Convert all known datetime columns (ONE TIME)
    sales = safe_to_datetime(sales, 'order_datetime')
    ecommerce = safe_to_datetime(ecommerce, 'order_datetime')
    customers = safe_to_datetime(customers, 'register_datetime')
    traffic = safe_to_datetime(traffic, 'start_date')
    events = safe_to_datetime(events, 'start_date')
    inventory_value = safe_to_datetime(inventory_value, 'Week Ending Date') # NEW: Convert 'Week Ending Date'

    # --- Core Sales Pre-calculation (OPTIMIZATION) ---
    # Merge sales with SKU cost and calculate profit
    sales = sales.merge(sku[['sku', 'cost_unit_price_sgd']], on='sku', how='left')
    sales['profit_sgd'] = sales['net_sales_sgd'] - (sales['cost_unit_price_sgd'] * sales['quantity'])
    sales['cogs'] = sales['cost_unit_price_sgd'] * sales['quantity']
    sales['channel_type'] = sales['platform'].apply(lambda x: 'Offline' if pd.isna(x) or x == '' else 'Online')
    
    # --- DSI Fallback Pre-calculation ---
    last_non_zero_inventory = 0
    if not inventory_value.empty:
        non_zero_inventory = inventory_value[inventory_value['Average Inventory'] > 0]
        if not non_zero_inventory.empty:
            # Get the last non-zero inventory value as a safe proxy
            last_non_zero_inventory = non_zero_inventory['Average Inventory'].iloc[-1]
            
    # Load Projected Revenue Data (NEW)
    try:
        projected_revenue = pd.read_csv("dimprojectedrevenue.csv")
        if 'Quarters' not in projected_revenue.columns:
             raise ValueError("The 'Quarters' column is missing in dimprojectedrevenue.csv, and 'Date' is removed.")
        projected_revenue.rename(columns={'Total_Revenue': 'net_sales_sgd', 'Quarters': 'time_period'}, inplace=True)
        projected_revenue = projected_revenue.dropna(subset=['time_period', 'net_sales_sgd'])
        projected_revenue['time_period'] = projected_revenue['time_period'].astype(str)
    except FileNotFoundError:
        st.warning("⚠️ dimprojectedrevenue.csv not found. Forecasted revenue will not be shown.")
        projected_revenue = pd.DataFrame({'time_period': [], 'net_sales_sgd': []})
    except ValueError as e:
        st.error(f"⚠️ Error loading dimprojectedrevenue.csv: {e}")
        projected_revenue = pd.DataFrame({'time_period': [], 'net_sales_sgd': []})

    return sales, sku, customers, traffic, events, ecommerce, inventory_value, projected_revenue, last_non_zero_inventory

try:
    # Load all data and perform one-time preprocessing
    sales_raw, sku, customers, traffic_raw, events_raw, ecommerce_raw, inventory_value, projected_revenue, last_non_zero_inventory = load_and_preprocess_data()
except FileNotFoundError as e:
    st.error(f"⚠️ Missing file: {e.filename}. Please ensure all CSV files are in the same folder.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ An error occurred during data loading and preprocessing: {e}")
    st.stop()
    
# Deep copy the raw dataframes to apply aggregation without re-running load_and_preprocess_data
sales = sales_raw.copy()
ecommerce = ecommerce_raw.copy()
events = events_raw.copy()
traffic = traffic_raw.copy()

# -------------------- PREPROCESSING (centralised) -------------------
# Provide a reusable assign_time_group that sets time_period consistently
@st.cache_data(show_spinner=False)
def assign_time_group(df, date_col, agg_option):
    """
    Adds:
      - time_period (string): 'Wk{n} {MonYear}' or 'YYYY-MM' or 'YYYY-QQ' or 'YYYY'
    
    This function is slightly optimized for vectorization.
    """
    df = df.copy()
    
    if df.empty or date_col not in df.columns:
        df['time_period'] = pd.Series([], dtype='object')
        df['week_start'] = pd.Series([], dtype='datetime64[ns]')
        return df
        
    df = df.dropna(subset=[date_col])

    if agg_option == "Weekly":
        # Vectorised week-in-month calculation
        week_num = ((df[date_col].dt.day - 1) // 7) + 1
        month_year = df[date_col].dt.strftime('%b%Y') # 'Sep2025'
        df['time_period'] = 'Wk' + week_num.astype(str) + ' ' + month_year
        df['week_start'] = df[date_col] # keep original date for sorting; we'll derive period_dt later
    elif agg_option == "Monthly":
        df['time_period'] = df[date_col].dt.to_period('M').astype(str) # '2025-09'
        df['week_start'] = df[date_col]
    elif agg_option == "Quarterly":
        df['time_period'] = df[date_col].dt.to_period('Q').astype(str) # '2025Q3'
        df['week_start'] = df[date_col]
    else: # Yearly
        df['time_period'] = df[date_col].dt.to_period('Y').astype(str) # '2025'
        df['week_start'] = df[date_col]
        
    return df

# Centralised function to convert time_period strings to a sortable period_dt and a display label
# This function is used often, so it should be fast.
@st.cache_data(show_spinner=False)
def period_string_to_dt_and_label(series, agg_option):
    """
    Input: pd.Series of strings (time_period)
    Returns: tuple (period_dt_series (datetime), period_label_series (str_display))
    """
    agg_lower = agg_option.lower()
    s = series.astype(str).copy()

    if agg_lower == "weekly":
        extracted = s.str.extract(r'Wk(\d+)\s*([A-Za-z]{3})(\d{4})')
        extracted.columns = ['week_num', 'month_str', 'year']
        extracted['week_num'] = pd.to_numeric(extracted['week_num'], errors='coerce').fillna(1).astype(int)
        extracted['year'] = pd.to_numeric(extracted['year'], errors='coerce').astype('Int64')
        extracted['month_num'] = pd.to_datetime(extracted['month_str'], format='%b', errors='coerce').dt.month
        base = pd.to_datetime(extracted['year'].astype(str) + '-' + extracted['month_num'].astype(str) + '-01', errors='coerce')
        period_dt = base + pd.to_timedelta((extracted['week_num'] - 1) * 7, unit='d')
        period_dt = period_dt - pd.to_timedelta(period_dt.dt.weekday, unit='d') # align to Monday
        label = s 
        return period_dt, label

    elif agg_lower == "monthly":
        period_dt = pd.to_datetime(s, format=MONTHLY_FMT, errors='coerce')
        label = period_dt.dt.strftime('%b %Y').fillna(s)
        return period_dt, label

    elif agg_lower == "quarterly":
        period_index = pd.PeriodIndex(s, freq='Q')
        period_dt = period_index.to_timestamp(how='start')
        label = s 
        return period_dt, label

    elif agg_lower == "yearly":
        period_dt = pd.to_datetime(s, format=YEARLY_FMT, errors='coerce')
        label = period_dt.dt.strftime('%Y').fillna(s)
        return period_dt, label

    return pd.to_datetime(s, errors='coerce'), s

# Sidebar: aggregation selection (keeps behaviour)
st.sidebar.header("📅 Time Aggregation")
agg_option = st.sidebar.radio("Select Time Granularity:", ["Weekly", "Monthly", "Quarterly", "Yearly"])
change_label = {"Weekly": "WoW", "Monthly": "MoM", "Quarterly": "QoQ", "Yearly": "YoY"}[agg_option]

# Apply to all relevant frames (now much faster due to cache/vectorization)
sales = assign_time_group(sales, 'order_datetime', agg_option)
ecommerce = assign_time_group(ecommerce, 'order_datetime', agg_option)
events = assign_time_group(events, 'start_date', agg_option)
traffic = assign_time_group(traffic, 'start_date', agg_option)

# Add period_dt and human-readable label to sales (and reuse later if needed)
sales['period_dt'], sales['time_period_label'] = period_string_to_dt_and_label(sales['time_period'], agg_option)
ecommerce['period_dt'], ecommerce['time_period_label'] = period_string_to_dt_and_label(ecommerce['time_period'], agg_option)
events['period_dt'], events['time_period_label'] = period_string_to_dt_and_label(events['time_period'], agg_option)
traffic['period_dt'], traffic['time_period_label'] = period_string_to_dt_and_label(traffic['time_period'], agg_option)

# -------------------- DYNAMIC DSI CALCULATION FUNCTION (Required for Time Series and Topline Metric) --------------------
# NOTE: This function's logic is preserved as it is the required calculation.
# It is called on a group-by object below to calculate all DSI values at once.
def calculate_dsi_for_period(period_name, period_sales_df, inventory_df):
    """Calculates DSI for a single time period."""

    if period_sales_df.empty or 'cogs' not in period_sales_df.columns:
        return 0

    total_cogs = period_sales_df['cogs'].sum()

    # Determine date range and days for the current period
    min_date_period = period_sales_df['order_datetime'].min()
    max_date_period = period_sales_df['order_datetime'].max()
    if pd.isna(min_date_period) or pd.isna(max_date_period):
        total_days = 30 # Fallback to a proxy if dates are bad
    else:
        total_days = (max_date_period - min_date_period).days + 1

    average_inventory = 0

    # 1. Filter inventory entries that fall *within* the sales period
    filtered_inventory = inventory_df[
        (inventory_df['Week Ending Date'] >= min_date_period) &
        (inventory_df['Week Ending Date'] <= max_date_period)
    ]

    if not filtered_inventory.empty:
        # Use the average of all inventory entries within the period
        average_inventory = filtered_inventory['Average Inventory'].mean()
    else:
        # 2. If no weekly inventory entry falls *exactly* within the sales period,
        # find the latest inventory value that occurred on or before the period's end date.
        inventory_up_to_period = inventory_df[
            inventory_df['Week Ending Date'] <= max_date_period
        ].sort_values('Week Ending Date')

        if not inventory_up_to_period.empty:
            # Use the single latest inventory value (last row) as the Average Inventory proxy
            average_inventory = inventory_up_to_period['Average Inventory'].iloc[-1]
            
    # 3. Calculate DSI: (Average Inventory / COGS) * Days in Period
    if total_cogs > 0 and average_inventory > 0:
        dsi_value = (average_inventory / total_cogs) * total_days
        return round(dsi_value)
    else:
        return 0
# -------------------- MERGE & CALCULATIONS -----------------------
# Merging and COGS/Profit calculation is now done once in load_and_preprocess_data.
# sales already contains 'profit_sgd', 'cogs', and 'channel_type'

# -------------------- FULL METRIC CALCULATION BLOCK (OPTIMIZED) -------------------
# This block calculates all necessary variables once.

# 1. Grouped Revenue and Profit
period_revenue = (
    sales.groupby(['time_period', 'period_dt', 'time_period_label'], as_index=False)
    .agg(net_sales_sgd=('net_sales_sgd', 'sum'))
    .sort_values('period_dt')
)
period_profit = (
    sales.groupby(['time_period', 'period_dt', 'time_period_label'], as_index=False)
    .agg(profit_sgd=('profit_sgd', 'sum'))
    .sort_values('period_dt')
)

# 2. Online Share
share_summary = (
    sales.groupby(['time_period', 'period_dt', 'channel_type'], as_index=False)['net_sales_sgd']
    .sum()
    .pivot_table(index=['time_period', 'period_dt'], columns='channel_type', values='net_sales_sgd', fill_value=0)
    .reset_index()
)
if 'Online' not in share_summary.columns: share_summary['Online'] = 0
if 'Offline' not in share_summary.columns: share_summary['Offline'] = 0
share_summary['online_share'] = share_summary['Online'] / (share_summary['Online'] + share_summary['Offline']) * 100
share_summary = share_summary.sort_values('period_dt')


# 3. Topline Metric Values (Last Period)
if len(period_revenue) >= 1:
    current_revenue = period_revenue['net_sales_sgd'].iloc[-1]
    current_profit = period_profit['profit_sgd'].iloc[-1]
    current_online = share_summary['online_share'].iloc[-1]
    latest_period_str = period_revenue.iloc[-1]['time_period']
    latest_period_label = period_revenue.iloc[-1]['time_period_label']
    
    # >> FIX: Define the missing variable 'latest_period_dt'
    latest_period_dt = period_revenue.iloc[-1]['period_dt'] 
    
    # Change calculation
    revenue_change = period_revenue['net_sales_sgd'].pct_change().iloc[-1] * 100 if len(period_revenue) >= 2 else 0.0
    profit_change = period_profit['profit_sgd'].pct_change().iloc[-1] * 100 if len(period_profit) >= 2 else 0.0
    online_change = current_online - share_summary['online_share'].iloc[-2] if len(share_summary) >= 2 else 0.0

else: # Default for empty data
    current_revenue = current_profit = current_online = revenue_change = profit_change = online_change = 0.0
    latest_period_str = None
    latest_period_label = ""
    latest_period_dt = None # Set a safe default for the date object
    

# 4. DSI Metrics (Calling the DSI function for the latest period only)
if latest_period_str:
    latest_sales_period = sales[sales['time_period'] == latest_period_str].copy()
    dsi_value = calculate_dsi_for_period(latest_period_str, latest_sales_period, inventory_value)
else:
    dsi_value = 0

dsi_status = "Healthy" if 30 <= dsi_value <= 45 else ("Risk" if dsi_value > 45 else "Excellent")
change_label = {"Weekly": "WoW", "Monthly": "MoM", "Quarterly": "QoQ", "Yearly": "YoY"}.get(agg_option, "Period-over-Period")


# -------------------- DASHBOARD HEADER AND SUMMARY DISPLAY ---------------------------
st.title("Cashew AI Dashboard")

# --- Aggregate Metrics for LLM Call ---
metrics_for_llm = {
    "revenue_latest": current_revenue,
    "revenue_change": revenue_change,
    "profit_latest": current_profit,
    "profit_change": profit_change,
    "online_share": current_online,
    "dsi_value": dsi_value,
    "dsi_status": dsi_status,
}

# --- GENERATE AND DISPLAY LLM SUMMARY ---

# FIX: Update the function call to expect the new return tuple
llm_summary_markdown, actionables_status, actionables_content = generate_llm_summary_api_call(agg_option, change_label, metrics_for_llm)

# -------------------- END OF USER-REQUESTED MODIFICATION (Display) --------------------
# -------------------- START OF USER-REQUESTED MODIFICATION (Display) --------------------
# 1. Display the Immediate Actionables box using st.success (green) or st.warning (amber)
if actionables_status == "amber":
    st.warning(f"**🔴 IMMEDIATE ACTION REQUIRED**\n\n{actionables_content}")
elif actionables_status == "green":
    st.success(f"**🟢 ACTION STATUS: HEALTHY**\n\n{actionables_content}")
elif actionables_status == "error":
    # If there was an error, the main markdown already contains the error message, so we just pass.
    pass
else:
    # If extraction failed, display the content neutrally in the summary
    pass

# 2. Display the full LLM summary
st.markdown(llm_summary_markdown, unsafe_allow_html=True)
st.markdown("---") # Separator after the summary

# -------------------- END OF USER-REQUESTED MODIFICATION (Display) --------------------

#-------------------- I. Growth & Profitability -----------------
st.subheader("I. Growth & Profitability")
col1, col2, col3, col4 = st.columns(4)

# Use pre-calculated metrics
col1.metric(f"Total Revenue ({agg_option})", f"${current_revenue:,.0f}", f"{revenue_change:.2f}% {change_label}")
col2.metric(f"Total Profit ({agg_option})", f"${current_profit:,.0f}", f"{profit_change:.2f}% {change_label}")
col3.metric("Online Revenue Share", f"{current_online:.1f}%", f"{online_change:.1f} pp {change_label}")

# DSI Display
status_color = "green" if dsi_status in ["Healthy", "Excellent"] else "red"
col4.markdown(
    f"""
    <div style='text-align:left; font-family:sans-serif;'>
        <div style='font-size: 0.9em; color:white; margin:0.05em 0;'>Days Sales of Inventory (DSI)</div>
        <div style='font-size:2.2em; color:white; margin:-0.05em 0;'>{dsi_value} days</div>
        <div style='font-size: 0.9em; color:{status_color}; font-weight:bold; margin:-0.3em 0;'>{dsi_status}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------- Total Revenue for Each Period with Holiday Lines & Forecast ---
# --- Load holiday data ---
dimdate = pd.read_csv("dimdate.csv")
dimdate['holiday_dt'] = pd.to_datetime(dimdate['Date'], errors='coerce', dayfirst=False)
dimdate = dimdate[dimdate['Public_Holiday'] == 'Y'].dropna(subset=['holiday_dt', 'PH_name'])

# --- Map holiday dates to the same period as revenue ---
if agg_option.lower() == 'weekly':
    dimdate['period_dt'] = dimdate['holiday_dt'].dt.to_period('W').apply(lambda r: r.start_time)
elif agg_option.lower() == 'monthly':
    dimdate['period_dt'] = dimdate['holiday_dt'].dt.to_period('M').apply(lambda r: r.start_time)
elif agg_option.lower() == 'quarterly':
    dimdate['period_dt'] = dimdate['holiday_dt'].dt.to_period('Q').apply(lambda r: r.start_time)
else:  # yearly
    dimdate['period_dt'] = dimdate['holiday_dt'].dt.to_period('Y').apply(lambda r: r.start_time)

# Aggregate holidays by period
holiday_bins = dimdate.groupby('period_dt', as_index=False).agg(
    PH_name=('PH_name', lambda x: "; ".join(x.unique()))
)

# Use period_revenue calculated above as total_revenue
total_revenue = period_revenue.copy()
total_revenue.rename(columns={'net_sales_sgd': 'net_sales_sgd_actual'}, inplace=True) # Rename for clarity

# --- Forecasted Revenue Logic (MODIFIED to remove projected data) ---
# MODIFICATION: This block ensures 'plot_data' only contains actual revenue and explicitly removes projection.
plot_data = total_revenue.copy()
plot_data['is_forecast'] = False
plot_data['net_sales_sgd'] = plot_data['net_sales_sgd_actual']
plot_data['net_sales_sgd_proj'] = np.nan
# END MODIFICATION

# Merge holiday info by period_dt 
plot_data = plot_data.merge(
    holiday_bins[['period_dt', 'PH_name']],
    on='period_dt',
    how='left'
)

# Only mark target holidays (assumes target_holidays is defined)
target_holidays = ["Chinese New Year", "Hari Raya Puasa"]
plot_data['is_target_holiday'] = plot_data['PH_name'].apply(
    lambda x: any(h in str(x) for h in target_holidays)
)

# Sort chronologically
plot_data = plot_data.sort_values('period_dt')

# Limit last 10 periods for weekly/monthly/quarterly
if agg_option.lower() in ['weekly', 'monthly', 'quarterly']:
    plot_data = plot_data.tail(10)

# --- Plot revenue line using go.Figure for layering ---
fig = go.Figure()

# Plot Actual Sales (Light Blue, Bold Line) - Only points where is_forecast is False
actual_data = plot_data[plot_data['is_forecast'] == False]
fig.add_trace(go.Scatter(
    x=actual_data['time_period_label'],
    y=actual_data['net_sales_sgd'],
    mode='lines+markers',
    name='Actual Revenue',
    # MODIFICATION: Changed color to 'lightblue' and added width=4 for bold
    line=dict(color='lightblue', width=4) 
))

# Plot Projected Sales (Orange Dashed Line) - REMOVED

# Map for shorter holiday labels (assumed to be defined)
holiday_labels = {"Chinese New Year": "CNY", "Hari Raya Puasa": "Hari Raya P"}

# Add vertical lines and annotations (HOLIDAY FIX: Exclude Quarterly/Yearly)
categories = list(plot_data['time_period_label'].unique()) 
n = len(categories)

for idx, row in plot_data.iterrows():
    if row['is_target_holiday']:
        # Skip drawing if aggregation is Yearly or Quarterly
        if agg_option in ["Yearly", "Quarterly"]:
            continue

        period_label = row['time_period_label']
        try:
            x_position = categories.index(period_label)
        except ValueError:
            continue 

        # --- Add vertical yellow line at exact position --- 
        fig.add_shape(
            type="line",
            x0=x_position, x1=x_position,
            y0=0, y1=1,
            xref='x', yref='paper',
            line=dict(color='yellow', width=2, dash='dot')
        )
        
        holidays_in_period = [h.strip() for h in str(row['PH_name']).split(";")]
        holiday_in_period = next((h for h in holidays_in_period if h in target_holidays), None)
        if holiday_in_period:
            short_label = holiday_labels.get(holiday_in_period, holiday_in_period)

            fig.add_annotation(
                x=x_position,  
                y=1.02,
                xref='x',
                yref='paper',
                text=short_label,
                showarrow=False,
                font=dict(color="orange", size=12),
                align="center"
            )

# Layout styling
fig.update_layout(
    title_text="Total Revenue",
    title_x=0,
    title_y=0.95,
    yaxis_title="Total Revenue (SGD)",
    xaxis=dict(tickangle=45, title='Period', categoryorder='array', categoryarray=categories), 
    legend=dict(yanchor="top", y=1.0, xanchor="left", x=0),
    margin=dict(t=50, b=40, l=40, r=40)
)

# Render in Streamlit
col = st.columns(1)[0]
col.plotly_chart(fig, use_container_width=True)

# -------------------- II. Visual Insights (Charts) -----------------
col1, col2, col3 = st.columns([1, 0.8, 1.2])

# --- Revenue Change by Channel ---
# Determine last two period labels in a robust way
if len(period_revenue) >= 2:
    last_period_time_str = period_revenue['time_period'].iloc[-1]
    prev_period_time_str = period_revenue['time_period'].iloc[-2]
    current_period_mask = sales['time_period'] == last_period_time_str
    previous_period_mask = sales['time_period'] == prev_period_time_str
    current_period_label_fmt = latest_period_label
else:
    current_period_mask = previous_period_mask = pd.Series([False] * len(sales), index=sales.index)
    current_period_label_fmt = latest_period_label

current_channel_revenue = sales[current_period_mask].groupby('channel', as_index=False)['net_sales_sgd'].sum()
previous_channel_revenue = sales[previous_period_mask].groupby('channel', as_index=False)['net_sales_sgd'].sum()

revenue_channel_change = pd.merge(
    current_channel_revenue, previous_channel_revenue, on='channel', how='outer', suffixes=('_current', '_previous')
).fillna(0)

revenue_channel_change['revenue_change_abs'] = revenue_channel_change['net_sales_sgd_current'] - revenue_channel_change['net_sales_sgd_previous']
revenue_channel_change['bar_color'] = revenue_channel_change['revenue_change_abs'].apply(lambda x: 'green' if x > 0 else 'red')
revenue_channel_change_sorted = revenue_channel_change.sort_values('revenue_change_abs', ascending=False)

fig_revenue_change_by_channel = px.bar(
    revenue_channel_change_sorted,
    x='channel',
    y='revenue_change_abs',
    text='revenue_change_abs',
    color='bar_color',
    color_discrete_map={'green': 'green', 'red': 'red'},
    title=f"Change in Revenue by Channel ({current_period_label_fmt})",
)

fig_revenue_change_by_channel.update_traces(
    showlegend=False,
    texttemplate='$%{text:.0f}',
    textfont=dict(size=14)
)

fig_revenue_change_by_channel.update_layout(
    title_x=0, title_y=0.95, yaxis_title="Absolute Change in Revenue (SGD)",
    xaxis=dict(tickangle=45, title=''),  
    margin=dict(t=50, b=40, l=40, r=40)
)
col1.plotly_chart(fig_revenue_change_by_channel, use_container_width=True)

# --- Profit by Channel ---
last_period_str = period_profit['time_period'].iloc[-1] if not period_profit.empty else None
formatted_period = latest_period_label

current_period_profit = sales[sales['time_period'] == last_period_str] if last_period_str else sales.iloc[0:0]
profit_channel = current_period_profit.groupby('channel', as_index=False)['profit_sgd'].sum().sort_values('profit_sgd', ascending=False)
total_profit = profit_channel['profit_sgd'].sum() if not profit_channel.empty else 1.0
top_3_profit_channel = profit_channel.head(3).copy()
top_3_profit_channel['profit_percentage'] = (top_3_profit_channel['profit_sgd'] / total_profit) * 100
top_3_profit_channel['formatted_label'] = top_3_profit_channel.apply(
    lambda x: f"${x['profit_sgd']:,.0f} ({x['profit_percentage']:.1f}%)", axis=1
)

fig_profit_by_channel = px.bar(
    top_3_profit_channel,
    x='channel',
    y='profit_sgd',
    text='formatted_label'
)
fig_profit_by_channel.update_traces(marker_color=['#4CAF50'] * len(top_3_profit_channel), textfont=dict(size=14))
fig_profit_by_channel.update_layout(
    title=f"Profit by Channel ({formatted_period})",
    title_x=0, title_y=0.95,
    yaxis_title="Profit (SGD)",
    xaxis=dict(tickangle=45, title=''),
    margin=dict(t=50, b=40, l=40, r=40)
)
col2.plotly_chart(fig_profit_by_channel, use_container_width=True)

# --- Online Sales (Smoothed) ---
# Select last N periods (6 for weekly/monthly/quarterly, else all for yearly)
agg_lower = agg_option.lower()
unique_periods = sales[['period_dt', 'time_period_label']].drop_duplicates().sort_values('period_dt')
if agg_lower in ['weekly', 'monthly', 'quarterly']:
    last_periods = unique_periods.tail(6)
else:
    last_periods = unique_periods

sales_filtered = sales[sales['period_dt'].isin(last_periods['period_dt'])]
# Aggregate by period_dt, label, channel_type
sales_by_type = sales_filtered.groupby(['period_dt', 'time_period_label', 'channel_type'], as_index=False)['net_sales_sgd'].sum()
window_size = st.sidebar.slider("Select Rolling Average Window Size", 1, 7, DEFAULT_ROLLING_WINDOW, 1)
sales_by_type = sales_by_type.sort_values(['channel_type', 'period_dt'])
sales_by_type['rolling_avg'] = sales_by_type.groupby('channel_type')['net_sales_sgd'].transform(
    lambda x: x.rolling(window=window_size, min_periods=1).mean()
)

# Pivot and compute online %
pivot = sales_by_type.pivot_table(index=['period_dt', 'time_period_label'], columns='channel_type', values='rolling_avg', fill_value=0).reset_index()
# ensure columns exist
if 'Online' not in pivot.columns: pivot['Online'] = 0
if 'Offline' not in pivot.columns: pivot['Offline'] = 0
pivot['online_pct'] = pivot['Online'] / (pivot['Online'] + pivot['Offline']) * 100

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=pivot['period_dt'],
        y=pivot['Online'],
        name='Online',
        marker_color="#0F0C61",
        text=pivot['Online'].apply(lambda x: f"${x:,.0f}"),
        textposition='auto'
    )
)
fig.add_trace(
    go.Scatter(
        x=pivot['period_dt'],
        y=pivot['online_pct'],
        mode='lines+markers',
        name='Online % of Total Revenue',
        yaxis='y2',
        line=dict(color='#2196F3', width=3)
    )
)
# Competitor benchmark line on secondary axis (kept at 20)
fig.add_shape(
    type="line",
    x0=pivot['period_dt'].min() if not pivot.empty else 0,
    x1=pivot['period_dt'].max() if not pivot.empty else 0,
    y0=20,
    y1=20,
    line=dict(color="red", width=2, dash="dash"),
    xref='x', yref='y2'
)
fig.add_annotation(
    x=1.01,
    y=20,
    xref='paper',
    yref='y2',
    text="Competitor <br> Benchmark",
    showarrow=False,
    font=dict(color="red", size=12),
    xanchor="left",
    yanchor="bottom"
)
fig.update_layout(
    title=f"Online Sales ({agg_option} - Smoothed)",
    title_x=0, title_y=0.95,
    barmode='overlay',
    showlegend=False,
    xaxis=dict(
        title='',
        tickvals=pivot['period_dt'],
        ticktext=pivot['time_period_label'],
        tickangle=45
    ),
    yaxis=dict(title="Online Sales (SGD)"),
    yaxis2=dict(title="Online % of Total Revenue", overlaying='y', side='right', range=[0, 100]),
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1.02),
    margin=dict(t=50, b=40, l=40, r=40)
)
col3.plotly_chart(fig, use_container_width=True)

# -------------------- III. Customers & Orders ---------------------
st.subheader("II. Customers & Orders")

# PREP: Last N periods for tabulations (kept default last_n = 12)
last_n = 12
all_periods = sales['time_period'].drop_duplicates().sort_values()
last_periods_for_tables = all_periods.tail(last_n)

# Determine latest period (by period_dt)
# This line now relies on the corrected definition of latest_period_dt
current_sales = sales[sales['period_dt'] == latest_period_dt].copy()

# Extract product name (kept same logic)
def extract_product_name(sku_str):
    if pd.isna(sku_str):
        return ""
    return re.sub(r'\s*\d+(g|kg|ml|L)?$', '', sku_str).strip()

current_sales['product_name'] = current_sales['sku_description'].apply(extract_product_name)

# Split online/offline
online_sales = current_sales[current_sales['channel_type'] == 'Online'].copy()
offline_sales = current_sales[current_sales['channel_type'] == 'Offline'].copy()

# Age groups for customers (kept labels)
bins = [0, 35, 55, 200]
labels = ['Young (<35)', 'Middle-Aged (35-55)', 'Senior (>50)']
customers['age_group'] = pd.cut(customers['age'], bins=bins, labels=labels, right=False)

# Merge and compute popular age group per product
current_sales_with_age = current_sales.merge(customers[['customer_id', 'age_group']], on='customer_id', how='left')
age_group_sales_current = current_sales_with_age.groupby(['product_name', 'age_group'], as_index=False)['quantity'].sum()
age_group_pivot_current = age_group_sales_current.pivot(index='product_name', columns='age_group', values='quantity').fillna(0)
age_group_pivot_current['Popular Amongst'] = age_group_pivot_current.idxmax(axis=1).astype(str)
age_group_pivot_current = age_group_pivot_current.reset_index()

# Top products and display formatting
total_online_packets = online_sales['quantity'].sum() if not online_sales.empty else 1
total_offline_packets = offline_sales['quantity'].sum() if not offline_sales.empty else 1

top_online_products = online_sales.groupby('product_name', as_index=False)['quantity'].sum().sort_values('quantity', ascending=False).head(5)
top_offline_products = offline_sales.groupby('product_name', as_index=False)['quantity'].sum().sort_values('quantity', ascending=False).head(5)

# -------------------- NEW LOGIC START: TOP 5 LAST PERIOD FLAG --------------------
all_unique_periods = sales['period_dt'].drop_duplicates().sort_values()
if len(all_unique_periods) < 2:
    prior_period_dt = None
    top_online_prior = []
    top_offline_prior = []
else:
    # Identify the Second-to-Last Period
    prior_period_dt = all_unique_periods.iloc[-2]
    
    # Filter Sales for Prior Period and calculate top 5
    prior_sales = sales[sales['period_dt'] == prior_period_dt].copy()
    prior_sales['product_name'] = prior_sales['sku_description'].apply(extract_product_name)

    top_online_prior = prior_sales[prior_sales['channel_type'] == 'Online'].groupby('product_name')['quantity'].sum().nlargest(5).index.tolist()
    top_offline_prior = prior_sales[prior_sales['channel_type'] == 'Offline'].groupby('product_name')['quantity'].sum().nlargest(5).index.tolist()

top_online_products = top_online_products.merge(age_group_pivot_current[['product_name', 'Popular Amongst']], on='product_name', how='left')
top_offline_products = top_offline_products.merge(age_group_pivot_current[['product_name', 'Popular Amongst']], on='product_name', how='left')

# Add the new 'In Top 5 Last Period' flag
top_online_products['In Top 5 Last Period'] = top_online_products['product_name'].apply(
    lambda x: 'Y' if x in top_online_prior else 'N'
)
top_offline_products['In Top 5 Last Period'] = top_offline_products['product_name'].apply(
    lambda x: 'Y' if x in top_offline_prior else 'N'
)


top_online_products['% of Total Orders'] = (top_online_products['quantity'] / max(total_online_packets, 1) * 100).round(1)
top_offline_products['% of Total Orders'] = (top_offline_products['quantity'] / max(total_offline_packets, 1) * 100).round(1)

top_online_products.rename(columns={'product_name': 'Product', 'quantity': 'Number of Packets Bought'}, inplace=True)
top_offline_products.rename(columns={'product_name': 'Product', 'quantity': 'Number of Packets Bought'}, inplace=True)

# String formatting for display (keeps large font size)
top_online_products['Number of Packets Bought'] = top_online_products['Number of Packets Bought'].apply(lambda x: f"{x:,.0f}")
top_online_products['% of Total Orders'] = top_online_products['% of Total Orders'].astype(str) + '%'
top_offline_products['Number of Packets Bought'] = top_offline_products['Number of Packets Bought'].apply(lambda x: f"{x:,.0f}")
top_offline_products['% of Total Orders'] = top_offline_products['% of Total Orders'].astype(str) + '%'

table_col1, table_col2 = st.columns([1, 1])
with table_col1:
    st.write(f"Top 5 Most Ordered Online Products ({latest_period_label})")
    st.dataframe(
        top_online_products[['Product', 'Number of Packets Bought', '% of Total Orders', 'In Top 5 Last Period']]
            .style.set_properties(**{'font-size': '20px'}),
        use_container_width=True,
        hide_index=True
    )
with table_col2:
    st.write(f"Top 5 Most Ordered Offline Products ({latest_period_label})")
    st.dataframe(
        top_offline_products[['Product', 'Number of Packets Bought', '% of Total Orders', 'In Top 5 Last Period']]
            .style.set_properties(**{'font-size': '20px'}),
        use_container_width=True,
        hide_index=True
    )
# -------------------- Row 2 — Charts (Loyalty Pie & Age Distribution) --------------
chart_col1, chart_col2 = st.columns([1, 1.5])

# -------------------- Prepare customer sales --------------------
customer_sales = sales.copy()
customer_sales['loyalty_tier'] = customer_sales['loyalty_tier'].fillna('Unknown')

# Group by loyalty_tier
loyalty_sales = customer_sales[customer_sales['period_dt'] == latest_period_dt].groupby('loyalty_tier', as_index=False)['net_sales_sgd'].sum()

# -------------------- Plot Loyalty Pie Chart --------------------
with chart_col1:
    if not loyalty_sales.empty:
        fig_loyalty_tiers = px.pie(
            loyalty_sales,
            names='loyalty_tier',
            values='net_sales_sgd',
            hole=0.3,
            title=f"Sales by Customer Loyalty Tier ({latest_period_label})",
            color='loyalty_tier',
            color_discrete_map={
                'Bronze': '#CD7F32', 'Silver': '#C0C0C0',
                'Gold': '#FFD700', 'Platinum': '#7a5195',
                'Unknown': '#808080'
            }
        )
        fig_loyalty_tiers.update_traces(textfont=dict(size=18))
        fig_loyalty_tiers.update_layout(
            title_x=0, title_y=0.95, margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_loyalty_tiers, use_container_width=True, height=400)
    else:
        st.warning("No sales data for the selected period.")

# -------------------- Plot ROMS Chart --------------------
with chart_col2:
    # Check if required columns for ROMS calculation are present in the 'events' (events_amended) DataFrame
    if 'marketing_spend_sgd' in events.columns and 'attributed_sales_sgd' in events.columns and 'channel_type' in events.columns:
        # Filter for the latest period (consistent with the Loyalty Pie Chart logic)
        latest_period_dt_events = events['period_dt'].max() if not events['period_dt'].empty else None
        
        if latest_period_dt_events:
            roms_data = events[events['period_dt'] == latest_period_dt_events].copy()
            latest_period_label_roms = latest_period_label
        else:
            roms_data = events.iloc[0:0] # Empty DataFrame
            latest_period_label_roms = ""
        
        # Aggregate Marketing Spend and Attributed Sales by Channel Type
        roms_summary = roms_data.groupby('channel_type', as_index=False).agg(
            total_spend=('marketing_spend_sgd', 'sum'),
            total_sales=('attributed_sales_sgd', 'sum')
        )
        
        # Calculate ROMS: (Total Sales / Total Spend)
        EPSILON = 1e-6 
        roms_summary['roms'] = roms_summary.apply(
            lambda row: row['total_sales'] / (row['total_spend'] + EPSILON) if row['total_spend'] > 0 else 0,
            axis=1
        )
        
        # Format for display
        roms_summary['roms_label'] = roms_summary['roms'].apply(lambda x: f"{x:.1f}x")

        fig_roms = px.bar(
            roms_summary,
            x='channel_type',
            y='roms',
            text='roms_label',
            title=f"Return on Marketing Spend (ROMS) by Channel ({latest_period_label_roms})",
            color='channel_type',
            color_discrete_sequence=px.colors.qualitative.Bold,
            color_discrete_map={'Online': "#200dad"},
            category_orders={'channel_type': ['Online', 'Offline']} # NEW: Enforce fixed order
        )
        
        fig_roms.update_traces(
            textfont=dict(size=12),
            textposition='outside',
            width=0.4
        )
        
        # Add a horizontal dashed red line at y=2.5
        fig_roms.add_shape(
            type="line",
            y0=2.5,
            y1=2.5,
            xref="paper", 
            x0=0,
            x1=1,
            line=dict(
                color="Red",
                width=2,
                dash="dash"
            ),
            name="Threshold (2.5x)"
        )
        
        fig_roms.add_annotation(
            xref="paper", yref="y",
            x=1,
            y=2.5,
            text="Threshold 2.5x",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            yshift=5,
            font=dict(color="Red", size=14),
            bgcolor="rgba(255, 255, 255, 0.7)"
        )
        
        fig_roms.update_layout(
            title_x=0, title_y=0.95,
            xaxis_title='Channel Type',
            yaxis_title='Gross ROMS (x)',
            yaxis_tickformat='.1f',
            yaxis=dict(range=[0, 5.0]), 
            margin=dict(t=30, b=20, l=30, r=30) 
        )
        st.plotly_chart(fig_roms, use_container_width=True, height=300)
    else:
        st.warning("⚠️ events_amended.csv data (loaded into events DF) is missing required columns ('marketing_spend_sgd', 'attributed_sales_sgd', or 'channel_type'). Cannot compute ROMS.")

# -------------------- III. Inventory & Operational Efficiency ------------------- 
st.subheader("III. Inventory & Operational Efficiency")

# -------------------- DSI TIME-SERIES CALCULATION (OPTIMIZED) --------------------
# 1. Prepare grouped sales data (COGS, Min/Max Dates)
dsi_prep = sales.groupby('time_period').agg(
    total_cogs=('cogs', 'sum'),
    min_date=('order_datetime', 'min'),
    max_date=('order_datetime', 'max'),
).reset_index()

# 2. Add period_dt and label for sorting/display
dsi_prep['period_dt'], dsi_prep['time_period_label'] = period_string_to_dt_and_label(dsi_prep['time_period'], agg_option)
dsi_prep = dsi_prep.sort_values('period_dt').reset_index(drop=True)

# 3. Vectorized calculation of total_days
# Handle NaT by falling back to 30 days, then calculate difference
is_valid_date = dsi_prep['min_date'].notna() & dsi_prep['max_date'].notna()
dsi_prep['total_days'] = 30 # Default fallback
dsi_prep.loc[is_valid_date, 'total_days'] = (dsi_prep['max_date'] - dsi_prep['min_date']).dt.days + 1


# 4. Apply DSI Calculation Function in a more controlled, iterative way over the aggregated prep table
dsi_values = []

# This loop is necessary due to the complex, non-vectorizable lookup logic (latest inventory date <= max_date)
for index, row in dsi_prep.iterrows():
    # Only proceed if we have valid dates for the period
    if pd.isna(row['min_date']) or pd.isna(row['max_date']):
        dsi_values.append(0)
        continue
        
    total_cogs = row['total_cogs']
    max_date_period = row['max_date']
    total_days = row['total_days']
    
    average_inventory = 0
    
    # 1. Filter inventory entries that fall *within* the sales period (uses min_date/max_date)
    filtered_inventory = inventory_value[
        (inventory_value['Week Ending Date'] >= row['min_date']) &
        (inventory_value['Week Ending Date'] <= max_date_period)
    ]
    
    if not filtered_inventory.empty:
        average_inventory = filtered_inventory['Average Inventory'].mean()
    else:
        # 2. Find the latest inventory value that occurred on or before the period's end date.
        inventory_up_to_period = inventory_value[
            inventory_value['Week Ending Date'] <= max_date_period
        ].sort_values('Week Ending Date')

        if not inventory_up_to_period.empty:
            average_inventory = inventory_up_to_period['Average Inventory'].iloc[-1]
            
    # FINAL FALLBACK (Preserving original logic)
    if average_inventory == 0 and last_non_zero_inventory > 0:
        average_inventory = last_non_zero_inventory
        
    # 3. Calculate DSI: (Average Inventory / COGS) * Days in Period
    if total_cogs > 0 and average_inventory > 0:
        dsi_value = (average_inventory / total_cogs) * total_days
        dsi_values.append(round(dsi_value))
    else:
        dsi_values.append(0)

dsi_prep['DSI'] = dsi_values
dsi_summary = dsi_prep[dsi_prep['DSI'] > 0].reset_index(drop=True)


if not dsi_summary.empty:
    
    # APPLY THE LIMIT: Take only the last 6 data points for the chart
    dsi_chart_data = dsi_summary.iloc[-6:].copy()

    fig_dsi = px.line(
        dsi_chart_data,
        x='time_period_label', # Use the human-readable label for the x-axis
        y='DSI',
        title=f"DSI - Last 6 {agg_option} Periods",
        markers=True,
    )
    
    # Add target lines for the desired DSI range (30 to 45 days)
    fig_dsi.add_hrect(
        y0=30, y1=45, 
        line_width=0, 
        fillcolor="green", 
        opacity=0.1, 
        annotation_text="Target Range (30-45 Days)",
        annotation_position="right",
        annotation=dict(font_size=12)
    )
    
    fig_dsi.update_layout(
        xaxis_title=f"{agg_option} Period",
        yaxis_title="DSI (Days)",
        hovermode="x unified"
    )

    st.plotly_chart(fig_dsi, use_container_width=True)
else:
    st.info("Insufficient data to calculate DSI across time or no valid COGS/Inventory data found.")
