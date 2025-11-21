# --- Load Data Function (no caching) ---
def load_all_data_for_chatbot():
    data = {}
    file_names = [
        "sales_transactions_expanded_modified.csv",
        "sku_master_expanded.csv",
        "customers.csv",
        "traffic_acquisition.csv",
        "events_amended.csv",
        "diminventoryvalue.csv",
        "dimprojectedrevenue.csv",
        "dimdate.csv",
        "dim_retailinv.csv"
    ]

    for file_name in file_names:
        try:
            df = pd.read_csv(file_name)
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    df[col] = df[col].astype(str)
            if "line_net_sales_sgd" in df.columns:
                df.rename(columns={"line_net_sales_sgd": "net_sales_sgd"}, inplace=True)
            key = file_name.replace(".csv", "").replace("_", " ").title().replace(" ", "")
            data[key] = df
        except Exception as e:
            st.warning(f"⚠️ Failed to load {file_name}: {e}")
            data[file_name] = None
    return data


# --- Before passing to PandasAI ---
valid_dfs = []
for k, df in all_data_frames.items():
    if isinstance(df, pd.DataFrame) and not df.empty:
        valid_dfs.append(df)
    else:
        st.warning(f"⚠️ {k} is empty or invalid")

# Debug: show DataFrame shapes
for i, df in enumerate(valid_dfs):
    st.write(f"DataFrame {i}: shape={df.shape}, columns={list(df.columns)}")

agent = Agent(valid_dfs, config={"llm": llm_data_query, "enable_error_correction": True, "verbose": True})
