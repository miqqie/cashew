# ask.py (in the root directory)

import os
import streamlit as st
import pandas as pd
# ... other imports

# ---------------------------
# Define the base directory as the location of the script itself
# This is the most reliable way to find files committed to the same folder.
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) 
# If ask.py is in the root, DATA_DIR will be /app/your-repo/
# ---------------------------

# Load CSVs (with Caching) - Use this entire updated function
@st.cache_data(show_spinner="Loading all datasets...")
def load_all_data_for_chatbot():
    data = {}
    file_names = [
        "sales_transactions_expanded_modified.csv",
        "sku_master_expanded.csv",
        # ... all other files
        "dim_retailinv.csv"
    ]

    for file_name in file_names:
        # Create the full path: /app/your-repo/filename.csv
        full_path = os.path.join(DATA_DIR, file_name)
        
        try:
            # Added a simple check to confirm files exist
            if not os.path.exists(full_path):
                 st.warning(f"⚠️ File not found. Looked for: {full_path}")
                 continue 
                 
            df = pd.read_csv(full_path) # Use the full, absolute path
            
            # ... rest of your data normalization and key assignment code ...
            
            key = file_name.replace(".csv", "").replace("_", " ").title().replace(" ", "")
            data[key] = df
            
        except Exception as e:
            st.error(f"❌ Failed to load {file_name} at {full_path}: {e}")
            data[file_name] = None
            
    return data

# ... rest of your script
