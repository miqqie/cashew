# pages/ask.py

import os
# ... other imports

# ---------------------------
# Set the base path to the directory where this script is located (pages/)
# This uses the Python magic variable __file__ to get the script's location
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) 
# Note: os.path.dirname(__file__) might be sufficient, but abspath makes it robust.

# ---------------------------
# Load CSVs (with Caching) - Use this entire updated function
# ---------------------------
@st.cache_data(show_spinner="Loading all datasets...")
def load_all_data_for_chatbot():
    data = {}
    
    # ... file_names list remains the same ...
    file_names = [
        "sales_transactions_expanded_modified.csv",
        "sku_master_expanded.csv",
        "customers.csv",
        # ... all other files
        "dim_retailinv.csv" 
    ]

    for file_name in file_names:
        # Create the full path: /path/to/repo/pages/filename.csv
        full_path = os.path.join(DATA_DIR, file_name)
        
        try:
            # Check for file existence to provide better debugging on Streamlit Cloud
            if not os.path.exists(full_path):
                 st.warning(f"⚠️ File not found. Looked for: {full_path}")
                 continue # Skip this file if not found
                 
            df = pd.read_csv(full_path) # Use the full, absolute path
            
            # ... rest of your normalization code ...
            
            key = file_name.replace(".csv", "").replace("_", " ").title().replace(" ", "")
            data[key] = df
            
        except Exception as e:
            st.error(f"❌ Failed to load {file_name} at {full_path}: {e}")
            data[file_name] = None
            
    return data

# ... rest of your script
