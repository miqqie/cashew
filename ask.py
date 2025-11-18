# pages/ask.py
import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from pandasai import Agent
from pandasai.llm import OpenAI as PandasAIOpenAI

# --- Page Configuration ---
st.set_page_config(page_title="Ask a Question", layout="wide")

# --- Load Data Function ---
@st.cache_data
def load_all_data_for_chatbot():
    """Loads all CSV files and returns them as a dictionary of DataFrames."""
    data = {}
    
    file_names = [
        "sales_transactions_expanded_modified.csv",
        "sku_master_expanded.csv",
        "customers.csv",
        "traffic_acquisition.csv",
        "events_amended.csv",
        "ecommerce_purchases.csv",
        "diminventoryvalue.csv",
        "dimprojectedrevenue.csv",
        "dimdate.csv"
    ]
    
    for file_name in file_names:
        try:
            df = pd.read_csv(file_name)

            # Clean datetime columns
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    df[col] = df[col].astype(str)

            # Rename known messy column
            if "line_net_sales_sgd" in df.columns:
                df.rename(columns={"line_net_sales_sgd": "net_sales_sgd"}, inplace=True)

            df_name = file_name.replace('.csv', '').replace('_', ' ').title().replace(' ', '')
            data[df_name] = df

        except Exception:
            # If any file fails, store None to keep loading process robust
            data[file_name] = None

    return data


# --- Chatbot Core Function ---
def get_chatbot_response(user_prompt, data_frames):
    """Runs PandasAI + OpenAI synthesis."""
    
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ Missing OpenAI API Key. Please add it to Streamlit secrets."

    # Filter: keep ONLY valid DataFrames
    valid_dfs = [df for df in data_frames.values() if isinstance(df, pd.DataFrame)]

    if not valid_dfs:
        return "⚠️ No valid datasets loaded. Please check your CSV files."

    # 1. PandasAI agent (multi-DF mode)
    llm_data_query = PandasAIOpenAI(api_token=api_key, model="gpt-4o")

    try:
        agent = Agent(
            valid_dfs,
            config={
                "llm": llm_data_query,
                "enable_error_correction": True,
                "verbose": False,
            }
        )

        internal_insight = agent.chat(user_prompt)

        if isinstance(internal_insight, pd.DataFrame):
            internal_result_str = (
                "The query returned a table. Here is a preview:\n\n" +
                internal_insight.head(3).to_markdown()
            )
        else:
            internal_result_str = str(internal_insight)

    except Exception as e:
        internal_result_str = f"Internal Data Tool Error: {e}"

    # 2. Strategy synthesis (OpenAI)
    system_message = (
        "You are a C-suite Strategic Advisor for a Singaporean FMCG snack company. "
        "Combine internal data insights with external market forces and give concise, "
        "actionable strategic recommendations. Limit to under 250 words."
    )

    user_message = (
        f"User question: '{user_prompt}'\n\n"
        f"Internal data insight: {internal_result_str}\n\n"
        "Produce a concise strategic answer for senior leadership."
    )

    try:
        # NOTE: Model name was fixed to a known-good model (e.g., gpt-4) 
        # as 'gpt-4.1' is not a standard OpenAI model.
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=350,
            temperature=0.3
        )
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Final LLM Error: {e}"

# --- Helper Function for Chat History Clearing ---
def clear_chat_history():
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hello! I’m your C-suite AI Advisor. I have access to Sales, Inventory, "
                "Customers, Traffic, Events and more. What would you like to explore?"
            )
        }
    ]

# --- Streamlit UI ---
st.title("💬 Cashew AI Chatbot: C-suite Advisor")
st.markdown("Ask any question about sales, customers, inventory or performance. I’ll analyse internal data and provide strategic insights.")
st.markdown("---")

# Add the Clear Chat button to the sidebar
st.sidebar.button("🗑️ Clear Chat", on_click=clear_chat_history)

# Load data
with st.spinner("Loading data…"):
    all_data_frames = load_all_data_for_chatbot()

# Chat history
if "messages" not in st.session_state:
    clear_chat_history() # Initialize history

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if prompt := st.chat_input("Enter your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Thinking…"):
        reply = get_chatbot_response(prompt, all_data_frames)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)