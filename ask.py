# pages/ask.py

import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from pandasai import Agent
from pandasai.llm import OpenAI as PandasAIOpenAI
from streamlit_mic_recorder import mic_recorder
import tempfile
import base64
from streamlit.components.v1 import html

#########################################
# LOAD ALL DATASETS (CLOUD SAFE)
#########################################

@st.cache_data
def load_all_data_for_chatbot():
    """Loads data from CSVs in repo and returns a dictionary."""

    data = {}

    # FIXED LIST OF FILES TO MATCH YOUR CLOUD DIRECTORY
    file_names = [
        "sales_transactions_expanded_modified.csv",
        "sku_master_expanded.csv",
        "customers.csv",
        "traffic_acquisition.csv",
        "events_amended.csv",
        "diminventoryvalue.csv",
        "dimprojectedrevenue.csv",
        "dimdate.csv",
        "dim_retailinv.csv"        # <-- Added because it's in your repo
    ]

    for file_name in file_names:
        try:
            if not os.path.exists(file_name):
                st.error(f"❌ FILE NOT FOUND: {file_name}")
                data[file_name] = None
                continue

            df = pd.read_csv(file_name)

            # Convert date columns to string so PandasAI doesn't struggle
            for col in df.columns:
                if "date" in col.lower():
                    df[col] = df[col].astype(str)

            # Clean known messy column
            if "line_net_sales_sgd" in df.columns:
                df.rename(columns={"line_net_sales_sgd": "net_sales_sgd"}, inplace=True)

            key = file_name.replace(".csv", "").replace("_", " ").title().replace(" ", "")
            data[key] = df

        except Exception as e:
            st.error(f"⚠️ ERROR LOADING {file_name}: {e}")
            data[file_name] = None

    return data


#########################################
# AUDIO TRANSCRIPTION (NO CHANGE)
#########################################

def transcribe_audio(audio_bytes):
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None, "⚠️ Missing API Key."

    temp_file_path = None
    try:
        client = OpenAI(api_key=api_key)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            temp_file_path = tmp.name

        with open(temp_file_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )

        os.remove(temp_file_path)
        return transcription.text, None

    except Exception as e:
        return None, f"Transcription Error: {e}"


#########################################
# TEXT-TO-SPEECH (NO CHANGE)
#########################################

def synthesize_speech(text_to_speak):
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None, "⚠️ Missing API Key."

    try:
        client = OpenAI(api_key=api_key)
        response = client.audio.speech.create(
            model="tts-1",
            voice="shimmer",
            input=text_to_speak,
            response_format="mp3"
        )
        return response.content, None
    except Exception as e:
        return None, f"TTS Error: {e}"


def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio controls autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    html(audio_html)


#########################################
# MAIN CHATBOT LOGIC — FIXED
#########################################

def prepare_master_dataset(data_frames):
    """
    Creates ONE unified dataset for PandasAI by merging sales and SKU tables.
    This fixes all 'data retrieval' errors on Streamlit Cloud.
    """

    # Main sales table
    sales = data_frames.get("SalesTransactionsexpandedmodified")
    if sales is None:
        st.error("❌ Sales dataset missing. Cannot answer sales questions.")
        return None

    # Merge SKU master if available
    sku = data_frames.get("Skumasterexpanded")
    if isinstance(sku, pd.DataFrame):
        try:
            if "sku_id" in sales.columns and "sku_id" in sku.columns:
                sales = sales.merge(sku, on="sku_id", how="left")
        except Exception as e:
            st.warning(f"⚠️ Could not merge SKU master: {e}")

    return sales


def get_chatbot_response(user_prompt, data_frames, response_mode):

    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ Missing API Key."

    # Build unified dataset
    master_df = prepare_master_dataset(data_frames)

    if master_df is None or master_df.empty:
        return "⚠️ Data could not be loaded or merged."

    # PandasAI LLM
    llm_data_query = PandasAIOpenAI(api_token=api_key, model="gpt-4o")

    # PandasAI Agent
    try:
        agent = Agent(
            master_df,
            config={
                "llm": llm_data_query,
                "enable_error_correction": True,
                "verbose": False,
            }
        )
        internal_insight = agent.chat(user_prompt)

        if isinstance(internal_insight, pd.DataFrame):
            internal_result = "Preview:\n\n" + internal_insight.head(5).to_markdown()
        else:
            internal_result = str(internal_insight)

    except Exception as e:
        internal_result = f"Internal Data Tool Error: {e}"

    # System instructions
    if response_mode == "Info":
        system_message = "You summarize strictly based on internal data."
    else:
        system_message = "You combine internal data insight with business strategy."

    user_message = (
        f"User asked: {user_prompt}\n\n"
        f"Internal Data Insight:\n{internal_result}"
    )

    client = OpenAI(api_key=api_key)
    reply = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3,
        max_tokens=350
    )

    return reply.choices[0].message.content


#########################################
# UI + VOICE + CHAT HISTORY (UNCHANGED)
#########################################

def clear_chat_history():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! What would you like to explore?"}
    ]


def process_user_prompt(prompt, data_frames, response_mode, narration_mode):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner(f"Thinking ({response_mode} Mode)…"):
        reply = get_chatbot_response(prompt, data_frames, response_mode)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)

    if narration_mode:
        audio_bytes, err = synthesize_speech(reply)
        if audio_bytes:
            autoplay_audio(audio_bytes)


#########################################
# STREAMLIT UI
#########################################

st.title("💬 Cashew AI Chatbot")
st.write("Ask questions about sales, customers, or performance.")
st.markdown("---")

with st.spinner("Loading datasets…"):
    all_data_frames = load_all_data_for_chatbot()

if "messages" not in st.session_state:
    clear_chat_history()

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

with st.sidebar:
    st.header("Controls")
    st.button("🗑️ Clear Chat", on_click=clear_chat_history)

    st.subheader("Response Mode")
    st.session_state["response_mode"] = st.radio(
        "", ["Strategy", "Info"], index=0
    )

    st.subheader("Narration")
    st.session_state["narration_mode"] = st.checkbox("Enable Voice Output")

prompt = st.chat_input("Ask me anything…")
if prompt:
    process_user_prompt(
        prompt,
        all_data_frames,
        st.session_state["response_mode"],
        st.session_state["narration_mode"]
    )
