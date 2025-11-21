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

# --- Load Data Function ---
@st.cache_data
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
            if not os.path.exists(file_name):
                st.warning(f"⚠️ File not found: {file_name}")
                data[file_name] = None
                continue

            df = pd.read_csv(file_name)

            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    df[col] = df[col].astype(str)

            if "line_net_sales_sgd" in df.columns:
                df.rename(columns={"line_net_sales_sgd": "net_sales_sgd"}, inplace=True)

            key = file_name.replace(".csv", "").replace("_", " ").title().replace(" ", "")
            data[key] = df

        except Exception as e:
            st.warning(f"⚠️ Error loading {file_name}: {e}")
            data[file_name] = None

    return data


# --- Helper Functions for Audio ---
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
            transcription = client.audio.transcriptions.create(model="whisper-1", file=f)
        os.remove(temp_file_path)
        return transcription.text, None
    except Exception as e:
        return None, f"Whisper API Error: {e}"


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
        return None, f"TTS API Error: {e}"


def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio controls autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    html(audio_html)


# --- Chatbot Core Function ---
def get_chatbot_response(user_prompt, data_frames, response_mode):
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ Missing OpenAI API Key."

    # --- FIX: Force the main sales table first ---
    dfs = []
    sales_df = data_frames.get("SalesTransactionsexpandedmodified")
    if isinstance(sales_df, pd.DataFrame):
        dfs.append(sales_df)

    for k, df in data_frames.items():
        if k != "SalesTransactionsexpandedmodified" and isinstance(df, pd.DataFrame):
            dfs.append(df)

    if not dfs:
        return "⚠️ No valid datasets loaded."

    # PandasAI for data insight
    llm_data_query = PandasAIOpenAI(api_token=api_key, model="gpt-4o")
    try:
        agent = Agent(
            dfs,
            config={"llm": llm_data_query, "enable_error_correction": True, "verbose": False}
        )
        internal_insight = agent.chat(user_prompt)
        if isinstance(internal_insight, pd.DataFrame):
            internal_result_str = "The query returned a table. Preview:\n\n" + internal_insight.head(3).to_markdown()
        else:
            internal_result_str = str(internal_insight)
    except Exception as e:
        internal_result_str = f"Internal Data Tool Error: {e}"

    system_message = (
        "You are a factual summarizer. Provide concise sentences strictly based on internal data."
        if response_mode == "Info"
        else "You are a C-suite Strategic Advisor. Combine internal insight with external context."
    )

    user_message = f"User question: '{user_prompt}'\n\nInternal data insight: {internal_result_str}"

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_message},
                      {"role": "user", "content": user_message}],
            max_tokens=350,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Final LLM Error: {e}"


# --- Clear History ---
def clear_chat_history():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! What would you like to explore?"}
    ]


# --- Process User Prompt ---
def process_user_prompt(prompt, data_frames, response_mode, narration_mode):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.spinner(f"Thinking in {response_mode} Mode…"):
        reply = get_chatbot_response(prompt, data_frames, response_mode)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)
    if narration_mode:
        audio_bytes, error = synthesize_speech(reply)
        if audio_bytes:
            autoplay_audio(audio_bytes)


# --- STREAMLIT UI ---
st.title("💬 Cashew AI Chatbot")
st.markdown("Ask any question about sales, customers, or performance.")
st.markdown("---")

with st.spinner("Loading data…"):
    all_data_frames = load_all_data_for_chatbot()

if "messages" not in st.session_state:
    clear_chat_history()

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

with st.sidebar:
    st.header("Input Controls")
    st.button("🗑️ Clear Chat", on_click=clear_chat_history, use_container_width=True)
    st.markdown("---")

    st.subheader("Output Options")
    st.session_state["narration_mode"] = st.checkbox("🎧 Enable Narration (Auto-Play)",
                                                    value=st.session_state.get("narration_mode", False))
    st.session_state["response_mode"] = st.radio(
        "Choose the output style:", ("Strategy", "Info"),
        index=0 if st.session_state.get("response_mode", "Strategy") == "Strategy" else 1
    )

    st.markdown("---")

    st.subheader("Input Mode")
    input_mode = st.session_state.get("input_mode", "Text")
    if input_mode == "Text":
        new_mode = "Voice"
        label = "🎙️ Switch to Voice Mode"
    else:
        new_mode = "Text"
        label = "⌨️ Switch to Text Mode"

    if st.button(label, use_container_width=True):
        st.session_state["input_mode"] = new_mode
        st.rerun()

    if st.session_state.get("input_mode", "Text") == "Voice":
        st.markdown("### Voice Input Mode 🎙️")
        audio_data = mic_recorder(start_prompt="Click to Record", stop_prompt="Recording... Click to Stop", key="mic_recorder", format="wav")
        if audio_data:
            st.info("Transcribing audio...")
            voice_prompt, transcription_error = transcribe_audio(audio_data['bytes'])
            if transcription_error:
                st.error(transcription_error)
                st.session_state["input_mode"] = "Text"
                st.rerun()
            else:
                st.success(f"Transcription: **{voice_prompt}**")
                st.session_state["pending_prompt"] = voice_prompt
                st.session_state["pending_response_mode"] = st.session_state["response_mode"]
                st.session_state["pending_narration_mode"] = st.session_state["narration_mode"]
                st.session_state["input_mode"] = "Text"
                st.rerun()
    else:
        st.markdown("### Text Input Mode ⌨️")

# --- Handle pending prompt ---
if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
    prompt = st.session_state.pop("pending_prompt")
    response_mode = st.session_state.pop("pending_response_mode", "Strategy")
    narration_mode = st.session_state.pop("pending_narration_mode", False)
    process_user_prompt(prompt, all_data_frames, response_mode, narration_mode)

# --- Chat input ---
if st.session_state.get("input_mode", "Text"):
    if prompt := st.chat_input("Enter your question..."):
        process_user_prompt(prompt, all_data_frames, st.session_state["response_mode"], st.session_state["narration_mode"])
