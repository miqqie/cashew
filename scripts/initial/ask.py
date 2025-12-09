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
from streamlit.components.v1 import html # <--- NEW: Import Streamlit HTML component

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
        "dimbroadinvdist.csv",
        "dimdemandforecast.csv",
        "dimdeliverylocation.csv",
        "diminventoryvalue.csv",
        "dimprojectedrevenue.csv",
        "dimdate.csv"
    ]
    
    for file_name in file_names:
        try:
            df = pd.read_csv(file_name)

            # Clean datetime columns
            # NOTE: We keep it as string here, but conversion is necessary for calculation later.
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


# --- Helper Function for Transcription ---
def transcribe_audio(audio_bytes):
    """
    Transcribes audio bytes into text using OpenAI's Whisper model (via API).
    Includes fix for Windows Permission Denied error using tempfile.
    """
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None, "⚠️ Missing OpenAI API Key for transcription."

    temp_file_path = None
    try:
        client = OpenAI(api_key=api_key)
        
        # 1. Create and write to the file, setting delete=False to manage file closing manually
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            temp_file_path = tmp_file.name # Store the path
        
        # 2. Read and send the file to OpenAI Whisper API
        with open(temp_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                language="en"
            )
        
        # 3. Clean up the temporary file
        os.remove(temp_file_path)

        return transcription.text, None
            
    except Exception as e:
        # 4. Ensure cleanup if the error occurred before the final remove
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as cleanup_e:
                print(f"Cleanup failed for {temp_file_path}: {cleanup_e}") 
                
        return None, f"Whisper API Error: {e}"

# --- Helper Function for TTS Synthesis ---
def synthesize_speech(text_to_speak):
    """
    Synthesizes text into speech using OpenAI's TTS API.
    Returns audio bytes in MP3 format.
    """
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None, "⚠️ Missing OpenAI API Key for speech synthesis."

    try:
        client = OpenAI(api_key=api_key)
        
        # Call the TTS API
        response = client.audio.speech.create(
            model="tts-1",
            voice="shimmer",  # A pleasant, clear voice
            input=text_to_speak,
            response_format="mp3"
        )
        
        # The response is a stream, we read it fully into bytes
        audio_bytes = response.content
        return audio_bytes, None
            
    except Exception as e:
        return None, f"TTS API Error: {e}"

# --- NEW: Function to display audio with autoplay using HTML ---
def autoplay_audio(audio_bytes):
    """
    Renders an HTML audio player with the autoplay attribute set.
    """
    # Convert audio bytes to base64 string
    b64 = base64.b64encode(audio_bytes).decode()
    
    # Construct the HTML for the audio player
    audio_html = f"""
    <audio controls autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """
    # Use Streamlit's HTML component to inject the audio player
    html(audio_html)


# --- Chatbot Core Function ---
def get_chatbot_response(user_prompt, data_frames, response_mode):
    """Runs PandasAI + OpenAI synthesis based on the selected mode."""
    
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

    # --- Mode-Specific Response Generation ---

    # If Info Mode, just return the internal data insight
    if response_mode == "Info":
        # Check if the insight is just an error or no-op
        if internal_result_str.startswith(("Internal Data Tool Error", "No internal data insight found")):
            return f"Info Mode: Failed to retrieve data. {internal_result_str}"
        
        # 2. Strategy synthesis (OpenAI) for factual summary
        system_message = (
            "You are a Factual Data Summarizer. Given a user question and an internal data insight, "
            "provide a concise, to-the-point summary of the data in sentence format. "
            "Do not add strategic commentary, external context, or conversational filler. "
            "Focus strictly on the facts and figures derived from the internal data."
        )

        user_message = (
            f"User question: '{user_prompt}'\n\n"
            f"Internal data insight: {internal_result_str}\n\n"
            "Produce a concise, factual answer based *only* on the internal data."
        )

    # If Strategy Mode (current default behavior)
    else: # response_mode == "Strategy"
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
    
    # 3. Final LLM Call
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
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
                "Hello! I’m your AI Advisor. "
                "What would you like to explore?"
            )
        }
    ]

# --- Function to handle prompt processing (text or voice) ---
def process_user_prompt(prompt, data_frames, response_mode, narration_mode):
    """Handles the common logic for processing a user's prompt."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner(f"Thinking in **{response_mode} Mode**…"):
        reply = get_chatbot_response(prompt, data_frames, response_mode)

    # 1. Store and display the text response
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)
    
    # 2. Add Narration if enabled
    if narration_mode:
        with st.spinner("Synthesizing audio response..."):
            audio_bytes, error = synthesize_speech(reply)
            
            if error:
                st.error(error)
            elif audio_bytes:
                # Use custom function for autoplay
                autoplay_audio(audio_bytes) # <--- MODIFIED CALL

# --- Streamlit UI ---
st.title("💬 Cashew AI Chatbot")
st.markdown("Ask any question about sales, customers, inventory or performance. I’ll analyse internal data and provide strategic insights.")
st.markdown("---")

# Load data
with st.spinner("Loading data…"):
    all_data_frames = load_all_data_for_chatbot()

# Chat history
if "messages" not in st.session_state:
    clear_chat_history() # Initialize history

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# ===============================================
# --- INPUT AND MODE MANAGEMENT (IN SIDEBAR) ---
# ===============================================

with st.sidebar:
    st.header("Input Controls")
    
    # Add the Clear Chat button to the sidebar
    st.button("🗑️ Clear Chat", on_click=clear_chat_history, use_container_width=True)
    st.markdown("---")

    st.subheader("Output Options")
    
    # Initialize the *Narration* mode (Off/On)
    if "narration_mode" not in st.session_state:
        st.session_state["narration_mode"] = False

    # Add the Narration checkbox
    st.session_state["narration_mode"] = st.checkbox(
        "🎧 Enable Narration (Auto-Play)", # <--- MODIFIED LABEL
        value=st.session_state["narration_mode"],
        key="narration_mode_checkbox",
        help="Reads out the assistant's final response automatically after a question."
    )
    
    st.markdown("---")
    
    # Initialize the *Response* mode (Strategy/Info)
    if "response_mode" not in st.session_state:
        st.session_state["response_mode"] = "Strategy" 
    
    # Add the Option Button (radio) for Response Mode
    st.subheader("Response Mode")
    st.session_state["response_mode"] = st.radio(
        "Choose the output style:",
        ("Strategy", "Info"),
        index=0 if st.session_state["response_mode"] == "Strategy" else 1,
        key="response_mode_radio",
        help="Strategy: C-suite advice with external context. Info: Factual, data-driven sentences only."
    )
    
    st.markdown("---")

    # Initialize the *Input* mode (Text/Voice)
    if "input_mode" not in st.session_state:
        st.session_state["input_mode"] = "Text" 

    # Create a toggle button to switch modes
    if st.session_state["input_mode"] == "Text":
        new_mode = "Voice"
        button_label = "🎙️ Switch to Voice Mode"
    else:
        new_mode = "Text"
        button_label = "⌨️ Switch to Text Mode"

    # Use a unique key for the button
    if st.button(button_label, use_container_width=True, key="mode_switch"):
        st.session_state["input_mode"] = new_mode
        st.rerun() 
        
    st.markdown("---")

    voice_prompt = None
    transcription_error = None

    # Create a dedicated placeholder for status messages
    status_placeholder = st.empty()

    if st.session_state["input_mode"] == "Voice":
        status_placeholder.markdown("### Voice Input Mode 🎙️")
        
        # 1. Render the mic recorder
        audio_data = mic_recorder(
            start_prompt="Click to Record",
            stop_prompt="Recording... Click to Stop",
            key="mic_recorder", 
            format="wav"
        )

        # 2. Process voice input if available
        if audio_data:
            status_placeholder.info("Transcribing audio...") 
            voice_prompt, transcription_error = transcribe_audio(audio_data['bytes'])
            
            if transcription_error:
                status_placeholder.error(transcription_error) 
                st.session_state["input_mode"] = "Text" 
                status_placeholder.warning("Switching back to Text Mode due to transcription error.")
                
            elif voice_prompt:
                status_placeholder.success(f"Transcription: **{voice_prompt}**")
                
                # Set the transcribed text and the *current response mode* as the prompt to be processed
                st.session_state["pending_prompt"] = voice_prompt
                st.session_state["pending_response_mode"] = st.session_state["response_mode"]
                st.session_state["pending_narration_mode"] = st.session_state["narration_mode"]
                
                # Auto-switch back to Text mode after successful voice input
                st.session_state["input_mode"] = "Text" 
                st.rerun() 

    elif st.session_state["input_mode"] == "Text":
        status_placeholder.markdown("### Text Input Mode ⌨️")
    
# ===============================================
# --- END INPUT AND MODE MANAGEMENT IN SIDEBAR ---
# ===============================================


# --- Final Prompt Processing (handles successful voice transcription) ---

if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
    prompt = st.session_state.pop("pending_prompt")
    response_mode = st.session_state.pop("pending_response_mode", "Strategy") # Default to Strategy
    narration_mode = st.session_state.pop("pending_narration_mode", False)
    process_user_prompt(prompt, all_data_frames, response_mode, narration_mode)

# 2. Process text input (st.chat_input should be last to be sticky at the bottom)
if st.session_state["input_mode"] == "Text":
    # Get the current response mode for processing
    current_response_mode = st.session_state.get("response_mode", "Strategy")
    current_narration_mode = st.session_state.get("narration_mode", False)
    if prompt := st.chat_input("Enter your question..."):
        process_user_prompt(prompt, all_data_frames, current_response_mode, current_narration_mode)


