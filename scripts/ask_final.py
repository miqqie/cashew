# pages/ask.py
import os
import sys
import streamlit as st
import pandas as pd
from openai import OpenAI
from pandasai import Agent
from pandasai.llm import OpenAI as PandasAIOpenAI
from streamlit_mic_recorder import mic_recorder
import tempfile
import base64
from streamlit.components.v1 import html
from json import dumps as json_dumps # Added for sequential playback
import re
from concurrent.futures import ThreadPoolExecutor

def show_chatbot():
    # 1️⃣ Force UTF-8 encoding for stdout/stderr to avoid Windows Unicode errors
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

    # --- Page Configuration ---
    #st.set_page_config(page_title="Ask a Question", layout="wide")

    # --- Load Data Function ---
    @st.cache_data
    def load_all_data_for_chatbot():
        """Loads all CSV files and returns them as a dictionary of DataFrames."""
        data = {}

        file_names = [
            "customers.csv",
            "dimdeliverylocation.csv",
            "sales_transactions_expanded_modified.csv",
            "sku_master_expanded.csv",
            "traffic_acquisition.csv",
            "events_amended.csv",
            "dimbroadinvdist.csv",
            "dimdemandforecast.csv",            
            "diminventoryvalue.csv",
            
            "dimdate.csv"
        ]

        for file_name in file_names:
            try:
                df = pd.read_csv(file_name)

                
                                
                # Clean datetime columns
                # NOTE: Conversion is now done here to improve PandasAI performance.
                for col in df.columns:
                    if "date" in col.lower() or "time" in col.lower():
                        df[col] = pd.to_datetime(df[col], errors='coerce')

                # 2. CREATE A DEDICATED INTEGER YEAR COLUMN
                if 'order_datetime' in df.columns:
    
                # 💥 FIX: Force conversion of the specific column one more time.
                # This prevents issues if the general cleaning loop missed a format error.
                    df['order_datetime'] = pd.to_datetime(df['order_datetime'], errors='coerce')
    
                # Now that we're certain it's datetime, extract the year.
                # .dt.year will return NaN for any 'NaT' values created by 'coerce'.
                    df['order_year'] = df['order_datetime'].dt.year.astype('Int64')
                                    
                # Rename known messy column
                if "line_net_sales_sgd" in df.columns:
                    df.rename(columns={"line_net_sales_sgd": "revenue"}, inplace=True)
                    # ✅ FIX: Corrected column reference to 'net_sales_sgd' to prevent the Dataframe loading error.
                    df["revenue"] = df["revenue"].round(2) 
                
                               
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
        Includes fix for Windows Permission denied error using tempfile.
        """
        api_key = st.secrets.get("ext_api", {}).get("open_api_key")
        if not api_key:
            api_key = os.environ.get("EXT_API_OPEN_API_KEY")

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

    # --- Helper Function for TTS Synthesis (Called Concurrently) ---
    def synthesize_speech(text_to_speak, client):
        """
        Synthesizes text into speech using OpenAI's TTS API.
        Returns audio bytes in OPUS format for improved streaming latency.
        """
        try:

            response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=text_to_speak,
                response_format="opus"
            )

            # The response is a stream, we read it fully into bytes
            audio_bytes = response.content
            return audio_bytes, None

        except Exception as e:
            # Return error tuple for concurrent processing
            return None, f"TTS API Error: {e}"

    # --- Function to display audio with sequential autoplay using HTML/JS ---
    def play_sequential_audio(audio_chunks_b64):
        """
        Renders an HTML/JS player to sequentially play an array of base64 audio chunks.
        This bypasses slow pydub merging and includes a custom Pause/Resume control.
        """
        # Convert Python list to JSON string for JavaScript
        json_chunks = json_dumps(audio_chunks_b64)

        # JavaScript to queue and play audio elements, now with controls
        js_code = f"""
        <div id="audio_controls_container" style="display:flex; align-items:center; padding: 5px 0;">
            <button id="pause_resume_btn" style="
                background-color: #4CAF50; /* Green */
                color: white;
                padding: 8px 15px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                transition: background-color 0.3s ease;
            " onmouseover="this.style.backgroundColor='#45a049'" onmouseout="this.style.backgroundColor='#4CAF50'">⏸️ Pause</button>
        </div>
        <script>
        (function() {{
            const audioChunksB64 = {json_chunks};
            let currentChunkIndex = 0;
            let currentAudioElement = null;
            let isPlaying = false;

            const btn = document.getElementById('pause_resume_btn');

            // --- Core Playback Logic ---
            function playNextChunk() {{
                if (currentChunkIndex >= audioChunksB64.length) {{
                    // Playback complete logic
                    btn.textContent = "▶️ Replay";
                    btn.style.backgroundColor = '#2196F3'; // Blue for Replay
                    btn.onmouseover = function(){{this.style.backgroundColor='#0b7dda'}};
                    btn.onmouseout = function(){{this.style.backgroundColor='#2196F3'}};
                    isPlaying = false;
                    currentAudioElement = null;
                    return; // Finished
                }}

                const b64 = audioChunksB64[currentChunkIndex];
                const newAudioElement = new Audio("data:audio/ogg;base64," + b64);
                currentAudioElement = newAudioElement;

                newAudioElement.onended = () => {{
                    currentAudioElement = null;
                    currentChunkIndex++;
                    if (isPlaying) {{
                        playNextChunk();
                    }}
                }};
                
                // 🐛 FIX: Wait for 'canplaythrough' event to ensure the audio is ready before playing.
                const attemptPlay = () => {{
                    if (isPlaying) {{
                        newAudioElement.play()
                            .then(() => {{
                                // Play succeeded
                            }})
                            .catch(e => console.error("Autoplay failed:", e));
                    }}
                }};

                newAudioElement.addEventListener('canplaythrough', attemptPlay);
                
                // Fallback: If 'canplaythrough' doesn't fire fast enough, or if it's already ready
                if (newAudioElement.readyState >= 4) {{ // HTMLMediaElement.HAVE_ENOUGH_DATA
                    attemptPlay();
                }}
                
            }}

            // --- Control Handler ---
            function togglePlayback() {{
                if (!currentAudioElement && currentChunkIndex === audioChunksB64.length) {{
                    // Replay logic
                    currentChunkIndex = 0;
                    isPlaying = true;
                    btn.textContent = "⏸️ Pause";
                    btn.style.backgroundColor = '#4CAF50'; // Green for Pause
                    btn.onmouseover = function(){{this.style.backgroundColor='#45a049'}};
                    btn.onmouseout = function(){{this.style.backgroundColor='#4CAF50'}};
                    playNextChunk();
                    return;
                }}

                isPlaying = !isPlaying;

                if (isPlaying) {{
                    btn.textContent = "⏸️ Pause";
                    btn.style.backgroundColor = '#4CAF50'; // Green for Pause
                    btn.onmouseover = function(){{this.style.backgroundColor='#45a049'}};
                    btn.onmouseout = function(){{this.style.backgroundColor='#4CAF50'}};
                    if (currentAudioElement && currentAudioElement.paused) {{
                        // Resume current chunk
                        currentAudioElement.play().catch(e => console.error("Resume failed:", e));
                    }} else if (!currentAudioElement) {{
                        // Start the next chunk if the current one ended during pause
                        playNextChunk();
                    }}
                }} else {{
                    // Pause
                    if (currentAudioElement) {{
                        currentAudioElement.pause();
                    }}
                    btn.textContent = "▶️ Resume";
                    btn.style.backgroundColor = '#F44336'; // Red for Resume
                    btn.onmouseover = function(){{this.style.backgroundColor='#da190b'}};
                    btn.onmouseout = function(){{this.style.backgroundColor='#F44336'}};
                }}
            }}

            // --- Initialization ---
            if (btn) {{
                btn.addEventListener('click', togglePlayback);
            }}
            
            // ✅ Kept at 200ms delay to initial playback call for better autoplay compatibility
            isPlaying = true;
            setTimeout(playNextChunk, 200); 
        }})(); // End of IIFE
        </script>
        """
        html(js_code, height=45)

    # --- Helper Function for Auto-Scrolling ---
    def scroll_to_bottom():
        """Injects JavaScript to force the main Streamlit container to scroll to the bottom."""
        js_code = """
        <script>
            // Scroll the main scrollable element of the Streamlit app. 
            // Use a slight delay to ensure the content has finished rendering.
            setTimeout(function() {
                // Target the scrollable content container
                var mainContent = window.parent.document.querySelector('section.main');
                if (mainContent) {
                    // ✅ FIX: Increased delay to 300ms to ensure the audio button is rendered and measured 
                    // by the DOM before calculating scrollHeight.
                    mainContent.scrollTop = mainContent.scrollHeight;
                }
            }, 300); 
        </script>
        """
        # Render the script using the imported 'html' component
        # We set height=0 to make the component invisible and not take up space.
        html(js_code, height=0, width=0)

    # 🆕 NEW HELPER FUNCTION: To format unformatted large numbers
    def format_large_numbers_for_tts(text):
        """
        Finds sequences of 5 or more digits and inserts thousands separators (commas)
        to ensure the TTS engine reads them as cardinal numbers, and for visual display.
        Example: 'The sales figure is 127626.88' -> 'The sales figure is 127,626.88'
        """
        
        # ✅ FIX: Updated regex to ignore currency symbols but prevent re-formatting already comma-separated numbers.
        # Pattern: Look for 5+ digits, optionally followed by a decimal point and more digits.
        # Negative lookbehind: Only match if the number is NOT immediately preceded by a comma.
        pattern = r'(?<!,)(\d{5,}(?:\.\d+)?)'
        
        def replacer(match):
            number_str = match.group(1)
            
            # Split number into integer and decimal parts
            if '.' in number_str:
                integer_part, decimal_part = number_str.split('.', 1)
            else:
                integer_part = number_str
                decimal_part = None
                
            # Insert commas into the integer part (from right to left)
            formatted_integer = []
            n = len(integer_part)
            for i in range(n):
                digit = integer_part[n - 1 - i]
                if i > 0 and i % 3 == 0:
                    formatted_integer.append(',')
                formatted_integer.append(digit)
            
            # Reverse and join the formatted integer part
            formatted_integer = "".join(reversed(formatted_integer))
            
            # Recombine with the decimal part if it existed
            if decimal_part is not None:
                # We enforce exactly two decimal places here for better reading/display
                try:
                    # Convert to float and back to string formatted to 2 decimal places
                    decimal_part = f"{float(integer_part + '.' + decimal_part):.2f}".split('.')[1]
                    return f"{formatted_integer}.{decimal_part}"
                except ValueError:
                    # Fallback if conversion fails
                    return f"{formatted_integer}.{decimal_part}"
            else:
                return formatted_integer
        
        # Apply the replacer function to the text
        return re.sub(pattern, replacer, text)

    # 🆕 NEW HELPER FUNCTION: Query Exclusion Filter
    def is_unsupported_query(prompt):
        """
        Checks if the user's prompt contains keywords related to unsupported 
        (non-existent) data.
        """
        unsupported_keywords = [
            'employee', 'employees', 'staff','salary', 'salaries', 'wage', 'exports', 'benefits'
            'wages', 'headcount', 'hr', 'human resources', 'worker', 'workers', 'address', 'compensation'
        ]
        
        # Convert prompt to lowercase for case-insensitive matching
        lower_prompt = prompt.lower()
        
        # Check if any keyword is present in the prompt
        for keyword in unsupported_keywords:
            # Use space padding or regex word boundary if necessary, but simple `in` 
            # is often sufficient and safer if keywords are specific.
            if keyword in lower_prompt:
                # Simple check for 'employee' vs 'customer_employee_id' (which doesn't exist 
                # but demonstrates complexity). The simple `in` is sufficient here.
                return True
        
        return False
    
    # --- Chatbot Core Function ---
    def get_chatbot_response(user_prompt, data_frames, response_mode):
        """Runs PandasAI + OpenAI synthesis based on the selected mode. Returns an iterable/generator."""

        api_key = st.secrets.get("ext_api", {}).get("open_api_key")
        if not api_key:
            api_key = os.environ.get("EXT_API_OPEN_API_KEY")

        if not api_key:
            yield "⚠️ Missing OpenAI API Key. Please add it to Streamlit secrets."
            return
        # 1. IMMEDIATE EXIT FOR UNSUPPORTED QUERIES
        if is_unsupported_query(user_prompt):
            # This message is designed to be treated as a final, complete answer
            yield "I'm sorry, this data is not available in the current dataset."
            return
        
        # Filter: keep ONLY valid DataFrames
        valid_dfs = [df for df in data_frames.values() if isinstance(df, pd.DataFrame)]

        if not valid_dfs:
            yield "⚠️ No valid datasets loaded. Please check your CSV files."
            return

    
        llm_data_query = PandasAIOpenAI(api_token=api_key, model="gpt-3.5-turbo")
        
        
        try:
            agent = Agent(
                valid_dfs,
                config={
                    "llm": llm_data_query,
                    "enable_error_correction": True,
                    "rows_limit": 2,
                    "verbose": False,
                    "custom_instruction": (
         "You are a Senior Data Analyst. Your primary goal is to generate the correct Python code. "
            "Follow these rules: "
            "1. You MUST NOT infer or assume the user meant something different from what they said. If the dataset does not contain the exact group or field the user requests, reply: The dataset does not include that information. Never replace the requested category with a different one."
            "2. **Formatting:** All currency figures (SGD, net_sales_sgd, etc.) must be reported in their original, unscaled units (e.g., '127,600' or '127.6 Thousand'), or rounded to 2 decimal places. NEVER convert or label figures as 'Million' or 'million' or 'K' unless explicitly asked to do so."
           
                    )
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

        # Determine the model for final synthesis based on mode
        if response_mode == "Info":
            # Use gpt-3.5-turbo for maximum speed on factual summarization
            final_llm_model = "gpt-3.5-turbo"

            if internal_result_str.startswith(("Internal Data Tool Error", "No internal data insight found")):
                yield f"Info Mode: Failed to retrieve data. {internal_result_str}"
                return

            system_message = (
                "You are a Factual Data Summarizer. Given a user question and an internal data insight, "
                "provide a concise, to-the-point summary of the data in sentence format. Sales figures no more than 2 decimal places"
                "Do much add strategic commentary, external context, or conversational filler. "
                "Focus strictly on the facts and figures derived from the internal data."
            )

            user_message = (
                f"User question: '{user_prompt}'\n\n"
                f"Internal data insight: {internal_result_str}\n\n"
                "Produce a concise, factual answer based *only* on the internal data."
            )

        # If Strategy Mode (current default behavior)
        else: # response_mode == "Strategy"

            final_llm_model = "gpt-4o"

            system_message = (
                "You are a C-suite Strategic Advisor for a Singaporean FMCG snack company. "
                "Combine internal data insights with external market forces and give concise, "
                "actionable strategic recommendations. Limit to under 120 words and keep to 6 sentences."
            )

            user_message = (
                f"User question: '{user_prompt}'\n\n"
                f"Internal data insight: {internal_result_str}\n\n"
                "Produce a concise strategic answer for senior leadership."
            )

        # 3. Final LLM Call - MODIFIED FOR STREAMING
        try:
            client = OpenAI(api_key=api_key)
            stream = client.chat.completions.create(
                model=final_llm_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=180,
                temperature=0.3,
                stream=True
            )

            # Yield chunks from the stream
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content

        except Exception as e:
            yield f"Final LLM Error: {e}"


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
        """Handles the common logic for processing a user's prompt with concurrent TTS and merging."""

        api_key = st.secrets.get("ext_api", {}).get("open_api_key")
        if not api_key:
            api_key = os.environ.get("EXT_API_OPEN_API_KEY")

        if not api_key:
            st.error("⚠️ Missing OpenAI API Key. Please add it to Streamlit secrets.")
            return
        client = OpenAI(api_key=api_key)

        # 1. Display user prompt (NOW USING CARD STYLING)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Render the user message immediately with the card style
        with st.chat_message("user"):
            st.markdown(f'<div class="user-response-card">{prompt}</div>', unsafe_allow_html=True)
        
        # ✅ SCROLL POINT 1: Immediately scroll to the bottom after the transcribed user prompt is rendered.
        scroll_to_bottom() 


        # 2. Get and stream assistant's response (with concurrent TTS setup)
        full_reply = ""
        sentence_buffer = ""
        futures = []

        # Using ThreadPoolExecutor for concurrent TTS synthesis
        with ThreadPoolExecutor(max_workers=5) as executor:
            with st.spinner(f"Thinking in **{response_mode} Mode**…"):
                with st.chat_message("assistant"):
                    # Use a dedicated placeholder for the styled card content
                    card_placeholder = st.empty() 
                    reply_generator = get_chatbot_response(prompt, data_frames, response_mode)

                    # Stream text and submit TTS jobs concurrently
                    for chunk in reply_generator:
                        full_reply += chunk
                        sentence_buffer += chunk

                        # ✅ FIX: Format the full reply for display before rendering
                        display_reply = format_large_numbers_for_tts(full_reply)

                        # Display the text immediately, wrapped in the custom card div
                        content_with_card = f"""
                        <div class="response-card">
                            {display_reply}
                        </div>
                        """
                        # Use unsafe_allow_html=True to render the custom HTML and CSS
                        card_placeholder.markdown(content_with_card, unsafe_allow_html=True) 

                        # Sentence Chunking Logic
                        # 💥 AGGRESSIVE CHUNKING: Split ONLY on hard stops or explicit newlines
                        sentence_end_match = re.search(r'[.?!]\s+|\n\n', sentence_buffer)

                        if narration_mode and sentence_end_match:
                            # Find the actual split point
                            split_point = sentence_end_match.end()

                            # Extract the sentence chunk for synthesis
                            sentence_to_synth = sentence_buffer[:split_point].strip()
                            # Keep remaining text and remove leading whitespace
                            sentence_buffer = sentence_buffer[split_point:].lstrip()

                            if sentence_to_synth:
                                
                                # ✅ FIX: Pre-process the sentence to insert commas for TTS to read large numbers correctly
                                pre_processed_sentence = format_large_numbers_for_tts(sentence_to_synth)
                                
                                # Submit the synthesis job to the thread pool
                                future = executor.submit(synthesize_speech, pre_processed_sentence, client)
                                futures.append(future)

                    # After the LLM stream is finished, process any remaining text in the buffer
                    if narration_mode and sentence_buffer.strip():
                        sentence_to_synth = sentence_buffer.strip()
                        # ✅ FIX: Pre-process the final sentence chunk as well
                        pre_processed_sentence = format_large_numbers_for_tts(sentence_to_synth)
                        
                        future = executor.submit(synthesize_speech, pre_processed_sentence, client)
                        futures.append(future)

            # 3. Store the *formatted* text response in history
            # Apply final formatting to the complete reply before storing
            final_display_reply = format_large_numbers_for_tts(full_reply)
            st.session_state.messages.append({"role": "assistant", "content": final_display_reply})

            # 4. Sequential Audio Playback (as chunks complete)
            if narration_mode and futures:
                st.toast("Synthesizing audio and starting playback...", icon="🎧")

                # 💥 NEW: Sequential Playback Logic - BYPASSES SLOW PYDUB MERGING
                audio_chunks_b64 = []
                combined_audio_valid = True # Reset for new logic

                # Iterate through the futures to collect chunks in sentence order
                for future in futures:
                    try:
                        audio_bytes, error = future.result(timeout=60) # Wait for the result
                        if error:
                            st.error(error)
                            combined_audio_valid = False # Signal failure
                            break
                        elif audio_bytes:
                            # Base64 encode the chunk immediately
                            b64_chunk = base64.b64encode(audio_bytes).decode()
                            audio_chunks_b64.append(b64_chunk)

                    except Exception as e:
                        # NOTE: Removed the pydub-specific error message
                        st.error(f"Error during audio processing: {e}. ")
                        combined_audio_valid = False
                        break

                if combined_audio_valid and audio_chunks_b64:
                    # Play the chunks sequentially using custom HTML/JS player
                    play_sequential_audio(audio_chunks_b64)
                    
        # 5. ✅ SCROLL POINT 2: Scroll to the bottom after all content, including the audio player, is rendered.
        # This addresses the second part of the user's request.
        scroll_to_bottom()
        
    # --- Streamlit UI ---
    st.markdown("""
    <style>

    /* --- USER MESSAGE CONTAINER (RIGHT ALIGNED, COLLAPSED WIDTH) --- */

    /* 1. Outer user message container: right-align (Flexbox) */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        display: flex !important;
        flex-direction: row-reverse !important;  /* Puts avatar on the right */
        
        /* CRITICAL FIX: Explicitly use flex-end to push the entire block flush right */
        justify-content: flex-end !important;  
        
        align-items: flex-start !important;
        width: 100% !important; 
        margin: 10px 0 !important;
        background-color: transparent !important;
        
        /* ADDED: Zero out right padding/margin that might be inherited from Streamlit's container */
        padding-right: 0 !important;
    }

    /* 2. Inner message content wrapper (The bubble container) - USER SPECIFIC */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) 
        [data-testid="stChatMessageContent"] {
        
        /* CRITICAL ALIGNMENT FIX: Pushes the entire content block to the right */
        margin-left: auto !important;
        margin-right: 0 !important;
        
        /* Use block display to ensure margin: auto works reliably */
        display: block !important; 
        
        /* Collapse and Max-Width */
        width: auto !important; 
        max-width: fit-content !important; 
        text-align: right; 
        
        /* Anti-stretch rules */
        flex-shrink: 1 !important;
        flex-grow: 0 !important; 
        min-width: 0 !important; /* Allows collapse */
        
        background-color: transparent !important;
        padding: 0 !important;
    }

    /* 3. Target the chat message internal DIVs */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) 
        [data-testid="stChatMessageContent"] > div {
        background-color: transparent !important;
        box-shadow: none !important;
        display: block !important; 
        width: fit-content !important; /* Forces internal collapse */
    }


    /* 4. User card styling (The actual bubble) */
    .user-response-card {
        background-color: #E6F3FF;
        color: #333;
        padding: 16px 20px;
        border-radius: 12px;
        border: 1px solid #C9E5FF;
        /* FIX: Corrected typo 'fit_content' to 'fit-content' */
        max-width: fit-content !important; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: left; 
        
        /* CRITICAL COLLAPSING RULES for the card */
        display: inline-block !important; /* Guarantees content-width collapse */
        /* FIX: Corrected typo 'fit_content' to 'fit-content' */
        width: fit-content !important;
        
        margin-left: 10px;  /* Space between card and avatar */
        margin-right: 0 !important; 
    }


    /* 5. Avatar styling: ensure it shows */
    [data-testid="chatAvatarIcon-user"] {
        display: block !important;
        margin-left: 5px !important; 
        float: none !important; 
        clear: none !important; 
        margin-right: 0 !important; 
    }
    [data-testid="chatAvatarIcon-user"] svg {
        width: 30px !important;
        height: 30px !important;
        opacity: 1 !important;
    }

    /* --- ASSISTANT MESSAGE CARD FIX (LEFT ALIGNED, COLLAPSING WIDTH) --- */

    /* 6. Ensure the assistant's custom card collapses to content width */
    .response-card {
        /* FIX: Changed from 'block' to 'inline-block' to force content width collapse */
        display: inline-block !important; 
        /* FIX: Corrected typo 'fit_content' to 'fit-content' and updated comment */
        width: fit-content !important; /* Ensures the assistant's bubble collapses to content width */
        
        /* Basic Styling */
        background-color: #F0F2F6; 
        color: #333;
        padding: 16px 20px;
        border-radius: 12px;
        border: 1px solid #E0E2E6;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: left;
    }

    /* 7. Dark mode styling */
    @media (prefers-color-scheme: dark) {
        .user-response-card {
            background-color: #3C4148 !important;
            color: #FFFFFF !important;
            border: 1px solid #555555 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
        }
        .response-card {
            background-color: #1E2327 !important;
            color: #FFFFFF !important;
            border: 1px solid #333333 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
        }
        
        /* Dark mode styling for the top button */
        button[data-testid="baseButton-secondary"]#clear_chat_top_button {
            background-color: #1E2327 !important;
            border: 1px solid #333333 !important;
            color: #FFFFFF !important;
        }
        button[data-testid="baseButton-secondary"]#clear_chat_top_button:hover {
            background-color: #3C4148 !important;
            border-color: #555555 !important;
        }
    }

    /* --- NEW CSS for the Top Clear Chat Button (Small Square) --- */
    button[data-testid="baseButton-secondary"]#clear_chat_top_button {
        width: 40px !important; /* Make it a small square */
        height: 40px !important;
        padding: 0 !important;
        border-radius: 8px !important; 
        
        /* Center the icon */
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        
        font-size: 16px !important;
        
        /* Custom button appearance (Light mode default) */
        background-color: #F0F2F6 !important;
        border: 1px solid #E0E2E6 !important;
        color: #333 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        
        /* Hover effect */
        transition: background-color 0.2s, transform 0.1s;
    }

    button[data-testid="baseButton-secondary"]#clear_chat_top_button:hover {
        background-color: #E6F3FF !important; /* Light blue on hover */
        border-color: #C9E5FF !important;
        transform: translateY(-1px);
    }

    </style>
    """, unsafe_allow_html=True)


    # ----------------------------------------

    st.title("Cashew Chatbot")

    # ✅ MODIFICATION: Removed the st.columns block for the button
    st.markdown("Data Analysis and Strategic Insights")
        
    st.markdown("---") # The line appears below the entire header/button block


    # Load data
    with st.spinner("Loading data…"):
        all_data_frames = load_all_data_for_chatbot()

    # Chat history
    if "messages" not in st.session_state:
        clear_chat_history() # Initialize history

    # Display chat history (NOW WITH CARD STYLING FOR BOTH ASSISTANT AND USER)
    for msg in st.session_state.messages:
        # Use the chat_message context manager to get the avatar and container
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                # Assistant uses the standard response card
                st.markdown(f'<div class="response-card">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                # User uses the distinct user response card
                st.markdown(f'<div class="user-response-card">{msg["content"]}</div>', unsafe_allow_html=True)


    # ===============================================
    # --- INPUT AND MODE MANAGEMENT (IN SIDEBAR) ---
    # ===============================================

    with st.sidebar:
        
        # 💥 ALIGNMENT MODIFICATION: ADDED EMPTY PLACEHOLDER FOR ALIGNMENT
        # Changed height from 40px to 5px to reduce the gap at the top of the sidebar.
        st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)
        
        # 1. RESPONSE MODE (MOVED UP)
        st.subheader("Response Mode")
        # Initialize the *Response* mode (Strategy/Info)
        if "response_mode" not in st.session_state:
            st.session_state["response_mode"] = "Strategy"

        # Add the Option Button (radio) for Response Mode
        st.session_state["response_mode"] = st.radio(
            "Choose the output style:",
            ("Strategy", "Info"),
            index=0 if st.session_state["response_mode"] == "Strategy" else 1,
            key="response_mode_radio",
            help="Strategy: C-suite advice with external context. Info: Factual, data-driven content."
        )
        
        st.markdown("---")
        
        # 2. OUTPUT OPTIONS (MOVED DOWN)
        st.subheader("Output Options")

        # Initialize the *Narration* mode (Off/On)
        if "narration_mode" not in st.session_state:
            st.session_state["narration_mode"] = False

        # Add the Narration checkbox
        st.session_state["narration_mode"] = st.checkbox(
            "🎧 Enable Narration (Auto-Play)",
            value=st.session_state["narration_mode"],
            key="narration_mode_checkbox",
            help="Reads out the assistant's final response automatically after a question."
        )

        st.markdown("---")

        # 3. INPUT MODE CONTROLS
        st.subheader("Input Mode") # 🆕 ADDED SUBHEADER
        # Initialize the *Input* mode (Text/Voice)
        if "input_mode" not in st.session_state:
            st.session_state["input_mode"] = "Text"

        # 4. Define button_label and new_mode (Needed for both Text and Voice modes)
        if st.session_state["input_mode"] == "Text":
            new_mode = "Voice"
            button_label = "🎙️ Switch to Voice Mode"
            help_text = "Click to activate microphone recording for spoken queries."
        else:
            new_mode = "Text"
            button_label = "⌨️ Switch to Text Mode"
            # 💥 AMENDMENT: Added the required help text/tooltip
            help_text = "Click to revert to the standard text input box."

        voice_prompt = None
        transcription_error = None

        # Create a dedicated placeholder for status messages
        status_placeholder = st.empty()
        
        # 💥 AMENDMENT: Conditional Layout for Buttons 💥

        if st.session_state["input_mode"] == "Voice":
            
            # Use columns to put mic_recorder and mode switch button on one line
            # FIX: Using equal distribution st.columns(2)
            col1, col2 = st.columns(2) 
            
            with col1:
                # 1. Render the mic recorder (Click to Record)
                audio_data = mic_recorder(
                    start_prompt="Click to Record",
                    stop_prompt="Recording... Click to Stop",
                    key="mic_recorder",
                    format="wav"
                )

            with col2:
                # 2. Render the Switch to Text Mode button
                # 💥 AMENDMENT: Added 'help=help_text'
                if st.button(
                    button_label, 
                    use_container_width=True, 
                    key="mode_switch",
                    help=help_text
                ):
                    st.session_state["input_mode"] = new_mode
                    st.rerun() 
            
        elif st.session_state["input_mode"] == "Text":
            # 1. Render the Switch to Voice Mode button (full width)
            # 💥 AMENDMENT: Added 'help=help_text'
            if st.button(
                button_label, 
                use_container_width=True, 
                key="mode_switch",
                help=help_text
            ):
                st.session_state["input_mode"] = new_mode
                st.rerun() 

        # The voice mode logic block starts here for post-button elements (status/transcription).
        if st.session_state["input_mode"] == "Voice":
            # The audio_data variable must be checked here since the recorder was rendered above.

            # 2. Process voice input if available
            if audio_data:
                # Only transcribe if the audio has changed (i.e., new recording completed)
                if 'last_audio_hash' not in st.session_state or st.session_state['last_audio_hash'] != hash(audio_data['bytes']):
                    
                    status_placeholder.info("Transcribing audio...")
                    voice_prompt, transcription_error = transcribe_audio(audio_data['bytes'])
                    
                    # Store the hash to prevent re-transcribing on simple sidebar clicks
                    st.session_state['last_audio_hash'] = hash(audio_data['bytes'])

                    if transcription_error:
                        status_placeholder.error(transcription_error)
                        
                        # ✅ MODIFICATION: Only switch back on error, not success
                        st.session_state["input_mode"] = "Text"
                        status_placeholder.warning("Switching back to Text Mode due to transcription error.")

                    elif voice_prompt:
                        status_placeholder.success(f"Transcription: **{voice_prompt}**")

                        # Set the transcribed text and the *current response mode* as the prompt to be processed
                        st.session_state["pending_prompt"] = voice_prompt
                        st.session_state["pending_response_mode"] = st.session_state["response_mode"]
                        st.session_state["pending_narration_mode"] = st.session_state["narration_mode"]
                        
                        # Rerun to process the prompt while keeping the input_mode as 'Voice'
                        st.rerun() 
            
            # ✅ FIX: Conditional separator only appears in Voice Mode
            st.markdown("---") 

        # 5. CLEAR CHAT HISTORY BUTTON (NEW LOCATION: AFTER VOICE INPUT)
        st.subheader("Chat History")
        st.button(
            "🗑️ Clear Chat History",
            on_click=clear_chat_history,
            key="clear_chat_sidebar_button", 
            
            use_container_width=True # Make it fill the sidebar width
        )
        st.markdown("---") # Final separator


    # ===============================================
    # --- END INPUT AND MODE MANAGEMENT IN SIDEBAR ---
    # ===============================================


    # --- Final Prompt Processing (handles successful voice transcription) ---

    # This block is correctly guarded by the session state keys and ensures that 
    # only a *new* pending prompt (only set by Voice Mode now) is processed.
    if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
        prompt = st.session_state.pop("pending_prompt")
        # Clean up other pending states
        response_mode = st.session_state.pop("pending_response_mode", "Strategy") # Default to Strategy
        narration_mode = st.session_state.pop("pending_narration_mode", False)
        
        # Process the prompt
        process_user_prompt(prompt, all_data_frames, response_mode, narration_mode)
        

    # 2. Process text input (st.chat_input should be last to be sticky at the bottom)
    if st.session_state["input_mode"] == "Text":
        # Get the current response mode for processing
        current_response_mode = st.session_state.get("response_mode", "Strategy")
        current_narration_mode = st.session_state.get("narration_mode", False)
        
        # ✅ FIX: Processing is done DIRECTLY inside st.chat_input to re-enable native smooth scrolling
        if prompt := st.chat_input("Enter your question..."):
            # The prompt is processed immediately here. Streamlit automatically handles 
            # smooth scrolling for text streaming within this block.

            process_user_prompt(prompt, all_data_frames, current_response_mode, current_narration_mode)

