# GECO Asia AI Hackathon Project

This repository contains our submission for the **GECO Asia AI Hackathon**, where we set out to transform how users interact with business data. Our goal was simple: **take a conventional dashboard and supercharge it with AI-powered insights**—making data exploration faster, smarter, and conversational.

---

## 🚀 What We Built

We developed an AI-enhanced **Dashboard + Chatbot** solution that goes beyond static charts and tables. The system integrates:

* **AI-powered insight generation**
  Instantly produces meaningful summaries, highlights anomalies, and surfaces trends without manual digging.

* **Conversational Chatbot Assistant**
  A natural-language chatbot that can discuss company matters based on the data you provide.
  Using a **Retrieval-Augmented Generation (RAG)** pipeline, the assistant can ingest files and respond with context-aware, accurate answers.

* **Actionable Recommendations**
  The AI not only points out insights—it suggests next steps and guides decision-making in real time.

* **Personalised, Instant Responses**
  From high-level summaries to deep-dive queries, the system delivers tailored insights instantly.

---

## 💡 Why This Matters

Traditional dashboards tell you *what* is happening.
Our AI-powered companion tells you **why**, **so what**, and **what to do next**—all through conversation.

This project demonstrates how human–AI collaboration can transform raw data into rapid, decision-ready intelligence.

---
## Solutions Architecture 

A Streamlit dashboard with Plotly charts meets a chatbot powered by OpenAI LLM and PandasAI. Cached Pandas data loads fuel real-time insights. Together, this stack makes exploring and acting on business data fast, smart, and conversational.

| **Component**               | **Technology / Function**              | **Description**                                                                                                         |
| --------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Presentation Layer**      | Streamlit                              | High-fidelity, interactive interface for Dashboard + Chatbot, handling navigation, input, and visualization.            |
| **Data & Core Logic Layer** | Python / Pandas, PandasAI              | Loads multi-source CSVs with caching; executes natural-language-driven analytics for charts, KPIs, and chatbot queries. |
| **AI / LLM Services Layer** | OpenAI (e.g. GPT-4.1, GPT-4o, Whisper, TTS) | Powers summaries, reasoning, transcription, and voice responses for conversational insights and recommendations.        |

---

## 🌟 Final Version Enhancements (Compared to the Initial Build)

The final version of the **Dashboard (dashboard_finalv.py) + Chatbot (ask_final.py)** introduced major improvements across speed, UX, prompting, and deployment.

### ⚡ Performance Enhancements

* **Concurrent TTS Processing**
  TTS runs in parallel threads, enabling immediate text streaming while audio is generated in the background.

* **API Caching for LLM Output**
  Heavy LLM summary calls are cached using `st.cache_data`, drastically reducing load times and API costs.

### 🎧 UX & Interaction Improvements

* **Native Smooth Chat Scrolling**
  Chat logic was moved into the correct `st.chat_input` block to enable Streamlit’s built-in auto-scroll.

* **Sequential Audio Playback**
  Long messages are split into ordered segments for smoother narration.

### 🧠 AI Strategy & Prompting Enhancements

* **Time-Horizon Adaptive Prompts**
  Strategies automatically adjust to weekly, monthly, or yearly timeframes—short-term for tactical insights, long-term for strategic direction.

### 🔐 Deployment & Reliability Improvements

* **Robust API Key Detection**
  A new helper checks Streamlit secrets first, then environment variables, ensuring consistent behavior across local, Docker, and cloud deployments.

---

## 🛠 Setup Instructions for Dashboard + Chatbot (Docker Version)

The final version of the **Dashboard + Chatbot** runs fully inside Docker for a clean and reproducible deployment environment.
Instructions are provided for **Windows (PowerShell)** and **Linux (bash)**.

---

## 📥 1. Download the Project Files

1. Visit the project’s GitHub repository.
2. Click **Code → Download ZIP** or clone the repository:

```bash
git clone https://github.com/miqqie/cashew.git
```

3. Navigate into the project directory:

```bash
cd cashew
```

---

## 🔐 2. Configure Your API Key

The Dashboard + Chatbot container reads your OpenAI API key from an environment variable.

Replace `"your_key_here"` with your actual API key in Step 4.

Alternatively, enter your API key in the secrets.toml file.

---

## 🛠 3. Build the Docker Image

### Windows (PowerShell)

```powershell
docker build -t streamlit-app-image .
```

### Linux (bash)

```bash
docker build -t streamlit-app-image .
```

---

## ▶️ 4. Run the Dashboard + Chatbot

### Windows (PowerShell)

```powershell
docker run -d `
  -p 8501:8501 `
  -e EXT_API_OPEN_API_KEY="your_key_here" `
  --name streamlit-dashboard `
  streamlit-app-image
```

### Linux (bash)

```bash
docker run -d \
  -p 8501:8501 \
  -e EXT_API_OPEN_API_KEY="your_key_here" \
  --name streamlit-dashboard \
  streamlit-app-image
```

If you store your API key in secrets.toml for local development, you can omit the -e EXT_API_OPEN_API_KEY when running Docker.

This initialises the full Dashboard + Chatbot application.

---

## 🌐 5. Open the Application in Your Browser

Docker does **not** open a browser automatically.
Open the following URL manually:

```
http://localhost:8501
```

This loads the interactive dashboard along with the integrated AI chatbot.

---

## ♻️ 6. Managing the Container

Commands below work on both Windows and Linux:

### Stop the container

```bash
docker stop streamlit-dashboard
```

### Restart the container

```bash
docker start streamlit-dashboard
```

### Remove the container

```bash
docker rm -f streamlit-dashboard
```

---

# 🎉 Done!

Your **Dashboard + Chatbot** is now running fully inside Docker.
If you experience issues, verify:

* Docker Desktop is running (Windows)
* Your user has Docker permissions (Linux)
* Port **8501** is available
* The API key is correctly passed via `EXT_API_OPEN_API_KEY`

