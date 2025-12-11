# GECO Asia AI Hackathon Project

This repository contains our submission for the **GECO Asia AI Hackathon**, where we set out to transform the way teams interact with business data. Our goal was simple: **take a conventional dashboard and supercharge it with AI-powered insights**—making data exploration faster, smarter, and conversational.

## 🚀 What We Built

We developed an AI-enhanced analytics solution that goes beyond static charts and tables. The system integrates:

- **AI-powered insight generation**  
  Instantly produces meaningful summaries, highlights anomalies, and surfaces trends without manual digging.

- **Conversational Chatbot Assistant**  
  A natural-language chatbot that can discuss company matters based on the data you provide.  
  Using a **Retrieval-Augmented Generation (RAG)** pipeline, the assistant can ingest your files and respond with context-aware, accurate answers.

- **Actionable Recommendations**  
  The AI not only provides insights—it also suggests next steps and guides decision-making in real time.

- **Personalised, Instant Responses**  
  Ask any question about your data, from high-level summaries to deep-dive queries, and get tailored answers immediately.

## 💡 Why This Matters

Traditional dashboards show you *what* is happening.  
Our AI-powered companion tells you **why**, **so what**, and **what to do next**—all through conversation.

This project showcases how human-AI collaboration can turn raw data into rapid decision-ready intelligence.

---

# Setup Instructions
---

````markdown
# Local Setup Guide

Follow the steps below to run the chatbot on your local machine.

---

## 📥 1. Download the Project Files

1. Go to the project’s GitHub repository.
2. Click **Code → Download ZIP**, or clone the repository

````

3. Navigate into the project directory:

```bash
cd your-repo
```

---

## 🧪 2. Create the Conda Environment

Ensure you have **Anaconda** or **Miniconda** installed.

Create the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

This will create an environment (e.g., `myenv`) with all necessary dependencies.

---

## ▶️ 3. Activate the Environment

```bash
conda activate myenv
```

---

## 🔐 4. Add Your OpenAI API Key

Streamlit reads secrets from `.streamlit/secrets.toml`.

1. Create the `.streamlit` directory:

```bash
mkdir -p .streamlit
```

2. Create the `secrets.toml` file:

```bash
nano .streamlit/secrets.toml
```

3. Add your OpenAI API key:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

Save and close the file.

---

## ▶️ 5. Run the Chatbot

Run the Streamlit app:

```bash
streamlit run dashboard_c.py
```

Your browser will open automatically.

---

## 🎉 Done!

The chatbot is now running locally. If you encounter issues, ensure:

* Your conda environment is activated
* Your `.streamlit/secrets.toml` contains a valid API key
* All dependencies were installed correctly

```

Thanks for checking out our hackathon project!  
Feel free to explore, test, and build on top of it.
