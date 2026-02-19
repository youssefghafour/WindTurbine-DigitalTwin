# Wind Turbine Digital Twin with LLM Interface
A proof-of-concept system demonstrating how conversational AI can enhance predictive maintenance for wind turbines.
The system uses an XGBoost failure prediction model and a local LLM (Ollama + Qwen3) for natural language interaction with the dataset.


## Key Idea
### Model predicts, LLM interprets.
The machine learning model generates real failure probabilities.
The LLM never invents numbers, it queries the dataset and model outputs, then explains results in clear natural language.


## Architecture
''' bash
┌─────────────────────────────────────────────────────────────┐
│                   Gradio Dashboard (app.py)                │
│  Fleet Distribution │ Instance Monitor │ Telemetry │ Chat  │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    AI Agent (agent_logic.py)               │
│     Local LLM (Ollama + Qwen3:0.6b)                        │
│     Builds context + Answers user questions                │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  Predictive Model (XGBoost)                │
│   wind_final_full_train.json → Failure Probability         │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Wind Turbine Dataset                    │
│   40 normalized sensor features (V1–V40) + Target         │
└─────────────────────────────────────────────────────────────┘






Predictive Maintenance + AI Agent (Local LLM)

This project implements a Digital Twin system for wind turbine predictive maintenance.

The system integrates:
✅ XGBoost predictive model
✅ Interactive Gradio dashboard
✅ Fleet-level risk monitoring
✅ Instance-level anomaly visualization
✅ Local AI Agent using Ollama + Qwen3

The AI Agent runs fully locally (no cloud API required).

How To Run The Project (Step-by-Step)?

Follow these steps in your terminal.

1️⃣ Clone The Repository

"git clone https://github.com/youssefghafour/WindTurbine-DigitalTwin.git"

"cd WindTurbine-DigitalTwin"

2️⃣ Create Virtual Environment

"python3 -m venv .venv"

"source .venv/bin/activate"

3️⃣ Install Python Dependencies

"pip install -r requirements.txt"

Or try: "pip install gradio pandas numpy matplotlib plotly xgboost langchain langchain-community langchain-experimental ollama tabulate"



Install Ollama (Required for AI Agent)

Install Ollama (Linux / macOS):

"curl -fsSL https://ollama.com/install.sh | sh"

"ollama pull qwen3:0.6b"



Run The Dashboard:

"python app.py"

Then open the link "http://127.0.0.1:7860"


