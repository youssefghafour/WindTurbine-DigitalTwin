# Wind Turbine Digital Twin with LLM Interface
A proof-of-concept system demonstrating how conversational AI can enhance predictive maintenance for wind turbines.
The system uses an XGBoost failure prediction model and a local LLM (Ollama + Qwen3) for natural language interaction with the dataset.


## Key Idea
### Model predicts, LLM interprets.
The machine learning model generates real failure probabilities.
The LLM never invents numbers, it queries the dataset and model outputs, then explains results in clear natural language.


## Architecture
```
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
```



## How the LLM Gets Data
The LLM does not directly access raw data blindly.

Instead:
```
User: "Analyze Instance 4968"

→ Model computes failure probability
→ Telemetry is sorted by largest deviations
→ LLM receives structured context
→ LLM explains risk in natural language
```
Why this approach?

- Raw data → hallucination risk
- Tool-based structure → reliable explanations
- Real-time inference → always consistent with model




# Tutorial: Getting Everything Working
## Prerequisites
- Python 3.9+
- 8GB RAM recommended
- 1GB disk space (for Ollama model)


## Step 1: Clone Repository
```
git clone https://github.com/youssefghafour/WindTurbine-DigitalTwin.git
cd WindTurbine-DigitalTwin
```


## Step 2: Create Virtual Environment
### Linux / macOS
```
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Step 3: Install Dependencies
```
pip install -r requirements.txt
```
 If needed:
 ```
pip install gradio pandas numpy matplotlib plotly xgboost langchain langchain-community langchain-experimental ollama tabulate
```


## Step 4: Install Ollama (Local LLM)
### Linux / macOS
```
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows
Download installer from:
[live_dome](https://ollama.com/download)

Then verify installation:
```
ollama --version
```

## Step 5: Pull the Qwen Model
```
ollama pull qwen3:0.6b
```

## Step 7: Run the Dashboard
´´´
python app.py
´´´

Open:
```
http://127.0.0.1:7860
``






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


