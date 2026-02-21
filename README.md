# Predictive Maintenance for Wind Turbine
AI-Enhanced Digital Twin Dashboard with Tool-Calling LLM

The system integrates:
- XGBoost failure prediction model
- Interactive Gradio monitoring dashboard
- Fleet-level risk analytics
- Instance-level anomaly detection
- Tool-Calling AI Agent (Ollama + Qwen3:0.6b)
The entire system runs fully locally, no cloud APIs required.


## Key Concept
The predictive model computes real failure probabilities.

The LLM:
- Calls real Python tools
- Retrieves actual model outputs
- Explains results in clear natural language
This guarantees consistency between dashboard results and AI explanations.



## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                   Gradio Dashboard (app.py)                 │
│  Fleet Distribution │ Instance Monitor │ Telemetry │ Chat   │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│        Wind Turbine Tool-Calling Agent (wind_agent.py)      │
│        Local LLM: Ollama + Qwen3:0.6b                       │
│        Executes structured tool calls                       │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                         Tools (tools.py)                    │
│  get_failure_probability()                                  │
│  get_fleet_summary()                                        │
│  get_top_deviating_variables()                              │
│  get_variable_info()                                        │
│  get_dashboard_help()                                       │      
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Predictive Model (XGBoost)                     │
│     wind_final_full_train.json → Failure Probability        │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Wind Turbine Dataset                     │
│   40 normalized variables features (V1–V40) + Target        │
│    Each row represents one independent instance.            │
└─────────────────────────────────────────────────────────────┘

```



## How the LLM Gets Data
The LLM never directly reads raw data.

Instead:
```
User: "Analyze Instance 4968"
→ LLM calls get_failure_probability
→ Tool returns real probability from XGBoost
→ LLM calls get_top_deviating_variables()
→ Tool returns actual deviation values
→ LLM explains the risk clearly

```
Why this approach?
- Raw data → hallucination risk
- Fine-tuned model → outdated when data changes
- Tool calling → real-time, deterministic, verifiable
This mirrors real industrial AI deployment.




# Dashboard Components
## Fleet Risk Distribution
- Histogram of predicted failure probabilities
- Adjustable fleet subset size
- Alarm threshold visualization
- Shows distribution shape (healthy vs high-risk cluster)


## Instance Monitor
- Failure probability gauge
- Instance-specific prediction
- Alarm threshold comparison
- Uses full dataset



## Normalized Variable Deviations
Each point represents one variable (V1–V40) for the selected instance.

Variables are standardized (mean ≈ 0).

Color coding is based on absolute deviation:

| Color | Meaning |
| --- | --- |
| Light Green | abs(value) < 1.5 (OK) |
| Orange | 1.5 ≤ abs(value) < 2.5 (WARNING) |
| Black | abs(value) ≥ 2.5 (CRITICAL) |

These colors describe feature deviation, not failure probability.


## AI Agent
- Continuous chat interface
- Tool-calling architecture
- Real-data explanations
- Maintenance recommendations
- Instance-aware responses


# Tutorial: Run the Project Locally
## Prerequisites
- Python 3.9+
- 8GB RAM recommended
- 1GB free disk space (for Ollama model)

## 1. Clone Repository
```
git clone https://github.com/youssefghafour/WindTurbine-DigitalTwin.git
cd WindTurbine-DigitalTwin
```

## 2. Create Virtual Environment
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

## 3. Install Dependencies
```
pip install -r requirements.txt
```
If needed:
```
pip install gradio pandas numpy matplotlib plotly xgboost ollama langchain langchain-community
```

## 4. Install Ollama (Local LLM)
### Linux / macOS
```
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows
Download from: [https://ollama.com/download](https://ollama.com/download)
Verify:
```
ollama --version
```

## 5. Start Ollama, Pull the Model, and Run the Dashboard
Just copy and limin at terminal:
```
ollama serve
ollama pull qwen3:0.6b
python app.py
```
Then open in browser: [http://127.0.0.1:7860/](http://127.0.0.1:7860/)


# Project Structure
```
.
├── app.py                  # Gradio dashboard
├── wind_agent.py           # Tool-calling LLM agent
├── tools.py                # Tool definitions & execution logic
├── model_logic.py          # XGBoost prediction wrapper
├── data_loader.py          # Dataset loader
├── wind_final_full_train.json
├── requirements.txt
└── data/
```

# Predictive Model
- Algorithm: XGBoost
- Input: 40 normalized variable features (V1–V40)
- Output: Failure probability (0–1)
- Trained offline
- Used in real-time inference


# System Behavior

## Dataset Representation
- Each row in the dataset is treated as an independent **instance**.
- The system does **not assume** that one row equals one physical turbine.
- All analytics operate strictly at the instance (row) level.

## Fleet Analytics
- Fleet Risk Distribution uses an adjustable **Fleet Monitoring Sample Size** (subset of rows).
- **Instances Flagged** counts rows where `failure_probability ≥ alarm_threshold`.
- **Avg Risk** is the **mean predicted failure probability** across the selected fleet subset.
- Changing the threshold affects flagged count but does not change ranking order.

## Instance Monitoring
- The Instance Monitor analyzes the selected instance (row) in detail.
- The Failure Probability Gauge compares predicted probability against the alarm threshold.
- The Normalized Variable Deviations scatter shows standardized feature values (V1–V40) for the selected instance.
- Variable color coding is based on absolute deviation:
  - Light Green → `abs(value) < 1.5` (OK)
  - Orange → `1.5 ≤ abs(value) < 2.5` (WARNING)
  - Black → `abs(value) ≥ 2.5` (CRITICAL)

## AI Agent
- The Tool-Calling Agent retrieves real analytics using deterministic Python tools.
- The LLM does **not** access raw data directly.
- The LLM does **not** invent numbers.
- All numerical outputs come from the XGBoost model or verified tool functions.

## Execution Environment
- The entire system runs fully locally.
- No external APIs are used.
- Model inference (XGBoost) and LLM reasoning (Ollama + Qwen) are executed on-device.


# Author
Youssef Abdul Ghafour

NTNU Ålesund – Mechatronics & Intelligent Systems

