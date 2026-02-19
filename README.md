# Wind Turbine Digital Twin with LLM Agent
Digital Twin system demonstrating how conversational AI enhances predictive maintenance for wind turbines.

The system integrates:
- XGBoost failure prediction model
- Interactive Gradio monitoring dashboard
- Fleet-level risk analytics
- Instance-level anomaly detection
- Tool-Calling AI Agent (Ollama + Qwen3:0.6b)
The entire system runs fully locally — no cloud APIs required.


## Key Concept
The predictive model computes real failure probabilities.

The LLM:
- Does NOT invent numbers
- Does NOT hallucinate values
- Calls real Python tools
- Retrieves actual model outputs
- Explains results in clear natural language
This guarantees consistency between dashboard results and AI explanations.



## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                   Gradio Dashboard (app.py)                │
│  Fleet Distribution │ Instance Monitor │ Telemetry │ Chat  │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│        Wind Turbine Tool-Calling Agent (wind_agent.py)    │
│        Local LLM: Ollama + Qwen3:0.6b                      │
│        Executes structured tool calls                      │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                         Tools (tools.py)                   │
│  get_failure_probability()                                 │
│  get_fleet_summary()                                       │
│  get_top_deviating_sensors()                               │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Predictive Model (XGBoost)                    │
│     wind_final_full_train.json → Failure Probability       │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Wind Turbine Dataset                    │
│   40 normalized sensor features (V1–V40) + Target          │
└─────────────────────────────────────────────────────────────┘

```



## How the LLM Gets Data
The LLM never directly reads raw data.

Instead:
```
User: "Analyze Instance 4968"
→ LLM calls get_failure_probability(instance_id=4968)
→ Tool returns real probability from XGBoost
→ LLM calls get_top_deviating_sensors()
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



## Telemetry Scatter
Sensors are sorted by absolute deviation.

Color coding:
| Color | Meaning |
| --- | --- |
| Light Green | Normal |
| Orange | Warning |
| Black | Critical |
This highlights abnormal sensors quickly.


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

## 5. Start Ollama Server
```
ollama serve
```
Leave this running.


## 6. Pull the Required Model
```
ollama pull qwen3:0.6b
```


## 7. Run the Dashboard
```
python app.py
```
Open in browser: [http://127.0.0.1:7860/](http://127.0.0.1:7860/)


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
- Input: 40 normalized sensor features (V1–V40)
- Output: Failure probability (0–1)
- Trained offline
- Used in real-time inference


# System Behavior
- Fleet distribution uses adjustable subset size
- Instance monitor always uses full dataset
- Tool-Calling Agent retrieves real analytics
- Entire system runs locally
- No external APIs


# Author
Youssef Abdul Ghafour

NTNU Ålesund – Mechatronics & Intelligent Systems

