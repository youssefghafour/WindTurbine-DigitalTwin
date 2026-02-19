import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import html  # safe HTML escaping

from data_loader import load_test_data
from model_logic import predict_proba, get_prediction

# ✅ IMPORTANT: use the NEW tool-calling agent (NOT agent_logic.py)
from wind_agent import ask_agent

# ========================
# Load dataset
# ========================
df = load_test_data()

# ========================
# CSS (professional fixed chat box)
# ========================
CUSTOM_CSS = """
#agent_chat_box {
    height: 260px;
    overflow-y: auto;
    border: 1px solid rgba(0,0,0,0.10);
    border-radius: 14px;
    padding: 10px 12px;
    background: white;
    box-shadow: 0 6px 14px rgba(0,0,0,0.05);
}
.chat_turn { margin-bottom: 10px; }
.bubble_user {
    display: inline-block;
    max-width: 92%;
    padding: 8px 10px;
    border-radius: 12px;
    background: rgba(52, 152, 219, 0.10);
    border: 1px solid rgba(52, 152, 219, 0.25);
    font-weight: 600;
}
.bubble_agent {
    display: inline-block;
    max-width: 92%;
    padding: 8px 10px;
    border-radius: 12px;
    background: rgba(0, 0, 0, 0.04);
    border: 1px solid rgba(0,0,0,0.10);
}
.small_meta {
    font-size: 12px;
    opacity: 0.75;
    margin-bottom: 4px;
}
"""

# ========================
# Helper: Fleet Distribution
# ========================
def make_distribution(probs, threshold, fleet_size):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.hist(probs, bins=30)
    ax.axvline(threshold, linestyle="--", linewidth=2)
    ax.set_title(f"Fleet Risk Distribution (First {fleet_size})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Failure Probability")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# ========================
# Gauge
# ========================
def make_gauge(prob, threshold):
    fig, ax = plt.subplots(figsize=(6, 1.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0.05, 0.4), 0.9, 0.15, alpha=0.2))
    thx = 0.05 + 0.9 * threshold
    ax.plot([thx, thx], [0.3, 0.8], linestyle="--", linewidth=2)

    pxv = 0.05 + 0.9 * prob
    ax.plot([pxv, pxv], [0.25, 0.85], linewidth=4)

    ax.text(0.05, 0.92, "Failure Probability", fontsize=10, fontweight="bold")
    ax.text(0.95, 0.92, f"{prob:.1%}", fontsize=10, fontweight="bold", ha="right")

    plt.tight_layout()
    return fig

# ========================
# Telemetry Scatter
# ========================
def make_telemetry_scatter(telemetry_df, warn_thr=1.5, crit_thr=2.5):
    t = telemetry_df.copy()
    t["AbsValue"] = t["Value"].abs()

    def risk_level(a):
        if a >= crit_thr:
            return "CRITICAL"
        if a >= warn_thr:
            return "WARNING"
        return "OK"

    t["Risk"] = t["AbsValue"].apply(risk_level)
    t["SensorIdx"] = np.arange(len(t))

    fig = px.scatter(
        t,
        x="SensorIdx",
        y="Value",
        color="Risk",
        color_discrete_map={
            "OK": "#90EE90",  # light green
            "WARNING": "orange",
            "CRITICAL": "black",
        },
        hover_data={"Sensor": True, "Value": ':.3f', "AbsValue": ':.3f', "SensorIdx": False},
        title="Telemetry Outliers (sorted by |value|)"
    )
    fig.update_layout(
        xaxis_title="Sensors (sorted by |value|)",
        yaxis_title="Normalized Sensor Value",
        legend_title_text="Risk",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    return fig

# ========================
# Fleet Overview
# ========================
def fleet_overview(threshold, top_k, fleet_size):
    fleet_size = int(max(1, min(int(fleet_size), len(df))))
    df_subset = df.iloc[:fleet_size].copy()

    X = df_subset.drop(columns=["Target"], errors="ignore")
    probs = predict_proba(X).astype(float)

    total = len(probs)
    flagged = int((probs >= threshold).sum())
    avg = float(np.mean(probs))
    p95 = float(np.quantile(probs, 0.95))

    kpi_html = f"""
    <div style='padding:12px; border-radius:12px;
                border:1px solid rgba(0,0,0,0.08);
                box-shadow:0 6px 15px rgba(0,0,0,0.05);'>
      <div><b>Fleet Size:</b> {total}</div>
      <div><b>Flagged (≥ {threshold:.2f}):</b> {flagged}</div>
      <div><b>Average Risk:</b> {avg:.2%}</div>
      <div><b>P95 Risk:</b> {p95:.2%}</div>
    </div>
    """

    top_k = int(max(1, min(int(top_k), total)))
    idx_sorted = np.argsort(-probs)[:top_k]
    top_df = pd.DataFrame({"InstanceID": idx_sorted, "FailureProb": probs[idx_sorted]})

    dist_fig = make_distribution(probs, threshold, fleet_size)
    return kpi_html, top_df, dist_fig

# ========================
# Instance Monitor
# ========================
def run_instance(instance_id, threshold):
    instance_id = int(max(0, min(int(instance_id), len(df) - 1)))
    row = df.iloc[[instance_id]]

    status, prob = get_prediction(row)

    status_html = f"""
    <div style='padding:12px; border-radius:12px;
                border:1px solid rgba(0,0,0,0.08);
                box-shadow:0 6px 15px rgba(0,0,0,0.05);'>
      <b>Instance:</b> #{instance_id}<br>
      <b>Status:</b> {status}<br>
      <b>Failure Probability:</b> {prob:.2%}<br>
      <b>Alarm Threshold:</b> {threshold:.2f}
    </div>
    """

    gauge = make_gauge(prob, threshold)

    telemetry = row.drop(columns=["Target"], errors="ignore").T.reset_index()
    telemetry.columns = ["Sensor", "Value"]
    telemetry["Value"] = telemetry["Value"].astype(float)
    telemetry["AbsValue"] = telemetry["Value"].abs()
    telemetry = telemetry.sort_values("AbsValue", ascending=False).drop(columns=["AbsValue"])
    telemetry["Value"] = telemetry["Value"].round(3)

    telemetry_plot = make_telemetry_scatter(telemetry)
    return status_html, gauge, telemetry_plot, telemetry

# ========================
# Refresh all
# ========================
def full_refresh(instance_id, threshold, top_k, fleet_size):
    fleet_kpi_html, top_df, dist_fig = fleet_overview(threshold, top_k, fleet_size)
    status_html, gauge_fig, telemetry_plot, telemetry_df = run_instance(instance_id, threshold)
    return fleet_kpi_html, top_df, dist_fig, status_html, gauge_fig, telemetry_plot, telemetry_df

# ========================
# AI Agent UI helpers
# ========================
def chat_html(history):
    if not history:
        return "<div id='agent_chat_box'><div class='small_meta'>Start asking the agent…</div></div>"

    history = history[-30:]
    blocks = ["<div id='agent_chat_box'>"]
    for u, a in history:
        u_safe = html.escape(str(u))
        a_safe = html.escape(str(a))

        blocks.append("<div class='chat_turn'>")
        blocks.append("<div class='small_meta'>You</div>")
        blocks.append(f"<div class='bubble_user'>{u_safe}</div>")
        blocks.append("</div>")

        blocks.append("<div class='chat_turn'>")
        blocks.append("<div class='small_meta'>Agent</div>")
        blocks.append(f"<div class='bubble_agent'>{a_safe}</div>")
        blocks.append("</div>")
    blocks.append("</div>")
    return "\n".join(blocks)

# ✅ Tool-calling Agent (pass df + context)
def agent_send(user_msg, history_state, instance_id, fleet_size, threshold):
    history_state = history_state or []
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return chat_html(history_state), history_state, ""

    try:
        instance_id = int(max(0, min(int(instance_id), len(df) - 1)))
        fleet_size = int(max(1, min(int(fleet_size), len(df))))
        threshold = float(threshold)

        # ✅ Pass ALL context so tools are correct
        answer = ask_agent(
            df,
            user_msg,
            instance_id=instance_id,
            fleet_size=fleet_size,
            threshold=threshold
        )

    except Exception as e:
        answer = f"⚠️ Agent error: {str(e)}"

    history_state.append((user_msg, answer))
    return chat_html(history_state), history_state, ""

def agent_clear():
    return chat_html([]), [], ""

def set_quick_question(choice, instance_id):
    instance_id = int(max(0, min(int(instance_id), len(df) - 1)))
    mapping = {
        "Fleet summary": "Give me a fleet summary (flagged count, avg risk, p95 risk, and top risky instances).",
        "How to read Fleet Risk Distribution?": "How to read Fleet Risk Distribution?",
        "Why risky? (this instance)": f"Which sensors deviate the most for InstanceID={instance_id}? Use real values.",
        "Failure probability (this instance)": f"Get failure probability for InstanceID={instance_id}.",
        "Top deviating sensors (this instance)": f"Get top deviating sensors for InstanceID={instance_id} (top 10).",
        "Explain telemetry colors": "Explain telemetry scatter colors: OK(light green), WARNING(orange), CRITICAL(black).",
    }
    return mapping.get(choice, "")

# ========================
# UI
# ========================
with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
    gr.Markdown("# 🌬️ Wind Turbine Digital Twin Dashboard")
    gr.Markdown("Instance Monitor uses **full dataset**. Fleet Distribution uses **subset size you choose**.")

    chat_state = gr.State([])

    with gr.Row():
        # LEFT
        with gr.Column(scale=1):
            instance_id = gr.Slider(0, len(df) - 1, value=min(520, len(df) - 1), step=1, label="Instance ID")
            threshold = gr.Slider(0.05, 0.95, value=0.50, step=0.01, label="Alarm Threshold")
            fleet_size = gr.Slider(50, len(df), value=min(500, len(df)), step=50,
                                   label="Fleet Sample Size (for distribution only)")
            top_k = gr.Slider(5, 50, value=15, step=1, label="Top-K risky instances (from fleet subset)")
            refresh_btn = gr.Button("Refresh", variant="primary")

            status_card = gr.HTML()
            fleet_kpi = gr.HTML()
            top_table = gr.DataFrame(interactive=False)

            with gr.Accordion("📋 Telemetry (sorted by |value|)", open=False):
                telemetry_tbl = gr.DataFrame(interactive=False)

        # RIGHT
        with gr.Column(scale=2):
            gauge_plot = gr.Plot()
            gr.Markdown("## 🎯 Telemetry Scatter (hover to see sensor)")
            telemetry_plot = gr.Plot()

            gr.Markdown("## 📈 Fleet Risk Distribution")
            dist_plot = gr.Plot()

            gr.Markdown("## 🤖 AI Agent (Tool Calling)")
            chat_view = gr.HTML(chat_html([]))

            with gr.Row():
                quick = gr.Dropdown(
                    choices=[
                        "Fleet summary",
                        "How to read Fleet Risk Distribution?",
                        "Failure probability (this instance)",
                        "Why risky? (this instance)",
                        "Top deviating sensors (this instance)",
                        "Explain telemetry colors",
                    ],
                    value="How to read Fleet Risk Distribution?",
                    label="Quick questions"
                )
                fill_btn = gr.Button("Fill", variant="secondary")

            agent_in = gr.Textbox(lines=2, label="Type your question")
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear chat", variant="secondary")

    # Refresh
    refresh_btn.click(
        full_refresh,
        inputs=[instance_id, threshold, top_k, fleet_size],
        outputs=[fleet_kpi, top_table, dist_plot, status_card, gauge_plot, telemetry_plot, telemetry_tbl],
    )

    # Fill quick
    fill_btn.click(
        set_quick_question,
        inputs=[quick, instance_id],
        outputs=[agent_in],
    )

    # Send / Enter  ✅ include fleet_size + threshold
    send_btn.click(
        agent_send,
        inputs=[agent_in, chat_state, instance_id, fleet_size, threshold],
        outputs=[chat_view, chat_state, agent_in],
    )
    agent_in.submit(
        agent_send,
        inputs=[agent_in, chat_state, instance_id, fleet_size, threshold],
        outputs=[chat_view, chat_state, agent_in],
    )

    # Clear chat
    clear_btn.click(
        agent_clear,
        inputs=None,
        outputs=[chat_view, chat_state, agent_in],
    )

if __name__ == "__main__":
    demo.launch()
