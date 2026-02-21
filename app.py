import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import html

from data_loader import load_test_data
from model_logic import predict_proba, get_prediction
from wind_agent import WindTurbineMaintenanceAgent


# Data
df = load_test_data()


# CSS
CUSTOM_CSS = """
#agent_chat_box {
    height: 350px;
    overflow-y: auto;
    border: 1px solid rgba(0,0,0,0.10);
    border-radius: 14px;
    padding: 15px;
    background: #fdfdfd;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
}
.chat_turn { margin-bottom: 15px; overflow: hidden; }
.bubble_user {
    display: inline-block;
    max-width: 85%;
    padding: 10px 14px;
    border-radius: 15px 15px 2px 15px;
    background: #3498db;
    color: white;
    font-weight: 500;
    float: right;
    clear: both;
}
.bubble_agent {
    display: inline-block;
    max-width: 85%;
    padding: 10px 14px;
    border-radius: 15px 15px 15px 2px;
    background: #e9e9eb;
    color: #2c3e50;
    border: 1px solid rgba(0,0,0,0.05);
    float: left;
    clear: both;
}
.small_meta {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    opacity: 0.6;
    margin-bottom: 2px;
}
.meta_user { text-align: right; }
"""


# Dashboard visualization logic
def make_distribution(probs, threshold, fleet_size):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.hist(probs, bins=30, color="#3498db", alpha=0.7)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2)
    ax.set_title(f"Fleet Risk Distribution (N={fleet_size})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Failure Probability")
    ax.set_ylabel("Instance Count")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def make_gauge(prob, threshold):
    fig, ax = plt.subplots(figsize=(6, 1.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    color = "red" if prob >= threshold else "#27ae60"
    ax.add_patch(plt.Rectangle((0.05, 0.4), 0.9, 0.15, color="gray", alpha=0.1))
    ax.add_patch(plt.Rectangle((0.05, 0.4), 0.9 * prob, 0.15, color=color, alpha=0.7))
    thx = 0.05 + 0.9 * threshold
    ax.plot([thx, thx], [0.3, 0.8], color="black", linestyle="--", linewidth=2)
    ax.text(0.5, 0.15, f"Prediction: {prob:.1%}", fontsize=12, fontweight="bold", ha="center")
    plt.tight_layout()
    return fig


def make_variable_scatter(variable_df, warn_thr=1.5, crit_thr=2.5):
    t = variable_df.copy()
    t["AbsValue"] = t["Value"].abs()

    def risk_level(a):
        if a >= crit_thr:
            return "CRITICAL"
        if a >= warn_thr:
            return "WARNING"
        return "OK"

    t["Risk"] = t["AbsValue"].apply(risk_level)
    t["VarIdx"] = np.arange(len(t))

    fig = px.scatter(
        t, x="VarIdx", y="Value", color="Risk",
        color_discrete_map={"OK": "#90EE90", "WARNING": "orange", "CRITICAL": "black"},
        hover_data=["Variable", "Value"],
        title="Normalized Variable Deviations (z-score)"
    )
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# Data orchestration
def fleet_overview(threshold, top_k, fleet_size):
    fleet_size = int(max(1, min(int(fleet_size), len(df))))
    df_subset = df.iloc[:fleet_size].copy()
    X = df_subset.drop(columns=["Target"], errors="ignore")
    probs = predict_proba(X).astype(float)

    flagged = int((probs >= threshold).sum())

    kpi_html = f"""
    <div style='padding:12px; border-radius:12px; border:1px solid #eee; background:#fff; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
      <div style='color:#7f8c8d; font-size:12px;'>FLEET STATS</div>
      <div style='font-size:18px;'><b>{flagged}</b> Instances Flagged</div>
      <div style='font-size:14px; color:#2980b9;'>Avg Risk: {np.mean(probs):.2%}</div>
    </div>
    """

    idx_sorted = np.argsort(-probs)[:int(top_k)]
    top_df = pd.DataFrame({"InstanceID": idx_sorted, "FailureProb": probs[idx_sorted].round(4)})
    return kpi_html, top_df, make_distribution(probs, threshold, fleet_size)


def run_instance(instance_id, threshold):
    instance_id = int(max(0, min(int(instance_id), len(df) - 1)))
    row = df.iloc[[instance_id]]
    status, prob = get_prediction(row)

    status_html = f"""
    <div style='padding:12px; border-radius:12px; border:1px solid #eee; background:#fff; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
      <div style='color:#7f8c8d; font-size:12px;'>INSTANCE #{instance_id}</div>
      <div style='font-size:18px;'>Status: <b>{status}</b></div>
      <div style='font-size:14px;'>Prob: {prob:.2%}</div>
    </div>
    """

    variables = row.drop(columns=["Target"], errors="ignore").T.reset_index()
    variables.columns = ["Variable", "Value"]
    variables["Value"] = variables["Value"].astype(float).round(3)
    variables = variables.reindex(variables.Value.abs().sort_values(ascending=False).index)
    return status_html, make_gauge(prob, threshold), make_variable_scatter(variables), variables


def full_refresh(instance_id, threshold, top_k, fleet_size):
    f_kpi, t_df, d_fig = fleet_overview(threshold, top_k, fleet_size)
    s_html, g_fig, v_plot, v_tbl = run_instance(instance_id, threshold)
    return f_kpi, t_df, d_fig, s_html, g_fig, v_plot, v_tbl


# AI Agent Interface
def chat_html(history):
    if not history:
        return "<div id='agent_chat_box'><div class='small_meta'>System: Ready for queries...</div></div>"

    blocks = ["<div id='agent_chat_box'>"]
    for u, a in history:
        blocks.append(
            f"<div class='chat_turn'><div class='small_meta meta_user'>You</div><div class='bubble_user'>{html.escape(u)}</div></div>"
        )
        blocks.append(
            f"<div class='chat_turn'><div class='small_meta'>Agent</div><div class='bubble_agent'>{html.escape(a)}</div></div>"
        )
    blocks.append("</div>")
    return "\n".join(blocks)


def get_or_create_agent(agent_state):
    if agent_state is not None:
        return agent_state
    return WindTurbineMaintenanceAgent(df=df)


def agent_send(user_msg, history_state, agent_state, instance_id, fleet_size, threshold):
    history_state = history_state or []
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return chat_html(history_state), history_state, agent_state, ""

    instance_id = int(max(0, min(int(instance_id), len(df) - 1)))
    fleet_size = int(max(1, min(int(fleet_size), len(df))))
    threshold = float(max(0.0, min(float(threshold), 1.0)))

    agent = get_or_create_agent(agent_state)
    agent.set_context(instance_id=instance_id, fleet_size=fleet_size, threshold=threshold)

    try:
        answer = agent.chat(user_msg)
    except Exception as e:
        answer = f"⚠️ Agent error: {str(e)}"

    history_state.append((user_msg, answer))
    return chat_html(history_state), history_state, agent, ""


def agent_clear(agent_state):
    agent = get_or_create_agent(agent_state)
    agent.clear_history()
    return chat_html([]), [], agent, ""


def set_quick_question(choice, instance_id):
    mapping = {
        "Fleet summary": "Provide a fleet health summary including flagged count and average risk.",
        "Why risky? (this instance)": f"Analyze variables for instance {int(instance_id)}. Which variables are most abnormal and could explain the risk?",
        "Explain variable colors": "Explain the risk levels used in the variable scatter plot: OK, WARNING, and CRITICAL.",
        "What is High Risk Instances table?": "Explain what the 'High Risk Instances' table represents and how it is computed."
    }
    return mapping.get(choice, "")


# UI assembly
with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
    gr.Markdown("# Predictive Maintenance for Wind Turbine ")

    chat_state = gr.State([])
    agent_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            instance_id = gr.Slider(
                0, len(df) - 1, value=min(100, len(df) - 1), step=1,
                label="Instance Selector"
            )
            threshold = gr.Slider(0.1, 0.9, value=0.5, label="Alarm Sensitivity Threshold")
            fleet_size = gr.Slider(50, len(df), value=min(500, len(df)), label="Fleet Monitoring Sample Size")
            top_k = gr.Slider(5, 30, value=10, label="Risk Leaderboard (Top K)")
            refresh_btn = gr.Button("Update Dashboard", variant="primary")

            status_card = gr.HTML()
            fleet_kpi = gr.HTML()
            top_table = gr.DataFrame(label="High Risk Instances", interactive=False)

            with gr.Accordion("Raw Variable Data", open=False):
                variable_tbl = gr.DataFrame(interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("## Health Analytics")

            gauge_plot = gr.Plot(label="Failure Probability Gauge")
            variable_plot = gr.Plot(label="Variable Deviation Scatter")
            dist_plot = gr.Plot(label="Fleet Risk Distribution")

            # AI agent
            gr.Markdown("## AI Maintenance Assistant")

            chat_view = gr.HTML(chat_html([]))
            agent_in = gr.Textbox(
                placeholder="Ask about instance risk, variables (V1..V40), fleet stats, or dashboard components...",
                label="Assistant Query"
            )

            with gr.Row():
                quick = gr.Dropdown(
                    choices=["Fleet summary", "Why risky?", "Explain variable colors", "What is High Risk Instances table?"],
                    label="Quick Insights"
                )
                fill_btn = gr.Button("Fill Query")
                send_btn = gr.Button("Ask Agent", variant="primary")
                clear_btn = gr.Button("Reset Chat")


    # Event bindings
    refresh_btn.click(
        full_refresh,
        inputs=[instance_id, threshold, top_k, fleet_size],
        outputs=[fleet_kpi, top_table, dist_plot, status_card, gauge_plot, variable_plot, variable_tbl]
    )

    fill_btn.click(set_quick_question, [quick, instance_id], [agent_in])

    send_btn.click(
        agent_send,
        inputs=[agent_in, chat_state, agent_state, instance_id, fleet_size, threshold],
        outputs=[chat_view, chat_state, agent_state, agent_in]
    )

    agent_in.submit(
        agent_send,
        inputs=[agent_in, chat_state, agent_state, instance_id, fleet_size, threshold],
        outputs=[chat_view, chat_state, agent_state, agent_in]
    )

    clear_btn.click(
        agent_clear,
        inputs=[agent_state],
        outputs=[chat_view, chat_state, agent_state, agent_in]
    )


if __name__ == "__main__":
    demo.launch()