# tools.py
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd
from model_logic import predict_proba


# Tool context
@dataclass
class ToolContext:
    df: pd.DataFrame
    alarm_threshold: float = 0.50
    fleet_size: int = 500
    current_instance_id: Optional[int] = None

    def set_current_instance(self, instance_id: Optional[int]):
        if instance_id is None:
            self.current_instance_id = None
            return
        instance_id = int(instance_id)
        instance_id = max(0, min(instance_id, len(self.df) - 1))
        self.current_instance_id = instance_id

    def set_fleet_size(self, fleet_size: int):
        fleet_size = int(fleet_size)
        fleet_size = max(1, min(fleet_size, len(self.df)))
        self.fleet_size = fleet_size

    def set_threshold(self, thr: float):
        thr = float(thr)
        self.alarm_threshold = max(0.0, min(thr, 1.0))


def get_tool_context(df: pd.DataFrame) -> ToolContext:
    return ToolContext(df=df)


# Tool implementation
def _safe_instance_id(ctx: ToolContext, instance_id: Optional[int]) -> int:
    if instance_id is None:
        if ctx.current_instance_id is None:
            return 0
        return int(ctx.current_instance_id)
    instance_id = int(instance_id)
    return max(0, min(instance_id, len(ctx.df) - 1))


def get_failure_probability(ctx: ToolContext, instance_id: Optional[int] = None) -> str:
    i = _safe_instance_id(ctx, instance_id)
    row = ctx.df.iloc[[i]].drop(columns=["Target"], errors="ignore")
    prob = float(predict_proba(row)[0])
    status = "Failure Predicted" if prob >= ctx.alarm_threshold else "Healthy"

    return json.dumps({
        "instance_id": i,
        "failure_probability": prob,
        "alarm_threshold": float(ctx.alarm_threshold),
        "status": status
    })


def get_fleet_summary(ctx: ToolContext, fleet_size: Optional[int] = None, top_k: int = 10) -> str:
    if fleet_size is not None:
        ctx.set_fleet_size(fleet_size)

    top_k = int(max(1, min(int(top_k), 100)))

    df_sub = ctx.df.iloc[:ctx.fleet_size].drop(columns=["Target"], errors="ignore")
    probs = predict_proba(df_sub).astype(float)

    total = int(len(probs))
    flagged = int((probs >= ctx.alarm_threshold).sum())
    avg = float(np.mean(probs))
    p95 = float(np.quantile(probs, 0.95))
    maxp = float(np.max(probs))

    top_idx = np.argsort(-probs)[:top_k].tolist()
    top_list = [{"instance_id": int(i), "prob": float(probs[i])} for i in top_idx]

    return json.dumps({
        "fleet_size": total,
        "alarm_threshold": float(ctx.alarm_threshold),
        "flagged_count": flagged,
        "avg_risk": avg,
        "p95_risk": p95,
        "max_risk": maxp,
        "top_risky": top_list
    })


def get_top_deviating_variables(ctx: ToolContext, instance_id: Optional[int] = None, top_n: int = 10) -> str:
    i = _safe_instance_id(ctx, instance_id)
    top_n = int(max(1, min(int(top_n), 40)))

    row = ctx.df.iloc[i]
    variables = [c for c in ctx.df.columns if c.startswith("V")]
    vals = pd.to_numeric(row[variables], errors="coerce")

    out = (
        pd.DataFrame({"variable": variables, "value": vals.values})
        .dropna()
        .assign(abs_value=lambda d: d["value"].abs())
        .sort_values("abs_value", ascending=False)
        .head(top_n)
    )

    return json.dumps({
        "instance_id": i,
        "top_n": top_n,
        "top_deviations": [
            {"variable": r.variable, "value": float(r.value), "abs_value": float(r.abs_value)}
            for r in out.itertuples(index=False)
        ],
        "note": "Variables are standardized/normalized (approx mean~0). Large |value| indicates unusual behavior."
    })


def get_variable_info(
    ctx: ToolContext,
    variable: str,
    instance_id: Optional[int] = None,
    fleet_size: Optional[int] = None
) -> str:
    variable = str(variable).strip().upper()
    if not variable.startswith("V"):
        return json.dumps({"error": "Variable must be like V15, V1, V40."})

    if variable not in ctx.df.columns:
        return json.dumps({"error": f"Variable '{variable}' not found in dataset columns."})

    if fleet_size is not None:
        ctx.set_fleet_size(fleet_size)

    i = _safe_instance_id(ctx, instance_id)

    row_val = pd.to_numeric(ctx.df.loc[i, variable], errors="coerce")
    if pd.isna(row_val):
        return json.dumps({
            "instance_id": i,
            "variable": variable,
            "value": None,
            "note": "Value is NaN for this instance."
        })

    val = float(row_val)

    sub = ctx.df.iloc[:ctx.fleet_size][variable]
    sub = pd.to_numeric(sub, errors="coerce").dropna()

    if len(sub) < 10:
        return json.dumps({
            "instance_id": i,
            "variable": variable,
            "value": val,
            "fleet_size": int(len(sub)),
            "note": "Not enough data to compute fleet statistics."
        })

    mean = float(sub.mean())
    std_val = float(sub.std(ddof=1))
    std = std_val if std_val > 0 else 0.0

    z = float((val - mean) / std) if std > 0 else None
    abs_z = float(abs(z)) if z is not None else None

    warn_thr = 1.5
    crit_thr = 2.5
    if abs_z is None:
        level = "UNKNOWN"
    elif abs_z >= crit_thr:
        level = "CRITICAL"
    elif abs_z >= warn_thr:
        level = "WARNING"
    else:
        level = "OK"

    q05 = float(np.quantile(sub, 0.05))
    q50 = float(np.quantile(sub, 0.50))
    q95 = float(np.quantile(sub, 0.95))

    return json.dumps({
        "instance_id": i,
        "variable": variable,
        "value": val,
        "fleet_size": int(ctx.fleet_size),
        "fleet_mean": mean,
        "fleet_std": std,
        "z_score": z,
        "abs_z_score": abs_z,
        "risk_level": level,
        "quantiles": {"q05": q05, "q50": q50, "q95": q95},
        "note": "Variables are standardized/normalized; z-score indicates how unusual this variable is vs fleet subset."
    })


def get_dataset_overview(ctx: ToolContext) -> str:
    df = ctx.df
    return json.dumps({
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "variable_columns": [c for c in df.columns if c.startswith("V")],
        "has_target": bool("Target" in df.columns)
    })


def explain_fleet_risk_distribution(ctx: ToolContext) -> str:
    return (
        "How to read Fleet Risk Distribution:\n"
        "1) X-axis = Failure Probability (0..1).\n"
        "2) Y-axis = Instance Count.\n"
        "3) The dashed vertical line is Alarm Threshold.\n"
        "   - Right of the line: flagged instances (needs attention)\n"
        "   - Left of the line: considered normal under current threshold\n"
        "4) Lower threshold -> more alarms (more sensitive).\n"
        "   Higher threshold -> fewer alarms (less sensitive).\n"
    )


def get_dashboard_help(ctx: ToolContext) -> str:
    return (
        "Dashboard Help:\n"
        "- Fleet Stats card:\n"
        "  - 'Instances Flagged' = number of rows in the selected fleet subset where failure_probability >= threshold.\n"
        "  - 'Avg Risk' = mean failure_probability across the selected fleet subset.\n"
        "- High Risk Instances table:\n"
        "  - Top-K rows (InstanceID) with the highest failure_probability in the fleet subset.\n"
        "  - Columns: InstanceID (row index), FailureProb (0..1).\n"
        "- Fleet Monitoring Sample Size:\n"
        "  - Number of rows from the start used for fleet analytics.\n"
        "- Risk Leaderboard (Top K):\n"
        "  - How many rows appear in the High Risk Instances table.\n"
        "- Alarm Threshold:\n"
        "  - Probability cutoff used to count flagged instances and shown as a dashed red line on the histogram.\n"
        "- Failure Probability Gauge:\n"
        "  - Selected instance failure_probability compared to threshold.\n"
        "- Normalized Variable Deviations scatter:\n"
        "  - Shows V1..V40 values for the selected instance.\n"
        "  - Colored by abs(value): OK abs<1.5, WARNING 1.5<=abs<2.5, CRITICAL abs>=2.5.\n"
    )


# Tool registry
TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_failure_probability",
            "description": "Get REAL failure probability and status for one selected instance using the XGBoost model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instance_id": {"type": "integer", "description": "Row index (0-based). Optional if already selected."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_fleet_summary",
            "description": "Get REAL fleet summary statistics and top risky instances for a chosen subset size.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fleet_size": {"type": "integer", "description": "Fleet subset size (rows from start)."},
                    "top_k": {"type": "integer", "description": "How many top risky instances to return."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_deviating_variables",
            "description": "Get REAL variable deviations: top variables by absolute value for a chosen instance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instance_id": {"type": "integer", "description": "Row index (0-based). Optional if already selected."},
                    "top_n": {"type": "integer", "description": "How many variables to return (1..40)."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_variable_info",
            "description": "Get REAL info about one variable (e.g., V15): value for current instance and how abnormal it is vs fleet (mean/std/z-score + OK/WARNING/CRITICAL).",
            "parameters": {
                "type": "object",
                "properties": {
                    "variable": {"type": "string", "description": "Variable name like V15."},
                    "instance_id": {"type": "integer", "description": "Row index (0-based). Optional if already selected."},
                    "fleet_size": {"type": "integer", "description": "Fleet subset size used for distribution stats (optional)."}
                },
                "required": ["variable"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_dataset_overview",
            "description": "Get dataset overview (rows, columns, missing values, variable columns).",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_fleet_risk_distribution",
            "description": "Explain precisely how to read the Fleet Risk Distribution chart (no hallucinations).",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_dashboard_help",
            "description": "Explain dashboard components (tables, plots, sliders) in a factual way.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
]


def execute_tool(ctx: ToolContext, name: str, args: Dict[str, Any]) -> str:
    if name == "get_failure_probability":
        return get_failure_probability(ctx, instance_id=args.get("instance_id"))

    if name == "get_fleet_summary":
        return get_fleet_summary(
            ctx,
            fleet_size=args.get("fleet_size"),
            top_k=args.get("top_k", 10),
        )

    if name == "get_top_deviating_variables":
        return get_top_deviating_variables(
            ctx,
            instance_id=args.get("instance_id"),
            top_n=args.get("top_n", 10)
        )

    if name == "get_variable_info":
        return get_variable_info(
            ctx,
            variable=args.get("variable", ""),
            instance_id=args.get("instance_id"),
            fleet_size=args.get("fleet_size"),
        )

    if name == "get_dataset_overview":
        return get_dataset_overview(ctx)

    if name == "explain_fleet_risk_distribution":
        return explain_fleet_risk_distribution(ctx)

    if name == "get_dashboard_help":
        return get_dashboard_help(ctx)

    return json.dumps({"error": f"Unknown tool: {name}"})