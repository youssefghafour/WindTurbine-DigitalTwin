# tools.py
"""
Tool definitions + execution for Wind Turbine Predictive Maintenance agent.
These tools return REAL numbers from:
- dataset (df)
- XGBoost model (model_logic.py)
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd
from model_logic import predict_proba


# Tool Context
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


# Tool Implementations
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


def get_fleet_summary(ctx: ToolContext, fleet_size: Optional[int] = None) -> str:
    if fleet_size is not None:
        ctx.set_fleet_size(fleet_size)

    df_sub = ctx.df.iloc[:ctx.fleet_size].drop(columns=["Target"], errors="ignore")
    probs = predict_proba(df_sub).astype(float)
    total = int(len(probs))
    flagged = int((probs >= ctx.alarm_threshold).sum())
    avg = float(np.mean(probs))
    p95 = float(np.quantile(probs, 0.95))
    maxp = float(np.max(probs))

    # top risky indices within the subset
    top_idx = np.argsort(-probs)[:10].tolist()
    top_list = [{"instance_id": int(i), "prob": float(probs[i])} for i in top_idx]

    return json.dumps({
        "fleet_size": total,
        "alarm_threshold": float(ctx.alarm_threshold),
        "flagged_count": flagged,
        "avg_risk": avg,
        "p95_risk": p95,
        "max_risk": maxp,
        "top10_risky": top_list
    })


def get_top_deviating_sensors(ctx: ToolContext, instance_id: Optional[int] = None, top_n: int = 10) -> str:
    i = _safe_instance_id(ctx, instance_id)
    top_n = int(max(1, min(int(top_n), 40)))

    row = ctx.df.iloc[i]
    sensors = [c for c in ctx.df.columns if c.startswith("V")]
    vals = pd.to_numeric(row[sensors], errors="coerce")

    out = (
        pd.DataFrame({"sensor": sensors, "value": vals.values})
        .dropna()
        .assign(abs_value=lambda d: d["value"].abs())
        .sort_values("abs_value", ascending=False)
        .head(top_n)
    )

    return json.dumps({
        "instance_id": i,
        "top_n": top_n,
        "top_deviations": [
            {"sensor": r.sensor, "value": float(r.value), "abs_value": float(r.abs_value)}
            for r in out.itertuples(index=False)
        ],
        "note": "Values are standardized/normalized features (approx mean~0). Large |value| indicates unusual behavior."
    })


def get_dataset_overview(ctx: ToolContext) -> str:
    df = ctx.df
    return json.dumps({
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "feature_columns": [c for c in df.columns if c.startswith("V")],
        "has_target": bool("Target" in df.columns)
    })


def explain_fleet_risk_distribution(ctx: ToolContext) -> str:
    return (
        "How to read Fleet Risk Distribution:\n"
        "1) X-axis = Failure Probability (0..1). Near 0 means low risk, near 1 means high risk.\n"
        "2) Y-axis = Count of instances in each probability bin.\n"
        "3) The dashed vertical line is Alarm Threshold.\n"
        "   - Right of the line: flagged (needs attention)\n"
        "   - Left of the line: considered normal under current threshold\n"
        "4) If most bars are near 0 -> fleet mostly healthy.\n"
        "   A tail/cluster near 1 -> a few high-risk cases.\n"
        "5) Lower threshold -> more alarms (safer, more false positives).\n"
        "   Higher threshold -> fewer alarms (risk missing failures).\n"
    )


# Tool Registry
TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_failure_probability",
            "description": "Get REAL failure probability and status for one turbine instance using the XGBoost model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instance_id": {"type": "integer", "description": "Row index of the instance (0-based). Optional if already selected."}
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
                    "fleet_size": {"type": "integer", "description": "Number of instances used for the fleet distribution (subset from start)."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_deviating_sensors",
            "description": "Get REAL telemetry deviations: top sensors by absolute value for a chosen instance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instance_id": {"type": "integer", "description": "Row index of the instance (0-based). Optional if already selected."},
                    "top_n": {"type": "integer", "description": "How many sensors to return (1..40)."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_dataset_overview",
            "description": "Get dataset overview (rows, columns, missing values, feature columns).",
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
]


def execute_tool(ctx: ToolContext, name: str, args: Dict[str, Any]) -> str:
    if name == "get_failure_probability":
        return get_failure_probability(ctx, instance_id=args.get("instance_id"))
    if name == "get_fleet_summary":
        return get_fleet_summary(ctx, fleet_size=args.get("fleet_size"))
    if name == "get_top_deviating_sensors":
        return get_top_deviating_sensors(ctx, instance_id=args.get("instance_id"), top_n=args.get("top_n", 10))
    if name == "get_dataset_overview":
        return get_dataset_overview(ctx)
    if name == "explain_fleet_risk_distribution":
        return explain_fleet_risk_distribution(ctx)

    return json.dumps({"error": f"Unknown tool: {name}"})
