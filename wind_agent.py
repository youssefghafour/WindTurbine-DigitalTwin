# wind_agent.py
"""
Ollama LLM agent with tool calling for Wind Turbine predictive maintenance dashboard.

Key guarantees:
- Keeps conversation state
- Robust tool-calling loop
- Dynamic system prompt with current dashboard context
- Never outputs <think> tags
- Terminology guardrail: NEVER assumes rows = physical turbines

Works with tools.py:
- TOOL_DEFINITIONS
- execute_tool(ctx, name, args)
- get_tool_context(df) -> ToolContext
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Generator, Any

from ollama import Client

from tools import (
    TOOL_DEFINITIONS,
    execute_tool,
    get_tool_context,
    ToolContext,
)

# Output cleanup/guardrails
def _strip_think(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _enforce_terminology(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\bturbines\b", "instances", text, flags=re.IGNORECASE)
    text = re.sub(r"\bturbine\b", "instance", text, flags=re.IGNORECASE)

    return text


def _finalize_output(text: str) -> str:
    text = _strip_think(text)
    text = _enforce_terminology(text)
    return text.strip()


# System prompt
SYSTEM_PROMPT = """You are an expert predictive maintenance assistant for a Wind Turbine Digital Twin dashboard.

OUTPUT RULES (VERY IMPORTANT):
- Do NOT output <think> or </think> or any hidden reasoning.
- Only output the final answer to the user.

CRITICAL RULES:
1) NEVER invent or guess numbers.
2) If the user asks for failure probability, fleet statistics, dataset facts, or variable values, you MUST call the relevant tool first.
3) Tool outputs are the single source of truth for numbers. You may interpret them, but never fabricate values.
4) You MAY answer fixed dashboard UI/UX questions without tools (definitions below are fixed).
   If unsure about a dashboard element, call get_dashboard_help.
5) If the user asks about a specific variable (e.g., "V15"), you MUST call get_variable_info first (if available).
6) If asked about the meaning of colors or "Normalized Variable Deviations", answer with the thresholds above.
Do NOT mention failure_probability threshold when explaining the scatter colors.

IMPORTANT TERMINOLOGY (MUST FOLLOW):
- The dataset contains "instances/rows/samples". Do NOT assume a row = a physical turbine.
- Never say "turbine(s)" to mean dataset rows. Use: "instances", "rows", or "samples".
- "Avg Risk" is NOT the % of rows at high risk. It is the MEAN predicted failure_probability across the fleet subset.

DASHBOARD DEFINITIONS (YOU MUST KNOW THESE):
- "Fleet Stats" card:
  - "Instances Flagged" = count of rows in the fleet subset where failure_probability >= threshold.
  - "Avg Risk" = mean(failure_probability) across the fleet subset.

- "High Risk Instances" table:
  - Top-K rows with the highest failure_probability within the chosen fleet subset.
  - Columns:
    - InstanceID: row index in the dataset (0-based).
    - FailureProb: predicted failure probability (0..1).
  - Updates when:
    - Fleet Monitoring Sample Size changes (subset size)
    - Risk Leaderboard (Top K) changes
    - Threshold changes flagged count, but not necessarily ranking.

- "Fleet Risk Distribution" plot:
  - Histogram of failure_probability for the fleet subset.
  - Red dashed line = alarm threshold.
  - Right of threshold = flagged rows.

- "Failure Probability Gauge":
  - Shows selected row’s failure_probability compared to threshold.

- "Normalized Variable Deviations" scatter:
  - The plot shows one point per variable V1..V40 for the selected instance.
  - Y value = the normalized/standardized value of that variable for that instance (around mean~0).
  - Color is based ONLY on ABS(value) using TWO thresholds:
    - OK (green): abs(value) < 1.5
    - WARNING (orange): 1.5 <= abs(value) < 2.5
    - CRITICAL (black): abs(value) >= 2.5
  - This plot is NOT related to the failure_probability threshold.
  - Do NOT claim the whole instance is OK/Warning/Critical based only on this plot.
    Instead say: "These variables are OK/Warning/Critical" and list them if needed.

SLIDERS/CONTROLS:
- Instance Selector (Instance ID): which row is inspected.
- Alarm Sensitivity Threshold: probability cutoff for flagging rows.
- Fleet Monitoring Sample Size: number of rows from the start used for fleet analytics.
- Risk Leaderboard Top K: number of rows shown in High Risk Instances.

PROJECT CONTEXT:
- Model: XGBoost classifier
- Output: failure_probability in [0, 1]
- Variables V1–V40 are standardized/normalized (mean approx 0).

RESPONSE STYLE:
- Clear, short, and dashboard-aware.
- Use bullets for explanations and next steps.
"""


# Conversation data structures
@dataclass
class Message:
    role: str  # "user", "assistant", "system", or "tool"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class ConversationState:
    messages: List[Message] = field(default_factory=list)
    model: str = "qwen3:0.6b"
    tool_context: Optional[ToolContext] = None

    def __post_init__(self):
        if self.tool_context is None:
            raise ValueError("tool_context must be provided (use get_tool_context(df)).")


# Agent implementation
class WindTurbineMaintenanceAgent:
    def __init__(
        self,
        df,
        model: str = "qwen3:0.6b",
        host: str = "http://localhost:11434",
        max_tool_iterations: int = 5,
    ):
        self.model = model
        self.client = Client(host=host)
        self.max_tool_iterations = int(max(1, max_tool_iterations))
        self.conversation = ConversationState(
            model=model,
            tool_context=get_tool_context(df),
        )
        self._check_model_available()

    # Utilities
    def _check_model_available(self) -> None:
        try:
            models = self.client.list()
            model_names = [m.model for m in models.models]
            available = any(
                self.model == name or name.startswith(self.model.split(":")[0])
                for name in model_names
            )
            if not available:
                print(f"Warning: Model '{self.model}' may not be available in Ollama.")
                print(f"Available models: {model_names}")
                print(f"Run: ollama pull {self.model}")
        except Exception as e:
            print(f"Warning: Could not check Ollama models: {e}")
            print("Make sure Ollama is running: ollama serve")

    def clear_history(self) -> None:
        self.conversation.messages = []

    def set_context(
        self,
        instance_id: Optional[int] = None,
        fleet_size: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> None:
        ctx = self.conversation.tool_context
        if ctx is None:
            return
        if instance_id is not None:
            ctx.set_current_instance(instance_id)
        if fleet_size is not None:
            ctx.set_fleet_size(fleet_size)
        if threshold is not None:
            ctx.set_threshold(threshold)

    # prompt formatting
    def _build_system_prompt(self) -> str:
        system_content = SYSTEM_PROMPT
        ctx = self.conversation.tool_context

        if ctx is not None:
            extra = []
            if ctx.current_instance_id is not None:
                extra.append(f"- Current InstanceID: {ctx.current_instance_id}")
            extra.append(f"- Alarm Threshold: {ctx.alarm_threshold}")
            extra.append(f"- Fleet Sample Size: {ctx.fleet_size}")
            system_content += "\n\nCURRENT DASHBOARD CONTEXT:\n" + "\n".join(extra)

        return system_content

    def _format_messages_for_ollama(self) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self._build_system_prompt()}]

        for msg in self.conversation.messages:
            if msg.role == "assistant":
                item: Dict[str, Any] = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    item["tool_calls"] = msg.tool_calls
                messages.append(item)
            else:
                messages.append({"role": msg.role, "content": msg.content})

        return messages

    # Tool calling
    @staticmethod
    def _safe_json_loads(maybe_json: Any) -> Dict[str, Any]:
        if isinstance(maybe_json, dict):
            return maybe_json
        if not isinstance(maybe_json, str):
            return {}
        try:
            obj = json.loads(maybe_json)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        ctx = self.conversation.tool_context

        for call in tool_calls or []:
            func = call.get("function", {}) or {}
            name = func.get("name", "") or ""
            args = self._safe_json_loads(func.get("arguments", "{}"))

            try:
                tool_output = execute_tool(ctx, name, args)
            except Exception as e:
                tool_output = json.dumps(
                    {"error": f"Tool '{name}' failed", "details": str(e)},
                    ensure_ascii=False,
                )

            results.append({
                "tool_call_id": call.get("id", ""),
                "name": name,
                "result": tool_output,
            })

        return results

    # Main chat
    def chat(self, user_message: str) -> str:
        self.conversation.messages.append(Message(role="user", content=user_message))
        messages = self._format_messages_for_ollama()

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )
        except Exception as e:
            return _finalize_output(f"Error communicating with Ollama: {e}")

        message = response.message
        iteration = 0

        while getattr(message, "tool_calls", None) and iteration < self.max_tool_iterations:
            iteration += 1
            tool_calls = message.tool_calls

            # Store assistant tool request
            self.conversation.messages.append(
                Message(role="assistant", content=message.content or "", tool_calls=tool_calls)
            )

            # Execute tools
            tool_results = self._execute_tool_calls(tool_calls)

            # Append tool outputs
            for r in tool_results:
                self.conversation.messages.append(Message(role="tool", content=r["result"]))
                messages.append({"role": "tool", "content": r["result"]})

            # Append assistant tool-call message to messages
            messages.append({"role": "assistant", "content": message.content or "", "tool_calls": tool_calls})

            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                )
                message = response.message
            except Exception as e:
                return _finalize_output(f"Error during tool execution: {e}")

        final_content = _finalize_output(message.content or "I couldn't generate a response.")
        self.conversation.messages.append(Message(role="assistant", content=final_content))
        return final_content

    # Streaming
    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        self.conversation.messages.append(Message(role="user", content=user_message))
        messages = self._format_messages_for_ollama()

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )
        except Exception as e:
            yield _finalize_output(f"Error communicating with Ollama: {e}")
            return

        message = response.message
        iteration = 0

        while getattr(message, "tool_calls", None) and iteration < self.max_tool_iterations:
            iteration += 1
            tool_calls = message.tool_calls

            self.conversation.messages.append(
                Message(role="assistant", content=message.content or "", tool_calls=tool_calls)
            )
            tool_results = self._execute_tool_calls(tool_calls)

            for r in tool_results:
                self.conversation.messages.append(Message(role="tool", content=r["result"]))
                messages.append({"role": "tool", "content": r["result"]})

            messages.append({"role": "assistant", "content": message.content or "", "tool_calls": tool_calls})

            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )
            message = response.message

        # Stream final answer
        messages = self._format_messages_for_ollama()
        full = ""
        try:
            for chunk in self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
            ):
                if chunk.message and chunk.message.content:
                    text = chunk.message.content
                    full += text
                    yield _finalize_output(text)  # stream cleaned chunks
        except Exception as e:
            yield _finalize_output(f"\nError during streaming: {e}")
            return

        full = _finalize_output(full)
        self.conversation.messages.append(Message(role="assistant", content=full))