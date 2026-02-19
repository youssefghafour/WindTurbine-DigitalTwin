# wind_agent.py
"""
Ollama LLM agent with tool calling for Wind Turbine predictive maintenance.
Model: qwen3:0.6b (local, lightweight)
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from ollama import Client

from tools import TOOL_DEFINITIONS, execute_tool, get_tool_context, ToolContext


SYSTEM_PROMPT = """You are an expert predictive maintenance assistant for wind turbine generators.

CRITICAL RULES:
1) NEVER invent or guess numbers. If a question needs data, call a tool.
2) For failure probability, fleet stats, or sensor deviations, you MUST call the correct tool.
3) Use tool outputs as the single source of truth.
4) Keep the answer short, correct, and actionable.

PROJECT CONTEXT:
- Model: XGBoost classifier
- Output: failure_probability in [0,1]
- Dashboard colors for telemetry scatter:
  OK = light green, WARNING = orange, CRITICAL = black
- Telemetry values are standardized/normalized features (mean approx 0).
"""

@dataclass
class Message:
    role: str
    content: str
    tool_calls: Optional[List[Dict]] = None


@dataclass
class ConversationState:
    messages: List[Message] = field(default_factory=list)
    model: str = "qwen3:0.6b"
    tool_context: Optional[ToolContext] = None


class WindTurbineMaintenanceAgent:

    def __init__(self, df, model: str = "qwen3:0.6b", host: str = "http://localhost:11434"):
        self.client = Client(host=host)
        self.model = model
        self.conversation = ConversationState(model=model, tool_context=get_tool_context(df))
        self._check_model()

    def _check_model(self):
        try:
            models = self.client.list()
            names = [m.model for m in models.models]
            if not any(self.model == n or n.startswith(self.model.split(":")[0]) for n in names):
                print(f"Warning: model '{self.model}' may not be available.")
                print(f"Available models: {names}")
                print(f"Run: ollama pull {self.model}")
        except Exception as e:
            print(f"Warning: Could not check Ollama models: {e}")
            print("Make sure Ollama is running: ollama serve")

    def set_context(self, instance_id: Optional[int] = None, fleet_size: Optional[int] = None, threshold: Optional[float] = None):
        ctx = self.conversation.tool_context
        if instance_id is not None:
            ctx.set_current_instance(instance_id)
        if fleet_size is not None:
            ctx.set_fleet_size(fleet_size)
        if threshold is not None:
            ctx.set_threshold(threshold)

    def clear_history(self):
        self.conversation.messages = []

    def _format_messages_for_ollama(self) -> List[Dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # include context into system message
        ctx = self.conversation.tool_context
        if ctx.current_instance_id is not None:
            messages[0]["content"] += f"\n\nCURRENT CONTEXT:\n- Current InstanceID: {ctx.current_instance_id}\n- Alarm Threshold: {ctx.alarm_threshold}\n- Fleet Sample Size: {ctx.fleet_size}"

        for m in self.conversation.messages:
            if m.role == "user":
                messages.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                am = {"role": "assistant", "content": m.content}
                if m.tool_calls:
                    am["tool_calls"] = m.tool_calls
                messages.append(am)
            elif m.role == "tool":
                messages.append({"role": "tool", "content": m.content})

        return messages

    def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        results = []
        ctx = self.conversation.tool_context

        for call in tool_calls:
            func = call.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")

            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}

            result = execute_tool(ctx, name, args)

            results.append({
                "tool_call_id": call.get("id", ""),
                "name": name,
                "result": result,
            })

        return results

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
            return f"Error communicating with Ollama: {e}"

        message = response.message

        # Tool-calling loop
        max_iterations = 5
        iteration = 0

        while message.tool_calls and iteration < max_iterations:
            iteration += 1

            tool_results = self._execute_tool_calls(message.tool_calls)

            # store assistant tool call request
            self.conversation.messages.append(Message(
                role="assistant",
                content=message.content or "",
                tool_calls=message.tool_calls
            ))

            # add each tool result as "tool" message
            for r in tool_results:
                self.conversation.messages.append(Message(role="tool", content=r["result"]))
                messages.append({"role": "tool", "content": r["result"]})

            # ask model again with tool outputs
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": message.tool_calls,
            })

            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                )
                message = response.message
            except Exception as e:
                return f"Error during tool execution: {e}"

        final_content = message.content or "I couldn't generate a response."

        self.conversation.messages.append(Message(role="assistant", content=final_content))
        return final_content


# Convenience function for your Gradio app
def ask_agent(df, user_question: str, instance_id: Optional[int] = None, fleet_size: Optional[int] = None, threshold: Optional[float] = None) -> str:
    agent = WindTurbineMaintenanceAgent(df=df, model="qwen3:0.6b")
    agent.set_context(instance_id=instance_id, fleet_size=fleet_size, threshold=threshold)
    return agent.chat(user_question)
