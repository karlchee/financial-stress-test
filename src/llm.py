"""LLM provider abstraction.

Single interface so the rest of the app does not need to know which provider
is in use. Switch providers via the LLM_PROVIDER env var.

Public API:
    get_provider()                  -> Provider
    propose_factor_shocks_tool()    -> tool spec dict (provider-agnostic)

Provider.chat(messages, system, tools) returns:
    {"text": str, "tool_calls": [{"name": str, "arguments": dict}, ...]}
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

from src.factors import FACTORS, FACTOR_NAMES


# ---- Tool schema (provider-agnostic) ----

def propose_factor_shocks_tool() -> dict:
    """Generate the propose_factor_shocks tool schema from FACTORS.

    Single source of truth: factor names, descriptions, units, and typical
    ranges all come from src/factors.py.
    """
    properties: dict[str, dict] = {}
    for f in FACTORS:
        lo, hi = f.typical_3m_range
        properties[f.name] = {
            "type": "number",
            "description": (
                f"{f.description}. Unit: {f.unit}. "
                f"Typical 3M range: [{lo}, {hi}]."
            ),
        }
    return {
        "name": "propose_factor_shocks",
        "description": (
            "Propose a 3-month factor shock vector representing the user's "
            "scenario. All 10 factors must be specified; use 0 for factors "
            "you believe are unaffected. The user can edit these in the UI "
            "before running the model."
        ),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(FACTOR_NAMES),
        },
    }


# ---- Provider interface ----

class Provider(ABC):
    name: str
    model_name: str

    @abstractmethod
    def chat(self, messages: list[dict], system: str,
             tools: list[dict] | None = None) -> dict:
        """Send a chat completion request.

        messages: list of {'role': 'user'|'assistant', 'content': str}
        system:   system prompt string
        tools:    list of tool specs in the provider-agnostic shape

        Returns: {"text": str, "tool_calls": [{"name": str, "arguments": dict}]}
        """
        ...


# ---- Gemini ----

class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-pro"):
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        self._genai = genai
        self.model_name = os.environ.get("GEMINI_MODEL", model)

    def chat(self, messages, system, tools=None):
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        gemini_tools = None
        if tools:
            gemini_tools = [{
                "function_declarations": [
                    {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"],
                    }
                    for t in tools
                ]
            }]

        model = self._genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system,
            tools=gemini_tools,
        )
        resp = model.generate_content(contents)

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for cand in (resp.candidates or []):
            for part in cand.content.parts:
                txt = getattr(part, "text", None)
                if txt:
                    text_parts.append(txt)
                fn = getattr(part, "function_call", None)
                if fn and fn.name:
                    tool_calls.append({
                        "name": fn.name,
                        "arguments": dict(fn.args) if fn.args else {},
                    })
        return {"text": "".join(text_parts), "tool_calls": tool_calls}


# ---- Anthropic ----

class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model_name = os.environ.get("ANTHROPIC_MODEL", model)

    def chat(self, messages, system, tools=None):
        anthropic_tools = None
        if tools:
            anthropic_tools = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"],
                }
                for t in tools
            ]
        resp = self._client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            system=system,
            messages=messages,
            tools=anthropic_tools or [],
        )
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "name": block.name,
                    "arguments": dict(block.input),
                })
        return {"text": "".join(text_parts), "tool_calls": tool_calls}


# ---- Groq ----

class GroqProvider(Provider):
    name = "groq"

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        import groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        self._client = groq.Groq(api_key=api_key)
        self.model_name = os.environ.get("GROQ_MODEL", model)

    def chat(self, messages, system, tools=None):
        groq_messages = [{"role": "system", "content": system}] + messages
        groq_tools = None
        if tools:
            groq_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"],
                    },
                }
                for t in tools
            ]
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=groq_messages,
            tools=groq_tools,
            max_tokens=4096,
        )
        msg = resp.choices[0].message
        tool_calls: list[dict] = []
        for tc in (msg.tool_calls or []):
            raw = tc.function.arguments
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            tool_calls.append({"name": tc.function.name, "arguments": args})
        return {"text": msg.content or "", "tool_calls": tool_calls}


# ---- Selector ----

def get_provider() -> Provider:
    """Return the provider selected by the LLM_PROVIDER env var.

    Default: gemini.
    """
    name = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if name == "gemini":
        return GeminiProvider()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "groq":
        return GroqProvider()
    raise ValueError(
        f"Unknown LLM_PROVIDER: {name!r}. Use one of: gemini, anthropic, groq."
    )
