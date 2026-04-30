"""Narrator agent — converts the full decision trace into natural language.

Uses the Anthropic SDK with prompt caching. The system prompt (analyst
persona + output spec) is cached so that repeated invocations during a
session pay only output tokens.
"""
from __future__ import annotations

import json
import os
from typing import Any

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None  # type: ignore

from ..schemas import OracleResult

SYSTEM_PROMPT = """You are an elite cricket analyst writing a single, decision-grade
match brief for an IPL franchise's strategy desk. You receive a structured
JSON trace from a multi-agent optimisation system. Your job is to translate
it into a tight natural-language verdict.

REQUIREMENTS
1. Open with a one-sentence fixture summary: home vs away at venue, date.
2. Describe the predicted opponent XI in one paragraph, naming 2-3 key
   threats and the bowling phase profile.
3. Recommend our XI as a numbered list (1-11). For each pick, give one
   short reason grounded in the trace (matchup, role, conditions, form).
4. State the chosen optimisation mode (deterministic / robust / bayesian)
   and why the strategist chose it.
5. Quote the win probability with its 95% confidence interval, and the
   recommended toss call with rationale.
6. Close with a one-paragraph 'tactical risks' section: what could
   invalidate this plan (dew, opposition swap, injury), and the smallest
   override that would change the answer.
7. If `decision_trace.analyst_enrichment.available` is true, incorporate the
   analyst context naturally: one team-level insight, one opponent-level
   insight, and one fixture-specific watchout.

STYLE
- 350-500 words total, no headers, no bullet sections except the XI list.
- Use the language of professional cricket analysis. Be specific, not
  generic. Cite numbers from the trace (par score, win prob, etc.).
- Do NOT fabricate stats absent from the trace.
- Treat enrichment commentary as guidance; never present it as guaranteed fact.
"""


class NarratorAgent:
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", max_tokens: int = 1200):
        self.model = model
        self.max_tokens = max_tokens
        self._client: Anthropic | None = None

    def _get_client(self) -> Anthropic:
        if self._client is not None:
            return self._client
        if Anthropic is None:
            raise RuntimeError("anthropic SDK not installed. `pip install anthropic`.")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Pass --no-narrative to skip the LLM call."
            )
        self._client = Anthropic(api_key=api_key)
        return self._client

    def narrate(self, result_without_narrative: dict[str, Any]) -> str:
        trace = json.dumps(result_without_narrative, default=str, indent=2)
        msg = self._get_client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Here is the decision trace:\n\n{trace}\n\nWrite the brief now.",
                        }
                    ],
                }
            ],
        )
        chunks = [block.text for block in msg.content if getattr(block, "type", None) == "text"]
        return "\n".join(chunks).strip()


def build_trace(result: OracleResult) -> dict[str, Any]:
    """Strip the recursive narrative field and produce a compact dict."""
    payload = result.model_dump(mode="json")
    payload.pop("narrative", None)
    return payload
