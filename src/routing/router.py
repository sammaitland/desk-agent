"""Routing: from a cost prediction to a configuration.

Three tiers, each a model plus a turn budget. The router reads a Prediction
and picks one. The rule is deliberately simple and readable, because a routing
policy nobody can explain is one nobody can audit — and because the literature
says simple policies match complex ones here.

    light     Haiku,  3 turns   single-tool lookups and definitions
    standard  Sonnet, 8 turns   the default; what the agent always was
    deep      Sonnet, 12 turns  multi-hop investigations

**Why the rule is conservative.** A router that sends a hard question to the
cheap model produces a shallow answer dressed as a saving. So `light` requires
high confidence *and* a predicted single tool *and* a low predicted cost. Any
doubt routes to `standard`. The bar the router must clear is "no worse on
held-out evals," and a conservative policy is how it clears it.

**What this is, in one sentence:** the trading system's spread-based
transaction-cost model applied to inference. Both replace a fixed cost
assumption with one conditioned on observable pre-trade quantities; both
predict the cost of the path, not the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.routing.predictor import CostPredictor, Prediction


@dataclass(frozen=True)
class Tier:
    name: str
    model: str
    max_turns: int
    compress: bool = True

    @property
    def label(self) -> str:
        return f"{self.name} ({self.model}, {self.max_turns} turns)"


TIERS: dict[str, Tier] = {
    "light": Tier("light", "claude-haiku-4-5-20251001", 3),
    "standard": Tier("standard", "claude-sonnet-4-6", 8),
    "deep": Tier("deep", "claude-sonnet-4-6", 12),
}

DEFAULT_TIER = TIERS["standard"]

# Thresholds. Tokens are total (in + out) for the whole run, pre-caching and
# pre-compression, as recorded on historical traces.
LIGHT_MAX_TOKENS = 9_000
LIGHT_MIN_CONFIDENCE = 0.45
DEEP_MIN_TOOLS = 3
DEEP_MIN_TOKENS = 25_000


@dataclass
class Decision:
    tier: Tier
    prediction: Prediction
    reason: str

    def as_dict(self) -> dict:
        return {"tier": self.tier.name, "model": self.tier.model,
                "max_turns": self.tier.max_turns, "reason": self.reason,
                "prediction": self.prediction.as_dict()}


def route(prediction: Prediction) -> Decision:
    """Pick a tier. Every branch states its reason, for the trace and the audit."""
    p = prediction

    if p.basis == "cold_start":
        return Decision(DEFAULT_TIER, p,
                        "no similar past queries — default tier")

    if (p.confidence >= LIGHT_MIN_CONFIDENCE
            and len(p.expected_tools) <= 1
            and p.expected_tokens <= LIGHT_MAX_TOKENS):
        return Decision(TIERS["light"], p,
                        f"high-confidence match to a single-tool query "
                        f"(~{p.expected_tokens:,} tokens)")

    if len(p.expected_tools) >= DEEP_MIN_TOOLS or p.expected_tokens >= DEEP_MIN_TOKENS:
        return Decision(TIERS["deep"], p,
                        f"expected multi-hop investigation "
                        f"({len(p.expected_tools)} tools, ~{p.expected_tokens:,} tokens)")

    return Decision(DEFAULT_TIER, p, "standard investigation")


class Router:
    """Predict, then decide. Holds the predictor so the trace store loads once."""

    def __init__(self, predictor: CostPredictor | None = None):
        self.predictor = predictor or CostPredictor.from_disk()

    def decide(self, question: str, exclude_question: str | None = None) -> Decision:
        return route(self.predictor.predict(question, exclude_question=exclude_question))

    @property
    def history_size(self) -> int:
        return len(self.predictor.runs)

    @property
    def corpus_report(self):
        """What the predictor loaded and what it rejected — so a thin corpus
        is visible rather than silently routing everything to default."""
        return self.predictor.report
