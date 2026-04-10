"""Base types and interfaces for the weak supervision framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ABSTAIN = "__ABSTAIN__"


@dataclass(frozen=True)
class LFOutput:
    """Output of a single labeling function on a single mention.

    Attributes:
        lf_name: Identifier of the labeling function (e.g. "LF-D1").
        label: Predicted label string, or ABSTAIN if the LF cannot decide.
        confidence: LF-internal confidence in [0, 1]. Only meaningful when
            label != ABSTAIN.
        evidence_span: Substring of the mention that triggered the label.
        source_meta: Free-form metadata (section, pattern name, cue words …).
    """

    lf_name: str
    label: str
    confidence: float = 1.0
    evidence_span: str | None = None
    source_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_abstain(self) -> bool:
        return self.label == ABSTAIN


@dataclass(frozen=True)
class AggregatedLabel:
    """Result of aggregating multiple LF outputs for one mention × one task.

    Attributes:
        label: Winning label, or ABSTAIN if all LFs abstained.
        confidence: Aggregated confidence = score(winner) / sum(all scores).
        lf_coverage: Number of non-ABSTAIN LFs.
        lf_agreement: Fraction of non-ABSTAIN LFs that voted for the winner.
        supporting_lfs: Names of LFs that voted for the winning label.
        all_votes: Full vote record {label: weighted_score}.
        evidence_spans: Evidence spans from supporting LFs.
    """

    label: str
    confidence: float
    lf_coverage: int
    lf_agreement: float
    supporting_lfs: list[str]
    all_votes: dict[str, float]
    evidence_spans: list[str]


@dataclass(frozen=True)
class GateResult:
    """Quality gate assignment for a single mention × task.

    Attributes:
        gate_level: One of "G1" .. "G5" (the *strictest* gate the sample passes).
        passed_gates: List of all gates the sample passes (e.g. ["G1","G2","G3"]).
        reason: Human-readable explanation.
    """

    gate_level: str
    passed_gates: list[str]
    reason: str


MentionRecord = dict[str, Any]
