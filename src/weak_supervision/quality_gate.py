from __future__ import annotations

from src.weak_supervision.base import ABSTAIN, AggregatedLabel, GateResult

GATE_DEFINITIONS = {
    "G1": {"min_confidence": 0.0, "min_coverage": 1, "min_agreement": 0.0},
    "G2": {"min_confidence": 0.7, "min_coverage": 1, "min_agreement": 0.0},
    "G3": {"min_confidence": 0.0, "min_coverage": 2, "min_agreement": 0.0},
    "G4": {"min_confidence": 0.0, "min_coverage": 1, "min_agreement": 0.8},
    "G5": {"min_confidence": 0.8, "min_coverage": 2, "min_agreement": 0.0},
}

GATE_ORDER = ["G1", "G2", "G3", "G4", "G5"]


def evaluate_gate(agg: AggregatedLabel) -> GateResult:
    if agg.label == ABSTAIN:
        return GateResult(
            gate_level="REJECTED",
            passed_gates=[],
            reason="all LFs abstained",
        )

    passed = []
    for gate_name in GATE_ORDER:
        criteria = GATE_DEFINITIONS[gate_name]
        if (
            agg.confidence >= criteria["min_confidence"]
            and agg.lf_coverage >= criteria["min_coverage"]
            and agg.lf_agreement >= criteria["min_agreement"]
        ):
            passed.append(gate_name)

    if not passed:
        return GateResult(
            gate_level="REJECTED",
            passed_gates=[],
            reason=f"conf={agg.confidence:.2f} cov={agg.lf_coverage} agr={agg.lf_agreement:.2f} — no gate passed",
        )

    strictest = passed[-1]
    return GateResult(
        gate_level=strictest,
        passed_gates=passed,
        reason=f"conf={agg.confidence:.2f} cov={agg.lf_coverage} agr={agg.lf_agreement:.2f}",
    )


def filter_by_gate(
    records: list[dict],
    gate_level: str,
    passed_gates_key: str = "passed_gates",
) -> list[dict]:
    return [r for r in records if gate_level in r.get(passed_gates_key, [])]
