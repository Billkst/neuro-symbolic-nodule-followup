import pytest

from src.weak_supervision.base import ABSTAIN, AggregatedLabel, LFOutput
from src.weak_supervision.aggregation import weighted_majority_vote
from src.weak_supervision.quality_gate import evaluate_gate, filter_by_gate, GATE_ORDER


class TestWeightedMajorityVote:
    def test_all_abstain_returns_abstain(self):
        outputs = [
            LFOutput(lf_name="LF-1", label=ABSTAIN),
            LFOutput(lf_name="LF-2", label=ABSTAIN),
        ]
        result = weighted_majority_vote(outputs)
        assert result.label == ABSTAIN
        assert result.confidence == 0.0
        assert result.lf_coverage == 0

    def test_single_vote(self):
        outputs = [
            LFOutput(lf_name="LF-1", label="solid", confidence=1.0, evidence_span="solid nodule"),
            LFOutput(lf_name="LF-2", label=ABSTAIN),
        ]
        result = weighted_majority_vote(outputs)
        assert result.label == "solid"
        assert result.confidence == 1.0
        assert result.lf_coverage == 1
        assert result.lf_agreement == 1.0
        assert result.supporting_lfs == ["LF-1"]

    def test_unanimous_vote(self):
        outputs = [
            LFOutput(lf_name="LF-1", label="ground_glass", confidence=1.0),
            LFOutput(lf_name="LF-2", label="ground_glass", confidence=0.8),
            LFOutput(lf_name="LF-3", label="ground_glass", confidence=0.7),
        ]
        result = weighted_majority_vote(outputs)
        assert result.label == "ground_glass"
        assert result.confidence == 1.0
        assert result.lf_coverage == 3
        assert result.lf_agreement == 1.0

    def test_conflict_majority_wins(self):
        outputs = [
            LFOutput(lf_name="LF-1", label="solid", confidence=1.0),
            LFOutput(lf_name="LF-2", label="solid", confidence=0.9),
            LFOutput(lf_name="LF-3", label="ground_glass", confidence=0.8),
        ]
        result = weighted_majority_vote(outputs)
        assert result.label == "solid"
        assert result.lf_coverage == 3
        assert result.lf_agreement == pytest.approx(2 / 3, abs=0.01)
        assert "LF-1" in result.supporting_lfs
        assert "LF-2" in result.supporting_lfs

    def test_weights_override_majority(self):
        outputs = [
            LFOutput(lf_name="LF-1", label="solid", confidence=1.0),
            LFOutput(lf_name="LF-2", label="ground_glass", confidence=1.0),
        ]
        weights = {"LF-1": 0.3, "LF-2": 0.9}
        result = weighted_majority_vote(outputs, weights=weights)
        assert result.label == "ground_glass"

    def test_empty_input(self):
        result = weighted_majority_vote([])
        assert result.label == ABSTAIN
        assert result.lf_coverage == 0

    def test_evidence_spans_collected(self):
        outputs = [
            LFOutput(lf_name="LF-1", label="solid", confidence=1.0, evidence_span="solid mass"),
            LFOutput(lf_name="LF-2", label="solid", confidence=0.8, evidence_span="solid nodule"),
        ]
        result = weighted_majority_vote(outputs)
        assert "solid mass" in result.evidence_spans
        assert "solid nodule" in result.evidence_spans

    def test_confidence_calculation(self):
        outputs = [
            LFOutput(lf_name="LF-1", label="solid", confidence=1.0),
            LFOutput(lf_name="LF-2", label="ground_glass", confidence=1.0),
        ]
        result = weighted_majority_vote(outputs)
        assert result.confidence == pytest.approx(0.5, abs=0.01)


class TestEvaluateGate:
    def test_all_abstain_rejected(self):
        agg = AggregatedLabel(
            label=ABSTAIN, confidence=0.0, lf_coverage=0,
            lf_agreement=0.0, supporting_lfs=[], all_votes={}, evidence_spans=[],
        )
        gate = evaluate_gate(agg)
        assert gate.gate_level == "REJECTED"
        assert gate.passed_gates == []

    def test_g1_minimal(self):
        agg = AggregatedLabel(
            label="solid", confidence=0.5, lf_coverage=1,
            lf_agreement=1.0, supporting_lfs=["LF-1"], all_votes={"solid": 0.5}, evidence_spans=[],
        )
        gate = evaluate_gate(agg)
        assert "G1" in gate.passed_gates
        assert gate.gate_level in GATE_ORDER

    def test_g5_strict(self):
        agg = AggregatedLabel(
            label="solid", confidence=0.9, lf_coverage=3,
            lf_agreement=1.0, supporting_lfs=["LF-1", "LF-2", "LF-3"],
            all_votes={"solid": 2.7}, evidence_spans=[],
        )
        gate = evaluate_gate(agg)
        assert "G5" in gate.passed_gates
        assert gate.gate_level == "G5"
        assert len(gate.passed_gates) == 5

    def test_g2_confidence_threshold(self):
        agg = AggregatedLabel(
            label="solid", confidence=0.65, lf_coverage=1,
            lf_agreement=1.0, supporting_lfs=["LF-1"], all_votes={"solid": 0.65}, evidence_spans=[],
        )
        gate = evaluate_gate(agg)
        assert "G1" in gate.passed_gates
        assert "G2" not in gate.passed_gates

    def test_g3_coverage_threshold(self):
        agg = AggregatedLabel(
            label="solid", confidence=0.5, lf_coverage=2,
            lf_agreement=0.5, supporting_lfs=["LF-1"], all_votes={"solid": 0.5, "unclear": 0.5}, evidence_spans=[],
        )
        gate = evaluate_gate(agg)
        assert "G3" in gate.passed_gates

    def test_g4_agreement_threshold(self):
        agg = AggregatedLabel(
            label="solid", confidence=0.6, lf_coverage=3,
            lf_agreement=0.9, supporting_lfs=["LF-1", "LF-2", "LF-3"],
            all_votes={"solid": 1.8}, evidence_spans=[],
        )
        gate = evaluate_gate(agg)
        assert "G4" in gate.passed_gates


class TestFilterByGate:
    def test_filter_g1(self):
        records = [
            {"sample_id": "a", "passed_gates": ["G1"]},
            {"sample_id": "b", "passed_gates": ["G1", "G2", "G3"]},
            {"sample_id": "c", "passed_gates": []},
        ]
        result = filter_by_gate(records, "G1")
        assert len(result) == 2
        assert result[0]["sample_id"] == "a"

    def test_filter_g5_strict(self):
        records = [
            {"sample_id": "a", "passed_gates": ["G1"]},
            {"sample_id": "b", "passed_gates": ["G1", "G2", "G3", "G4", "G5"]},
        ]
        result = filter_by_gate(records, "G5")
        assert len(result) == 1
        assert result[0]["sample_id"] == "b"

    def test_filter_empty(self):
        result = filter_by_gate([], "G1")
        assert result == []
