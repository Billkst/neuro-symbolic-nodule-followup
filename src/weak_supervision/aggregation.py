from __future__ import annotations

from collections import defaultdict

from src.weak_supervision.base import ABSTAIN, AggregatedLabel, LFOutput


def weighted_majority_vote(
    lf_outputs: list[LFOutput],
    weights: dict[str, float] | None = None,
) -> AggregatedLabel:
    non_abstain = [o for o in lf_outputs if not o.is_abstain]

    if not non_abstain:
        return AggregatedLabel(
            label=ABSTAIN,
            confidence=0.0,
            lf_coverage=0,
            lf_agreement=0.0,
            supporting_lfs=[],
            all_votes={},
            evidence_spans=[],
        )

    vote_scores: dict[str, float] = defaultdict(float)
    vote_lfs: dict[str, list[str]] = defaultdict(list)
    vote_evidence: dict[str, list[str]] = defaultdict(list)

    for output in non_abstain:
        w = 1.0
        if weights and output.lf_name in weights:
            w = weights[output.lf_name]
        score = w * output.confidence
        vote_scores[output.label] += score
        vote_lfs[output.label].append(output.lf_name)
        if output.evidence_span:
            vote_evidence[output.label].append(output.evidence_span)

    total_score = sum(vote_scores.values())
    winner = max(vote_scores, key=lambda k: vote_scores[k])
    winner_score = vote_scores[winner]

    return AggregatedLabel(
        label=winner,
        confidence=winner_score / total_score if total_score > 0 else 0.0,
        lf_coverage=len(non_abstain),
        lf_agreement=len(vote_lfs[winner]) / len(non_abstain),
        supporting_lfs=vote_lfs[winner],
        all_votes=dict(vote_scores),
        evidence_spans=vote_evidence.get(winner, []),
    )
