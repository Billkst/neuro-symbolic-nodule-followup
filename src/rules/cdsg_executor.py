"""Clinical Decision State Graph executor for Module 3 hard-rule inference.

The executor is deliberately deterministic: it consumes structured case facts,
walks a guideline graph, and renders a terminal-node template. It does not call
LLMs, retrieval systems, or learned models.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "outputs/phaseA3/guideline_graph/lung_rads_v2022_cdsg.json"

_CATEGORY_ORDER = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4A": 4,
    "4B": 5,
    "4X": 6,
    "S": 7,
}

_CATEGORY_RISK = {
    "2": "benign_or_indolent",
    "3": "probably_benign",
    "4A": "suspicious",
    "4B": "very_suspicious",
    "4X": "very_suspicious",
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_cdsg_graph(path: str | Path = DEFAULT_GRAPH_PATH) -> dict[str, Any]:
    graph_path = Path(path)
    with graph_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _normalize_density(value: Any) -> str | None:
    if value is None:
        return None
    density = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "partsolid": "part_solid",
        "part_solid": "part_solid",
        "groundglass": "ground_glass",
        "ground_glass": "ground_glass",
        "ground_glass_opacity": "ground_glass",
        "ggo": "ground_glass",
        "fat": "fat_containing",
        "fat_containing": "fat_containing",
    }
    return aliases.get(density, density)


def _extract_solid_component_mm(nodule: dict[str, Any]) -> float | None:
    explicit = nodule.get("solid_component_mm")
    if isinstance(explicit, (int, float)):
        return float(explicit)

    texts = [
        nodule.get("density_text"),
        nodule.get("evidence_span"),
        nodule.get("recommendation_cue"),
    ]
    pattern = re.compile(
        r"solid\s+component(?:\s+(?:measuring|measures|measure(?:d)?|of))?\s*(\d+(?:\.\d+)?)\s*mm\b",
        re.IGNORECASE,
    )

    for text in texts:
        if not text:
            continue
        match = pattern.search(str(text))
        if match:
            return float(match.group(1))
    return None


def _is_stable_for_two_years(nodule: dict[str, Any]) -> bool:
    change_text = nodule.get("change_text") or ""
    evidence_span = nodule.get("evidence_span") or ""
    combined_text = f"{change_text} {evidence_span}".lower()
    return bool(
        re.search(r"\b(2\s*years?|two\s+years?|24\s*months?|2-year|two-year)\b", combined_text)
    )


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _to_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _condition_to_text(condition: dict[str, Any]) -> str:
    op = condition.get("op")
    if op == "always":
        return "always"
    if op in {"exists", "missing"}:
        return f"{condition.get('fact')} {op}"
    if op in {"equals", "lt", "lte", "gt", "gte"}:
        return f"{condition.get('fact')} {op} {condition.get('value')}"
    if op == "in":
        return f"{condition.get('fact')} in {condition.get('values')}"
    if op == "missing_or_in":
        return f"{condition.get('fact')} missing_or_in {condition.get('values')}"
    if op == "range":
        return f"{condition.get('gte')} <= {condition.get('fact')} < {condition.get('lt')}"
    if op in {"all", "any"}:
        return f"{op}({'; '.join(_condition_to_text(c) for c in condition.get('conditions', []))})"
    return json.dumps(condition, ensure_ascii=False, sort_keys=True)


def _evaluate_condition(condition: dict[str, Any], facts: dict[str, Any]) -> tuple[bool, list[str], str]:
    op = condition.get("op")
    fact_name = condition.get("fact")

    if op == "always":
        return True, [], "matched"

    if op == "all":
        missing_fields: list[str] = []
        reasons: list[str] = []
        for child in condition.get("conditions", []):
            matched, child_missing, reason = _evaluate_condition(child, facts)
            missing_fields.extend(child_missing)
            if not matched:
                reasons.append(reason)
        if reasons:
            return False, _deduplicate(missing_fields), "; ".join(reasons)
        return True, _deduplicate(missing_fields), "matched"

    if op == "any":
        missing_fields = []
        reasons = []
        for child in condition.get("conditions", []):
            matched, child_missing, reason = _evaluate_condition(child, facts)
            if matched:
                return True, [], "matched"
            missing_fields.extend(child_missing)
            reasons.append(reason)
        return False, _deduplicate(missing_fields), "; ".join(reasons) or "no child condition matched"

    value = facts.get(str(fact_name)) if fact_name is not None else None

    if op == "exists":
        if _is_missing(value):
            return False, [str(fact_name)], f"{fact_name} is missing"
        return True, [], "matched"

    if op == "missing":
        if _is_missing(value):
            return True, [str(fact_name)], "matched"
        return False, [], f"{fact_name} is present"

    if op == "missing_or_in":
        values = condition.get("values", [])
        if _is_missing(value) or value in values:
            return True, [str(fact_name)], "matched"
        return False, [], f"{fact_name}={value} not missing or in {values}"

    if op == "equals":
        expected = condition.get("value")
        if _is_missing(value):
            return False, [str(fact_name)], f"{fact_name} is missing"
        if value == expected:
            return True, [], "matched"
        return False, [], f"{fact_name}={value} != {expected}"

    if op == "in":
        values = condition.get("values", [])
        if _is_missing(value):
            return False, [str(fact_name)], f"{fact_name} is missing"
        if value in values:
            return True, [], "matched"
        return False, [], f"{fact_name}={value} not in {values}"

    if op in {"lt", "lte", "gt", "gte"}:
        number = _to_number(value)
        threshold = _to_number(condition.get("value"))
        if number is None:
            return False, [str(fact_name)], f"{fact_name} is not numeric"
        if threshold is None:
            return False, [], "threshold is not numeric"
        if op == "lt" and number < threshold:
            return True, [], "matched"
        if op == "lte" and number <= threshold:
            return True, [], "matched"
        if op == "gt" and number > threshold:
            return True, [], "matched"
        if op == "gte" and number >= threshold:
            return True, [], "matched"
        return False, [], f"{fact_name}={number} failed {op} {threshold}"

    if op == "range":
        number = _to_number(value)
        if number is None:
            return False, [str(fact_name)], f"{fact_name} is not numeric"
        gte = condition.get("gte")
        lt = condition.get("lt")
        if gte is not None and number < float(gte):
            return False, [], f"{fact_name}={number} < lower bound {gte}"
        if lt is not None and number >= float(lt):
            return False, [], f"{fact_name}={number} >= upper bound {lt}"
        return True, [], "matched"

    return False, [], f"unsupported condition op={op}"


def _flatten_nodule_candidates(case_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    normalized_nodules = case_bundle.get("nodules")
    if isinstance(normalized_nodules, list):
        for nodule in normalized_nodules:
            candidates.append(
                {
                    "note_id": None,
                    "report_nodule_count": len(normalized_nodules),
                    "report_text": None,
                    "sections": {},
                    "nodule": nodule,
                }
            )

    for fact in case_bundle.get("radiology_facts", []) or []:
        report_nodule_count = fact.get("nodule_count")
        for nodule in fact.get("nodules", []) or []:
            candidates.append(
                {
                    "note_id": fact.get("note_id"),
                    "report_nodule_count": report_nodule_count,
                    "report_text": fact.get("report_text"),
                    "sections": fact.get("sections") or {},
                    "nodule": nodule,
                }
            )
    return candidates


def _build_candidate_facts(candidate: dict[str, Any] | None, has_nodule: bool) -> dict[str, Any]:
    if candidate is None:
        return {
            "has_nodule": has_nodule,
            "size_mm": None,
            "density": None,
            "change_status": None,
            "is_new": False,
            "stable_for_two_years": False,
            "solid_component_mm": None,
            "location_lobe": None,
            "evidence_span": None,
            "source_confidence": None,
        }

    nodule = candidate["nodule"]
    change_status = nodule.get("change_status")
    density = _normalize_density(nodule.get("density_category"))
    size_mm = nodule.get("size_mm")
    if size_mm is not None:
        size_mm = _to_number(size_mm)

    return {
        "has_nodule": has_nodule,
        "size_mm": size_mm,
        "density": density,
        "change_status": change_status,
        "is_new": change_status == "new",
        "stable_for_two_years": change_status == "stable" and _is_stable_for_two_years(nodule),
        "solid_component_mm": _extract_solid_component_mm(nodule),
        "location_lobe": nodule.get("location_lobe"),
        "location_text": nodule.get("location_text"),
        "evidence_span": nodule.get("evidence_span"),
        "source_confidence": nodule.get("confidence"),
        "missing_flags": nodule.get("missing_flags") or [],
        "nodule": nodule,
        "note_id": candidate.get("note_id"),
    }


def _anchor_from_graph(anchor: dict[str, Any], graph_element_id: str, graph_element_type: str) -> dict[str, Any]:
    return {
        "anchor_id": str(anchor.get("anchor_id") or f"{graph_element_id}_anchor"),
        "source": str(anchor.get("source") or "Lung-RADS_v2022"),
        "graph_element_id": graph_element_id,
        "graph_element_type": graph_element_type,
        "condition": str(anchor.get("condition") or "graph condition"),
        "text": anchor.get("text"),
    }


def _anchor_from_edge(edge: dict[str, Any], guideline_source: str) -> dict[str, Any]:
    return {
        "anchor_id": f"{guideline_source}:{edge['edge_id']}",
        "source": guideline_source,
        "graph_element_id": edge["edge_id"],
        "graph_element_type": "edge",
        "condition": edge.get("condition_text") or _condition_to_text(edge.get("condition", {})),
        "text": edge.get("reasoning_template"),
    }


def _template_key_for_category(category: str | None, is_new: bool, default_key: str | None = None) -> str:
    if category == "4A":
        return "4A_new" if is_new else "4A_existing"
    if category in {"2", "3", "4B"}:
        return category
    if default_key:
        return default_key
    return "insufficient_data"


def _upgrade_category(category: str | None) -> str | None:
    if category == "2":
        return "3"
    if category == "3":
        return "4A"
    if category == "4A":
        return "4B"
    return category


class CDSGExecutor:
    """Execute a deterministic guideline graph over structured case facts."""

    def __init__(self, graph: dict[str, Any]):
        self.graph = graph
        self.nodes = {node["node_id"]: node for node in graph.get("nodes", [])}
        self.edges_by_source: dict[str, list[dict[str, Any]]] = {}
        for edge in graph.get("edges", []):
            self.edges_by_source.setdefault(edge["from"], []).append(edge)
        for edges in self.edges_by_source.values():
            edges.sort(key=lambda item: int(item.get("priority", 1000)))

    @classmethod
    def from_path(cls, path: str | Path = DEFAULT_GRAPH_PATH) -> "CDSGExecutor":
        return cls(load_cdsg_graph(path))

    def execute(self, case_bundle: dict[str, Any]) -> dict[str, Any]:
        candidates = _flatten_nodule_candidates(case_bundle)
        if not candidates:
            evaluation = self._execute_candidate(None, has_nodule=False)
            return self._render_case_output(case_bundle, evaluation, nodule_count=0)

        evaluations = [self._execute_candidate(candidate, has_nodule=True) for candidate in candidates]
        selected = self._select_dominant_evaluation(evaluations)
        return self._render_case_output(case_bundle, selected, nodule_count=len(candidates))

    def _execute_candidate(self, candidate: dict[str, Any] | None, has_nodule: bool) -> dict[str, Any]:
        facts = _build_candidate_facts(candidate, has_nodule=has_nodule)
        current_node_id = str(self.graph["entry_node"])
        visited_nodes: list[str] = []
        matched_edges: list[str] = []
        failed_conditions: list[dict[str, Any]] = []
        reasoning_path: list[dict[str, Any]] = []
        decision_path: list[str] = []
        anchors: list[dict[str, Any]] = []
        step = 1

        while True:
            node = self.nodes[current_node_id]
            visited_nodes.append(current_node_id)
            node_type = node.get("node_type")

            if node_type in {"terminal_recommendation", "terminal_abstention"}:
                reasoning_path.append(
                    {
                        "step": step,
                        "node_id": current_node_id,
                        "edge_id": None,
                        "condition": "terminal node reached",
                        "matched": True,
                        "match_type": "abstention" if node_type == "terminal_abstention" else "terminal",
                        "evidence_span": facts.get("evidence_span"),
                        "facts_used": self._facts_for_reasoning(facts),
                    }
                )
                terminal_anchor = node.get("guideline_anchor")
                if isinstance(terminal_anchor, dict):
                    anchors.append(_anchor_from_graph(terminal_anchor, current_node_id, "node"))
                return self._build_terminal_evaluation(
                    node=node,
                    facts=facts,
                    visited_nodes=visited_nodes,
                    matched_edges=matched_edges,
                    failed_conditions=failed_conditions,
                    reasoning_path=reasoning_path,
                    decision_path=decision_path,
                    guideline_anchor=anchors,
                )

            outgoing_edges = self.edges_by_source.get(current_node_id, [])
            matched_edge = None
            for edge in outgoing_edges:
                condition = edge.get("condition", {"op": "always"})
                matched, missing_fields, reason = _evaluate_condition(condition, facts)
                if matched:
                    matched_edge = edge
                    break
                failed_conditions.append(
                    {
                        "node_id": current_node_id,
                        "edge_id": edge["edge_id"],
                        "condition": edge.get("condition_text") or _condition_to_text(condition),
                        "reason": reason,
                        "missing_fields": _deduplicate(missing_fields),
                    }
                )

            if matched_edge is None:
                return self._fallback_abstention(
                    current_node_id=current_node_id,
                    facts=facts,
                    visited_nodes=visited_nodes,
                    matched_edges=matched_edges,
                    failed_conditions=failed_conditions,
                    reasoning_path=reasoning_path,
                    decision_path=decision_path,
                    guideline_anchor=anchors,
                )

            matched_edges.append(matched_edge["edge_id"])
            decision_path.append(f"{matched_edge['from']}->{matched_edge['to']}")
            anchors.append(_anchor_from_edge(matched_edge, self.graph.get("guideline_source", "Lung-RADS_v2022")))
            reasoning_path.append(
                {
                    "step": step,
                    "node_id": current_node_id,
                    "edge_id": matched_edge["edge_id"],
                    "condition": matched_edge.get("condition_text")
                    or _condition_to_text(matched_edge.get("condition", {})),
                    "matched": True,
                    "match_type": "hard",
                    "evidence_span": facts.get("evidence_span"),
                    "facts_used": self._facts_for_reasoning(facts),
                }
            )
            step += 1
            current_node_id = matched_edge["to"]

    def _build_terminal_evaluation(
        self,
        node: dict[str, Any],
        facts: dict[str, Any],
        visited_nodes: list[str],
        matched_edges: list[str],
        failed_conditions: list[dict[str, Any]],
        reasoning_path: list[dict[str, Any]],
        decision_path: list[str],
        guideline_anchor: list[dict[str, Any]],
    ) -> dict[str, Any]:
        node_type = node.get("node_type")
        category = node.get("lung_rads_category")
        template_key = node.get("recommendation_template")
        missing_info = list(node.get("missing_info") or [])
        missing_info.extend(node.get("adds_missing_info") or [])
        evidence_notes = []
        if node.get("adds_evidence_note"):
            evidence_notes.append(node["adds_evidence_note"])

        triggered_rules = [node.get("triggered_rule")] if node.get("triggered_rule") else []
        risk_category = node.get("risk_category") or _CATEGORY_RISK.get(category)
        severity = int(node.get("severity", _CATEGORY_ORDER.get(str(category), -1)))
        abstention_reason = node.get("abstention_reason") if node_type == "terminal_abstention" else None

        evaluation = {
            "terminal_node_id": node["node_id"],
            "terminal_node_type": node_type,
            "facts": facts,
            "lung_rads_category": category,
            "risk_category": risk_category,
            "recommendation_template": template_key,
            "severity": severity,
            "missing_info": missing_info,
            "evidence_notes": evidence_notes,
            "abstention_reason": abstention_reason,
            "visited_nodes": visited_nodes,
            "matched_edges": matched_edges,
            "failed_conditions": failed_conditions,
            "reasoning_path": reasoning_path,
            "decision_path": decision_path,
            "guideline_anchor": guideline_anchor,
            "triggered_rules": triggered_rules,
        }
        if node_type == "terminal_recommendation":
            self._apply_modifiers(evaluation)
        return evaluation

    def _fallback_abstention(
        self,
        current_node_id: str,
        facts: dict[str, Any],
        visited_nodes: list[str],
        matched_edges: list[str],
        failed_conditions: list[dict[str, Any]],
        reasoning_path: list[dict[str, Any]],
        decision_path: list[str],
        guideline_anchor: list[dict[str, Any]],
    ) -> dict[str, Any]:
        node = {
            "node_id": f"A_NO_MATCH_{current_node_id}",
            "node_type": "terminal_abstention",
            "lung_rads_category": None,
            "risk_category": "insufficient_data",
            "recommendation_template": "insufficient_data",
            "severity": -1,
            "triggered_rule": "fallback_no_graph_edge_matched",
            "abstention_reason": "no_graph_edge_matched",
            "missing_info": [],
            "guideline_anchor": {
                "anchor_id": "LR2022_MVP_ABSTAIN_NO_EDGE",
                "source": self.graph.get("guideline_source", "Lung-RADS_v2022"),
                "condition": "no outgoing edge matched",
                "text": "No CDSG edge matched the available structured facts.",
            },
        }
        visited_nodes.append(node["node_id"])
        return self._build_terminal_evaluation(
            node=node,
            facts=facts,
            visited_nodes=visited_nodes,
            matched_edges=matched_edges,
            failed_conditions=failed_conditions,
            reasoning_path=reasoning_path,
            decision_path=decision_path,
            guideline_anchor=guideline_anchor,
        )

    def _apply_modifiers(self, evaluation: dict[str, Any]) -> None:
        facts = dict(evaluation["facts"])
        category = evaluation.get("lung_rads_category")
        facts["lung_rads_category"] = category

        for modifier in self.graph.get("modifiers", []):
            matched, _, _ = _evaluate_condition(modifier.get("condition", {"op": "always"}), facts)
            if not matched:
                continue

            modifier_id = modifier["modifier_id"]
            old_category = evaluation.get("lung_rads_category")
            new_category = old_category

            if modifier.get("new_lung_rads_category") is not None:
                new_category = modifier["new_lung_rads_category"]
            elif modifier.get("upgrade_category_by_one"):
                new_category = _upgrade_category(old_category)

            if new_category != old_category:
                evaluation["lung_rads_category"] = new_category
                evaluation["risk_category"] = _CATEGORY_RISK.get(new_category, evaluation.get("risk_category"))
                evaluation["recommendation_template"] = _template_key_for_category(
                    new_category,
                    is_new=bool(facts.get("is_new")),
                    default_key=evaluation.get("recommendation_template"),
                )
                evaluation["severity"] = _CATEGORY_ORDER.get(str(new_category), evaluation.get("severity", -1))

            if modifier.get("triggered_rule"):
                evaluation["triggered_rules"].append(modifier["triggered_rule"])
            if modifier.get("adds_evidence_note"):
                evaluation["evidence_notes"].append(modifier["adds_evidence_note"])

            evaluation["visited_nodes"].append(modifier_id)
            evaluation["decision_path"].append(f"{old_category}->{new_category}:{modifier_id}")
            evaluation["reasoning_path"].append(
                {
                    "step": len(evaluation["reasoning_path"]) + 1,
                    "node_id": modifier_id,
                    "edge_id": None,
                    "condition": modifier.get("condition_text") or _condition_to_text(modifier.get("condition", {})),
                    "matched": True,
                    "match_type": "modifier",
                    "evidence_span": facts.get("evidence_span"),
                    "facts_used": self._facts_for_reasoning(facts),
                }
            )
            anchor = modifier.get("guideline_anchor")
            if isinstance(anchor, dict):
                evaluation["guideline_anchor"].append(_anchor_from_graph(anchor, modifier_id, "modifier"))
            facts["lung_rads_category"] = evaluation.get("lung_rads_category")

    def _select_dominant_evaluation(self, evaluations: list[dict[str, Any]]) -> dict[str, Any]:
        actionable = [
            item for item in evaluations if item.get("terminal_node_type") == "terminal_recommendation"
        ]
        if actionable:
            return max(
                actionable,
                key=lambda item: (
                    int(item.get("severity", -1)),
                    _to_number(item.get("facts", {}).get("size_mm")) or -1.0,
                ),
            )

        return max(
            evaluations,
            key=lambda item: (
                1 if item.get("abstention_reason") == "missing_nodule_density" else 0,
                _to_number(item.get("facts", {}).get("size_mm")) or -1.0,
            ),
        )

    def _render_case_output(
        self,
        case_bundle: dict[str, Any],
        evaluation: dict[str, Any],
        nodule_count: int,
    ) -> dict[str, Any]:
        smoking_eligibility = case_bundle.get("smoking_eligibility")
        patient_risk_level = None
        smoking_eligible = None
        missing_info = list(evaluation.get("missing_info") or [])
        evidence_notes = list(evaluation.get("evidence_notes") or [])

        if smoking_eligibility is None:
            missing_info.append("smoking_eligibility")
            evidence_notes.append("缺少 smoking_eligibility，不阻断 Lung-RADS hard-rule 图执行。")
        else:
            smoking_eligible = smoking_eligibility.get("eligible_for_high_risk_screening")
            if smoking_eligible == "eligible":
                patient_risk_level = "high_risk"
            elif smoking_eligible == "not_eligible":
                patient_risk_level = "not_high_risk"
            else:
                patient_risk_level = "unknown"

        missing_info = _deduplicate(missing_info)
        facts = evaluation.get("facts", {})
        template_key = evaluation.get("recommendation_template") or "insufficient_data"
        template = deepcopy(self.graph["recommendation_templates"][template_key])
        category = evaluation.get("lung_rads_category")

        recommendation = {
            "level": template["recommendation_level"],
            "action": template["recommendation_action"],
            "followup_interval": template["followup_interval"],
            "followup_modality": template["followup_modality"],
        }

        output = {
            "case_id": case_bundle.get("case_id") or "unknown_case",
            "recommendation": recommendation,
            "recommendation_level": recommendation["level"],
            "recommendation_action": recommendation["action"],
            "followup_interval": recommendation["followup_interval"],
            "followup_modality": recommendation["followup_modality"],
            "risk_category": evaluation.get("risk_category"),
            "lung_rads_category": category,
            "guideline_source": self.graph.get("guideline_source", "Lung-RADS_v2022"),
            "guideline_anchor": evaluation.get("guideline_anchor") or [],
            "reasoning_path": evaluation.get("reasoning_path") or [],
            "missing_info": missing_info,
            "missing_information": missing_info,
            "evidence_quality": self._evidence_quality(facts, missing_info, evidence_notes, evaluation),
            "decision_path": evaluation.get("decision_path") or [str(evaluation.get("terminal_node_id"))],
            "visited_nodes": evaluation.get("visited_nodes") or [str(evaluation.get("terminal_node_id"))],
            "matched_edges": evaluation.get("matched_edges") or [],
            "failed_conditions": evaluation.get("failed_conditions") or [],
            "abstention_reason": evaluation.get("abstention_reason"),
            "triggered_rules": _deduplicate(evaluation.get("triggered_rules") or []),
            "input_facts_used": {
                "nodule_size_mm": facts.get("size_mm"),
                "nodule_density": facts.get("density"),
                "nodule_count": nodule_count,
                "change_status": facts.get("change_status"),
                "patient_risk_level": patient_risk_level,
                "smoking_eligible": smoking_eligible,
                "solid_component_mm": facts.get("solid_component_mm"),
                "location_lobe": facts.get("location_lobe"),
                "multiple_nodules": nodule_count > 1 if nodule_count is not None else None,
                "note_id": facts.get("note_id"),
            },
            "output_type": "cdsg_rule_based",
            "generation_metadata": {
                "engine_version": "cdsg_executor_0.1",
                "generation_timestamp": _timestamp(),
                "rules_version": self.graph.get("rules_version", "unknown_rules"),
                "graph_id": self.graph.get("graph_id", "unknown_graph"),
                "graph_version": self.graph.get("version", "unknown_version"),
                "terminal_node_id": str(evaluation.get("terminal_node_id")),
            },
        }
        return output

    def _evidence_quality(
        self,
        facts: dict[str, Any],
        missing_info: list[str],
        evidence_notes: list[str],
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        source_confidence = facts.get("source_confidence")
        if evaluation.get("abstention_reason"):
            overall = "insufficient"
        elif source_confidence == "low":
            overall = "low"
        elif missing_info:
            overall = "medium"
        elif source_confidence == "high":
            overall = "high"
        else:
            overall = "medium"

        return {
            "overall": overall,
            "fact_confidence": source_confidence,
            "source_confidence": source_confidence,
            "notes": _deduplicate(evidence_notes),
        }

    def _facts_for_reasoning(self, facts: dict[str, Any]) -> dict[str, Any]:
        return {
            "size_mm": facts.get("size_mm"),
            "density": facts.get("density"),
            "change_status": facts.get("change_status"),
            "solid_component_mm": facts.get("solid_component_mm"),
            "stable_for_two_years": facts.get("stable_for_two_years"),
            "location_lobe": facts.get("location_lobe"),
        }


def generate_cdsg_recommendation(
    case_bundle: dict[str, Any],
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
) -> dict[str, Any]:
    executor = CDSGExecutor.from_path(graph_path)
    return executor.execute(case_bundle)
