"""Candidate scoring and confidence-aware ranking."""
import re
from rapidfuzz import fuzz

def _norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())

def _number_forms(value):
    value = str(value or "").upper().replace(" ", "")
    value = re.sub(r"[^A-Z0-9/]", "", value)
    forms = {value}
    if "/" in value:
        left, right = value.split("/", 1)
        forms.add(left.lstrip("0") + "/" + right.lstrip("0"))
    return {x for x in forms if x}

def rank_candidates(signals, candidates, aligned_image=None):
    """Score name strongly; collector number is strong evidence when present.

    Confidence is a ranking heuristic, not a calibrated probability. A human confirmation
    remains available for every result, and low-confidence candidates are not auto-selected.
    """
    query_name = signals.get("name", "")
    query_number = signals.get("number", "")
    query_set = signals.get("set_hint", "")
    ranked = []

    for candidate in candidates:
        name_score = fuzz.WRatio(_norm(query_name), _norm(candidate.get("name", ""))) / 100 if query_name else 0.0
        cand_number = str(candidate.get("number", ""))
        qforms, cforms = _number_forms(query_number), _number_forms(cand_number)
        number_score = 1.0 if qforms and cforms and qforms.intersection(cforms) else 0.0
        # A partial OCR number can still contribute modest evidence.
        if not number_score and query_number and cand_number:
            qclean, cclean = _norm(query_number), _norm(cand_number)
            if len(qclean) >= 2 and (qclean in cclean or cclean in qclean):
                number_score = 0.45

        set_score = 0.0
        if query_set:
            set_score = fuzz.WRatio(_norm(query_set), _norm(candidate.get("set_name", ""))) / 100

        # Weights are intentionally transparent and easy to tune.
        if query_number:
            score = 0.68 * name_score + 0.27 * number_score + 0.05 * set_score
        else:
            score = 0.90 * name_score + 0.10 * set_score

        # Database retrieval similarity is a weak tie-breaker, not the final identity.
        db_score = float(candidate.get("database_name_score", 0) or 0)
        score = min(1.0, score * 0.94 + db_score * 0.06)

        item = dict(candidate)
        item["confidence"] = round(score, 4)
        item["evidence"] = {
            "name_score": round(name_score, 3),
            "collector_number_score": round(number_score, 3),
            "set_score": round(set_score, 3),
            "database_retrieval_score": round(db_score, 3),
            "note": "Heuristic ranking score, not a calibrated probability.",
        }
        ranked.append(item)

    ranked.sort(key=lambda x: (x["confidence"], x["evidence"]["collector_number_score"], x["evidence"]["name_score"]), reverse=True)
    return ranked[:10]
