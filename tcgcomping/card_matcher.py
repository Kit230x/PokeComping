"""Candidate scoring and confidence-aware ranking."""
import re

from rapidfuzz import fuzz

# Keys a card database might use for the set's printed card count.
_TOTAL_KEYS = ("printed_total", "set_total", "total", "card_count", "official")


def _norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


def _number_forms(value):
    """Equivalent spellings of a collector number.

    OCR usually gives "227/182" while databases usually store "227", so the
    numerator alone is included as a form. Leading zeros are normalized.
    """
    value = str(value or "").upper().replace(" ", "")
    value = re.sub(r"[^A-Z0-9/]", "", value)
    forms = set()
    if "/" in value:
        left = value.split("/", 1)[0]
        forms.update({value, left, left.lstrip("0") or "0"})
    elif value:
        forms.update({value, value.lstrip("0") or "0"})
    return forms


def _candidate_total(candidate):
    for key in _TOTAL_KEYS:
        raw = candidate.get(key)
        digits = re.sub(r"\D", "", str(raw or ""))
        if digits:
            return digits.lstrip("0")
    return ""


def rank_candidates(signals, candidates, aligned_image=None):
    """Score name strongly; collector number is strong evidence when present.

    Confidence is a ranking heuristic, not a calibrated probability. A human confirmation
    remains available for every result, and low-confidence candidates are not auto-selected.
    """
    query_name = signals.get("name", "")
    query_number = signals.get("number", "")
    query_set = signals.get("set_hint", "")
    query_total = re.sub(r"\D", "", str(signals.get("printed_total", "") or "")).lstrip("0")
    ranked = []

    for candidate in candidates:
        name_score = (
            fuzz.WRatio(_norm(query_name), _norm(candidate.get("name", ""))) / 100
            if query_name else 0.0
        )

        # Exact match on an equivalent form only. Substring matching was removed
        # because garbage like "TE002274" used to match "227".
        qforms = _number_forms(query_number)
        cforms = _number_forms(candidate.get("number", ""))
        number_score = 1.0 if qforms and cforms and qforms.intersection(cforms) else 0.0

        set_score = 0.0
        if query_set:
            set_score = fuzz.WRatio(_norm(query_set), _norm(candidate.get("set_name", ""))) / 100
        # A matching printed total (the 182 in 227/182) is strong set evidence.
        cand_total = _candidate_total(candidate)
        if query_total and cand_total and query_total == cand_total:
            set_score = max(set_score, 1.0)

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

    ranked.sort(
        key=lambda x: (
            x["confidence"],
            x["evidence"]["collector_number_score"],
            x["evidence"]["name_score"],
        ),
        reverse=True,
    )
    return ranked[:10]