"""OCR signals from multiple plausible regions; no single era-specific crop is assumed."""
import re
from collections import Counter

import cv2
import numpy as np
import pytesseract

IGNORE_TITLE_WORDS = {
    "stage", "basic", "evolves", "from", "hp", "vstar", "vmax", "rule",
    "ability", "weakness", "resistance", "retreat", "pokemon",
}

# "227/182" style: digits only on both sides.
DIGIT_SLASH_RE = re.compile(r"(?<!\d)(\d{1,3})\s*/\s*(\d{1,3})(?!\d)")
# "TG05/TG30", "GG12/GG70", "SV045/SV122" style.
PREFIXED_SLASH_RE = re.compile(r"\b(TG|GG|SV)\s?(\d{1,3})\s*/\s*(TG|GG|SV)?\s?(\d{1,3})\b")
# "SVP 045", "SWSH123" style promos.
PROMO_RE = re.compile(r"\b(SVP|SWSH|SM|XY|BW|HGSS|DP|TG|GG)\s?-?\s?(\d{1,3})\b")


def _prepare(gray, scale=2.0):
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if np.mean(out) < 127:
        out = cv2.bitwise_not(out)
    return cv2.copyMakeBorder(out, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


def _read(gray, psm, whitelist=None):
    config = f"--psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(_prepare(gray), config=config).strip()


def _clean_name(text):
    # Digits are dropped on purpose so HP values like "160" never end up in the name.
    text = re.sub(r"[^A-Za-z'’ .-]", " ", text)
    words = []
    for word in text.split():
        clean = re.sub(r"[^A-Za-z'’-]", "", word)
        if len(clean) >= 2 and clean.lower() not in IGNORE_TITLE_WORDS:
            words.append(clean)
    return " ".join(words).strip()


def extract_card_signals(card_bgr):
    h, w = card_bgr.shape[:2]
    gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)

    # ---- Title -------------------------------------------------------------
    # Try several title bands. Different eras and full-art cards move the title.
    title_regions = [
        ("top", gray[int(h * .025):int(h * .17), int(w * .035):int(w * .80)]),
        ("upper_wide", gray[int(h * .035):int(h * .22), int(w * .02):int(w * .96)]),
        ("upper_middle", gray[int(h * .10):int(h * .30), int(w * .02):int(w * .96)]),
    ]
    title_reads = []
    for label, crop in title_regions:
        if crop.size:
            raw = _read(crop, 6)
            cleaned = _clean_name(raw)
            if cleaned:
                title_reads.append((cleaned, label, raw))

    # Favor reads with more real letters (capped so long noisy reads don't win).
    # The app validates every read against the card database afterwards.
    title_reads.sort(key=lambda it: min(sum(c.isalpha() for c in it[0]), 24), reverse=True)
    name = title_reads[0][0] if title_reads else ""
    name_candidates = list(dict.fromkeys(t[0] for t in title_reads))

    # ---- Collector number --------------------------------------------------
    # Strict patterns only. If nothing matches, the number stays empty, which is
    # better than feeding garbage into the matcher as if it were evidence.
    number_regions = [
        ("bottom_left", gray[int(h * .93):int(h * .99), int(w * .02):int(w * .45)]),
        ("bottom_right", gray[int(h * .93):int(h * .99), int(w * .55):int(w * .98)]),
        ("lower_left", gray[int(h * .90):int(h * .99), int(w * .02):int(w * .50)]),
    ]
    number_candidates = []  # (quality, value, region label, raw text)
    for label, crop in number_regions:
        if not crop.size:
            continue

        # Pass 1: digits and slash only.
        for psm in (7, 6):
            raw = _read(crop, psm, "0123456789/")
            found = DIGIT_SLASH_RE.search(raw)
            if found:
                value = f"{int(found.group(1))}/{int(found.group(2))}"
                number_candidates.append((3, value, label, raw))

        # Pass 2: letters allowed, for prefixed and promo numbers.
        # No space or hyphen in the whitelist: pytesseract splits the config on spaces,
        # which turned "/ -" into a stray "-" argument and crashed Tesseract.
        raw = _read(crop, 7, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/")
        upper = raw.upper()
        found = PREFIXED_SLASH_RE.search(upper)
        if found:
            p1, n1, p2, n2 = found.groups()
            value = f"{p1}{int(n1):02d}/{(p2 or p1)}{int(n2):02d}"
            number_candidates.append((3, value, label, raw))
            continue
        found = PROMO_RE.search(upper)
        if found:
            value = f"{found.group(1)} {int(found.group(2)):03d}"
            number_candidates.append((2, value, label, raw))

    # Majority vote across reads beats a single "best-looking" read.
    votes = Counter(value for _, value, _, _ in number_candidates)
    quality_of = {value: quality for quality, value, _, _ in number_candidates}
    number = ""
    if votes:
        number = max(votes, key=lambda v: (votes[v], quality_of[v]))

    # Printed set total (the 182 in 227/182) is useful set evidence for the matcher.
    printed_total = ""
    if "/" in number:
        digits = re.sub(r"\D", "", number.split("/", 1)[1])
        printed_total = digits.lstrip("0") or ""

    # ---- Debug -------------------------------------------------------------
    set_hint = ""
    debug = "Title OCR attempts:\n"
    debug += "\n".join(f"- {label}: {raw!r} -> {cleaned!r}" for cleaned, label, raw in title_reads)
    debug += "\nNumber OCR attempts:\n"
    debug += "\n".join(
        f"- {label}: {raw!r} -> {value!r}" for _, value, label, raw in number_candidates[:8]
    )
    if not title_reads:
        debug += "\nNo title text was confidently extracted."
    if not number_candidates:
        debug += "\nNo collector number matched a strict pattern."

    return {
        "name": name,
        "name_candidates": name_candidates,
        "number": number,
        "printed_total": printed_total,
        "set_hint": set_hint,
        "debug": debug,
    }