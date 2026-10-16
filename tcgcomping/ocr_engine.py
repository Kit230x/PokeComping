"""OCR signals from multiple plausible regions; no single era-specific crop is assumed."""
import re
import cv2
import numpy as np
import pytesseract
from rapidfuzz import fuzz

IGNORE_TITLE_WORDS = {
    "stage", "basic", "evolves", "from", "hp", "vstar", "vmax", "rule",
    "ability", "weakness", "resistance", "retreat", "pokemon"
}

def _prepare(gray, scale=2.5):
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
    text = re.sub(r"[^A-Za-z0-9'’ .-]", " ", text)
    words = []
    for word in text.split():
        clean = re.sub(r"[^A-Za-z0-9'’-]", "", word)
        if len(clean) > 1 and clean.lower() not in IGNORE_TITLE_WORDS:
            words.append(clean)
    return " ".join(words).strip()

def extract_card_signals(card_bgr):
    h, w = card_bgr.shape[:2]
    gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)

    # Try several title bands. Different eras and full-art cards move the title.
    title_regions = [
        ("top", gray[int(h*.025):int(h*.17), int(w*.035):int(w*.80)]),
        ("upper_wide", gray[int(h*.035):int(h*.22), int(w*.02):int(w*.96)]),
        ("upper_middle", gray[int(h*.10):int(h*.30), int(w*.02):int(w*.96)]),
    ]
    title_reads = []
    for label, crop in title_regions:
        if crop.size:
            raw = _read(crop, 6)
            cleaned = _clean_name(raw)
            if cleaned:
                title_reads.append((cleaned, label, raw))

    # Pick the most plausible, favoring more readable text without excessive length.
    title_reads.sort(key=lambda item: (
        sum(ch.isalpha() for ch in item[0]) / max(1, len(item[0])),
        min(len(item[0]), 24),
        -max(0, len(item[0]) - 32),
    ), reverse=True)
    name = title_reads[0][0] if title_reads else ""

    # Read number-like text from multiple bottom zones and OCR modes.
    number_regions = [
        ("bottom_left", gray[int(h*.86):int(h*.99), int(w*.015):int(w*.43)]),
        ("bottom_center", gray[int(h*.86):int(h*.99), int(w*.25):int(w*.76)]),
        ("bottom_right", gray[int(h*.86):int(h*.99), int(w*.56):int(w*.99)]),
        ("lower_left", gray[int(h*.78):int(h*.94), int(w*.015):int(w*.48)]),
        ("lower_right", gray[int(h*.78):int(h*.94), int(w*.52):int(w*.99)]),
    ]
    number_candidates = []
    for label, crop in number_regions:
        if not crop.size:
            continue
        for psm in (7, 6, 11):
            raw = _read(crop, psm, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/")
            # Handles 58/102, TG05/TG30, SVP 012, 4, etc.
            compact = re.sub(r"\s+", "", raw.upper())
            patterns = [
                r"([A-Z]{1,4}\d{1,4}/[A-Z]{0,4}\d{1,4})",
                r"([A-Z]{1,4}\d{1,4}/\d{1,4})",
                r"(\d{1,4}/\d{1,4})",
                r"([A-Z]{1,4}\d{1,4})",
                r"(\d{1,4})",
            ]
            for pattern in patterns:
                found = re.search(pattern, compact)
                if found:
                    value = found.group(1).strip("/")
                    # Prefer slash-form collector numbers over short isolated digits.
                    quality = (3 if "/" in value else 2 if re.search(r"[A-Z]", value) else 1)
                    quality += min(len(value), 8) / 20
                    number_candidates.append((quality, value, label, raw))
                    break

    number_candidates.sort(reverse=True)
    number = number_candidates[0][1] if number_candidates else ""

    # Lightweight set-symbol/era detection is intentionally a hint, not a hard filter.
    # TCGdex metadata and the collector number remain the source of candidate evidence.
    set_hint = ""
    debug = "Title OCR attempts:\n"
    debug += "\n".join(f"- {label}: {raw!r} -> {cleaned!r}" for cleaned, label, raw in title_reads)
    debug += "\nNumber OCR attempts:\n"
    debug += "\n".join(f"- {label}: {raw!r} -> {value!r}" for _, value, label, raw in number_candidates[:8])
    if not title_reads:
        debug += "\nNo title text was confidently extracted."
    return {"name": name, "number": number, "set_hint": set_hint, "debug": debug}
