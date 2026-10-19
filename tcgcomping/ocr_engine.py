"""OCR signals from multiple plausible regions; no single era-specific crop is assumed."""
import os
import re
from collections import Counter

# Tesseract runs faster when each call is single-threaded. Set before the first OCR call.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import cv2
import numpy as np
import pytesseract

# Card label words printed near the name that are not part of it. Add more here if a new
# card layout leaks a label into the name. Matching is fuzzy (see _is_label), so truncated
# OCR reads like "BASI" or "STAGEI" are caught too.
IGNORE_TITLE_WORDS = {
    "basic", "stage", "trainer", "supporter", "item", "tool", "stadium",
    "evolves", "from", "pokemon", "pokémon", "hp", "vstar", "vmax", "rule",
    "ability", "weakness", "resistance", "retreat",
}
# Deliberately NOT ignored, because real card names contain them: "Energy" (Double Turbo
# Energy), "Special" (Special Charm), "Ace" (Ace Trainer), "Mega", "Tera", "Radiant".


def _is_label(word):
    """True for label words like BASIC / STAGE 2 / TRAINER, including clipped OCR reads."""
    w = re.sub(r"[^a-z]", "", str(word).lower())
    if not w:
        return False
    if w in IGNORE_TITLE_WORDS:
        return True
    if len(w) >= 4:
        for label in IGNORE_TITLE_WORDS:
            if len(label) < 4:
                continue
            # "basi" is a clipped "basic"; "stagei" is "stage" plus a stray character.
            if label.startswith(w) or (len(label) >= 5 and w.startswith(label)):
                return True
    return False

# "227/182" style: digits only on both sides.
DIGIT_SLASH_RE = re.compile(r"(?<!\d)(\d{1,3})\s*/\s*(\d{1,3})")  # up to 3 digits; trailing noise (a star icon) is ignored
# "TG05/TG30", "GG12/GG70", "SV045/SV122" style.
PREFIXED_SLASH_RE = re.compile(r"\b(TG|GG|SV)\s?(\d{1,3})\s*/\s*(TG|GG|SV)?\s?(\d{1,3})\b")
# "SVP 045", "SWSH123" style promos.
PROMO_RE = re.compile(r"\b(SVP|SWSH|SM|XY|BW|HGSS|DP|TG|GG)\s?-?\s?(\d{1,3})\b")

EARLY_EXIT_TITLE_SCORE = 6.0  # about 7+ letters read at roughly 85% confidence

# OCR bands as (label, y0, y1, x0, x1) fractions of the normalized card. Pokemon names sit
# right of the Basic/Stage label; Trainer and Energy names sit lower, under the
# Supporter/Item ribbon. The collector number is in the bottom-left or bottom-right corner.
TITLE_BANDS = [
    # Measured on a real Stage 2 card after correct alignment: the name's letters span about
    # 3.8%-7.2% of the card height and the "Evolves from" line starts at ~8.8%, so the first
    # band stops short of it. Later bands are looser to absorb alignment error.
    ("pokemon_name", .026, .086, .12, .72),
    ("name_loose", .018, .096, .10, .74),
    ("trainer_name", .045, .118, .03, .82),
]
NUMBER_BANDS = [
    # The number sits at ~94.5% height; the copyright line is just below (~97%), so the tight
    # bands exclude it. Modern cards print the number bottom-left, older ones bottom-right.
    ("bottom_left", .925, .962, .02, .31),
    ("bottom_right", .925, .962, .60, .98),
    # Full-art and illustration-rare layouts print the number about 3% higher, above a thick
    # bottom border, so the bands above stop just below it (measured on a real Aurorus scan).
    ("number_row_left", .893, .938, .02, .40),
    ("bottom_left_wide", .915, .975, .02, .50),
    ("lower_left", .900, .990, .02, .50),
]


def _crop(gray, band):
    _, y0, y1, x0, x1 = band
    h, w = gray.shape[:2]
    return gray[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]


def debug_overlay(card_bgr, max_width=520):
    """Normalized card with the OCR bands drawn on it (green = title, red = number).

    Shows at a glance whether the bands line up with the printed text. If they don't, the
    card boundary or the band positions are the problem, not the OCR itself.
    """
    img = card_bgr.copy()
    h, w = img.shape[:2]
    thick = max(2, w // 300)
    for color, bands in (((0, 160, 0), TITLE_BANDS), ((0, 0, 220), NUMBER_BANDS)):
        for label, y0, y1, x0, x1 in bands:
            p1, p2 = (int(w * x0), int(h * y0)), (int(w * x1), int(h * y1))
            cv2.rectangle(img, p1, p2, color, thick)
            cv2.putText(img, label, (p1[0] + 6, p2[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        w / 1500 * 0.9, color, thick)
    scale = max_width / float(w)
    return cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


NUMBER_WHITELIST_DIGITS = "0123456789/"
# No space or hyphen in whitelists: pytesseract splits the config on spaces.
NUMBER_WHITELIST_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/"


def _pad(binary):
    return cv2.copyMakeBorder(binary, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


def _otsu(gray):
    out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if np.mean(out) < 127:
        out = cv2.bitwise_not(out)
    return out


def _blackhat(gray):
    """Isolate dark, thin strokes (text) from a busy or uneven background.

    Plain Otsu thresholding turns busy artwork into noise. Black-hat keeps only dark
    features narrower than the kernel, which is what printed text looks like.
    """
    k = max(9, int(gray.shape[0] * 0.30) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    bh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
    text = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.bitwise_not(text)  # black text on white


def _adaptive(gray):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    block = max(15, (gray.shape[0] // 3) | 1)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 12)


def _prepare(gray, scale=2.0, mode="otsu"):
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if mode == "blackhat":
        return _pad(_blackhat(gray))
    if mode == "adaptive":
        return _pad(_adaptive(gray))
    return _pad(_otsu(gray))


def _read(gray, psm, whitelist=None, mode="otsu", scale=2.0):
    config = f"--psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    try:
        return pytesseract.image_to_string(_prepare(gray, scale, mode), config=config).strip()
    except (pytesseract.TesseractError, RuntimeError):
        return ""  # Tesseract occasionally crashes on blank or degenerate crops


def _clean_name(text):
    # Digits are dropped on purpose so HP values like "160" never end up in the name.
    text = re.sub(r"[^A-Za-z'’ .-]", " ", text)
    words = []
    for word in text.split():
        clean = re.sub(r"[^A-Za-z'’-]", "", word)
        if len(clean) >= 2 and not _is_label(clean):
            words.append(clean)
    # Drop trailing 1-2 letter ALL-CAPS fragments (usually a clipped "HP" label), but keep
    # real name suffixes like EX and GX.
    while len(words) > 1 and re.fullmatch(r"[A-Z]{1,2}", words[-1]) and words[-1] not in {"EX", "GX"}:
        words.pop()
    return " ".join(words).strip()


def _read_title(gray_band):
    """Read one title band with several preprocessing modes.

    Returns (cleaned_name, score, mode, raw_text) tuples. Score uses Tesseract's own
    word confidences, so garbage from busy artwork scores low even if it is long.
    """
    results = []
    for mode in ("otsu", "blackhat", "adaptive"):
        img = _prepare(gray_band, scale=1.0, mode=mode)
        try:
            data = pytesseract.image_to_data(
                img, config="--psm 7", output_type=pytesseract.Output.DICT
            )
        except (pytesseract.TesseractError, RuntimeError):
            continue
        kept, score = [], 0.0
        for text, conf in zip(data["text"], data["conf"]):
            text = text.strip()
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                continue
            if not text or conf < 35:
                continue
            kept.append(text)
            letters = re.sub(r"[^A-Za-z]", "", text)
            if len(letters) >= 3 and not _is_label(letters):
                score += (conf / 100.0) * len(letters)
        cleaned = _clean_name(" ".join(kept))
        if cleaned:
            results.append((cleaned, score, mode, " ".join(kept)))
    return results


def _plausible_number(num, total):
    """Reject collector numbers no real card has (0/5, 1/1, 92/0, 96/980).

    Sets print between about 12 and 400 cards, and secret rares number past the total
    (092/088), so the numerator may exceed it by up to ~200. Garbage that slips through looks
    like evidence to the matcher and, worse, makes a wrong orientation look like a good read.
    """
    return 10 <= total <= 500 and 1 <= num <= total + 200


def _name_variants(name):
    """The read itself plus versions with trailing 1-2 letter fragments removed.

    "Aurorus At" -> ["Aurorus At", "Aurorus"]. The fragment is usually the start of the
    "Evolves from" line or a clipped HP label. Real suffixes (ex, GX, V) are kept.
    """
    words = name.split()
    out = [name]
    while len(words) > 1 and len(words[-1].strip(".'’-")) <= 2 and words[-1].lower() not in {"ex", "gx", "v"}:
        words = words[:-1]
        out.append(" ".join(words))
    return out


def _extract(card_bgr):
    h, w = card_bgr.shape[:2]
    gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)

    # ---- Title -------------------------------------------------------------
    # Tight single-line bands. Pokemon names sit right of the Basic/Stage label; Trainer
    # and Energy names sit lower, under the Supporter/Item ribbon.
    title_reads = []  # (cleaned, score, "band/mode", raw)
    for band in TITLE_BANDS:
        label, crop = band[0], _crop(gray, band)
        if not crop.size:
            continue
        for cleaned, score, mode, raw in _read_title(crop):
            title_reads.append((cleaned, score, f"{label}/{mode}", raw))
        # Early exit: a confident read means the remaining bands aren't needed.
        if title_reads and max(t[1] for t in title_reads) >= EARLY_EXIT_TITLE_SCORE:
            break

    title_reads.sort(key=lambda it: it[1], reverse=True)
    good = [t for t in title_reads if t[1] > 0]
    title_score = good[0][1] if good else 0.0
    if good:
        name = _name_variants(good[0][0])[-1]    # most trimmed form of the best read
    else:
        name = title_reads[0][0] if title_reads else ""
    # Every read, trimmed forms first, so the database search sees the cleanest guess first.
    # Each read also gets a form without a short leading fragment ("wre Bastiodon" ->
    # "Bastiodon"). Those are extra candidates only; the displayed name is never altered that
    # way, because real names start with short words too ("N's Zoroark", "Mr. Mime").
    name_candidates = []
    for t in good:
        variants = list(reversed(_name_variants(t[0])))
        name_candidates.extend(variants)
        words = variants[0].split()
        if len(words) > 1 and len(words[0].strip(".'’-")) <= 3 and len(" ".join(words[1:])) >= 5:
            name_candidates.append(" ".join(words[1:]))
    name_candidates = list(dict.fromkeys(name_candidates))

    # ---- Collector number --------------------------------------------------
    # Strict patterns only. If nothing matches, the number stays empty, which is
    # better than feeding garbage into the matcher as if it were evidence.
    number_candidates = []  # (quality, value, region label, raw text)
    for band in NUMBER_BANDS:
        label, crop = band[0], _crop(gray, band)
        if not crop.size:
            continue
        for mode in ("blackhat", "otsu"):
            # Pass 1: digits and slash only, at two scales (text this small is scale-sensitive).
            for scale in (1.5, 2.0):
                raw = _read(crop, 7, NUMBER_WHITELIST_DIGITS, mode, scale)
                found = DIGIT_SLASH_RE.search(raw)
                if found and _plausible_number(int(found.group(1)), int(found.group(2))):
                    value = f"{int(found.group(1))}/{int(found.group(2))}"
                    number_candidates.append((3, value, f"{label}/{mode}@{scale}", raw))

            # Pass 2: letters allowed, for prefixed and promo numbers.
            raw = _read(crop, 7, NUMBER_WHITELIST_LETTERS, mode)
            upper = raw.upper()
            found = PREFIXED_SLASH_RE.search(upper)
            if found:
                p1, n1, p2, n2 = found.groups()
                value = f"{p1}{int(n1):02d}/{(p2 or p1)}{int(n2):02d}"
                number_candidates.append((3, value, f"{label}/{mode}", raw))
                continue
            found = PROMO_RE.search(upper)
            if found:
                value = f"{found.group(1)} {int(found.group(2)):03d}"
                number_candidates.append((2, value, f"{label}/{mode}", raw))

        # Early exit: two agreeing reads are enough, so skip the remaining regions.
        if number_candidates and Counter(v for _, v, _, _ in number_candidates).most_common(1)[0][1] >= 2:
            break

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
    debug = "Title OCR attempts (score = confidence-weighted letters):\n"
    debug += "\n".join(
        f"- {label}: {raw!r} -> {cleaned!r} (score {score:.1f})"
        for cleaned, score, label, raw in title_reads[:9]
    )
    debug += "\nNumber OCR attempts:\n"
    debug += "\n".join(
        f"- {label}: {raw!r} -> {value!r}" for _, value, label, raw in number_candidates[:8]
    )
    if not good:
        debug += "\nNo title text was read with usable confidence."
    if not number_candidates:
        debug += "\nNo collector number matched a strict pattern."

    return {
        "name": name,
        "name_candidates": name_candidates,
        "number": number,
        "printed_total": printed_total,
        "set_hint": set_hint,
        "title_score": title_score,
        "rotated180": False,
        "debug": debug,
    }


def _quality(signals):
    """How trustworthy a read looks: confident title letters plus a strict collector number."""
    return signals["title_score"] + (4.0 if signals["number"] else 0.0)


def extract_card_signals(card_bgr):
    """Read the name and collector number, correcting an upside-down card if needed.

    Card detection can only make the card portrait; it cannot tell right-side-up from
    upside-down. When the first read is weak, the card is rotated 180 degrees and read again,
    and the better result wins. The result's "rotated180" flag says which one it was, so the
    caller can rotate the picture it shows to match.
    """
    result = _extract(card_bgr)
    # Low-resolution scans read the right name with a modest score (about 2.5), while an
    # upside-down card reads close to nothing, so only a truly unreadable title is retried.
    if result["title_score"] >= 2.0:
        return result
    flipped = _extract(cv2.rotate(card_bgr, cv2.ROTATE_180))
    if _quality(flipped) > _quality(result) + 1.5:
        flipped["rotated180"] = True
        flipped["debug"] = "Card was upside down; read again after rotating 180 degrees.\n" + flipped["debug"]
        return flipped
    return result