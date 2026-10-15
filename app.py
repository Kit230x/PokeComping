import streamlit as st
import requests
from PIL import Image
import pytesseract
import cv2
import numpy as np
import io
import re
from rapidfuzz import process, fuzz

st.set_page_config(page_title="TCGComping - Automated Verification Queue", layout="wide")


# ---------------------------------------------------------
# Computer Vision: Card Boundary Detection & Deskewing
# ---------------------------------------------------------
def order_points(pts):
    """Orders coordinates: [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def auto_align_card(img, target_w=750, target_h=1050):
    """
    Detects the outer boundary of the trading card, removes background
    margins/scanner bed, deskews tilt, and warps to standard 750x1050 px.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection to pick up card borders
    edged = cv2.Canny(blurred, 30, 150)

    # Dilate slightly to close gaps in the edge contour
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edged, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    card_cnt = None
    for c in contours:
        # Require contour to cover at least 30% of the image
        if cv2.contourArea(c) < (img.shape[0] * img.shape[1] * 0.30):
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            card_cnt = approx
            break

    # If 4-point polygon was detected, perform perspective transform
    if card_cnt is not None:
        pts = card_cnt.reshape(4, 2)
        rect = order_points(pts)
        dst = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (target_w, target_h))
        return warped

    # Fallback: if edges blend into background, resize directly to target dimensions
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------
# Local Computer Vision & OCR
# ---------------------------------------------------------
def extract_card_info_ocr(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if raw_img is None:
            return "", ""

        # 1. Standardize card canvas to 750w x 1050h
        img = auto_align_card(raw_img, target_w=750, target_h=1050)

        # -------------------------------------------------------------
        # 2. TITLE EXTRACTION (Standardized coordinates: Y: 40-120, X: 35-430)
        # -------------------------------------------------------------
        title_crop = img[40:120, 35:430]
        t_gray = cv2.cvtColor(title_crop, cv2.COLOR_BGR2GRAY)
        t_scale = cv2.resize(t_gray, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        t_blur = cv2.GaussianBlur(t_scale, (3, 3), 0)
        _, t_thresh = cv2.threshold(t_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if np.mean(t_thresh) < 127:
            t_thresh = cv2.bitwise_not(t_thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        t_clean = cv2.morphologyEx(t_thresh, cv2.MORPH_OPEN, kernel)

        raw_title = pytesseract.image_to_string(t_clean, config="--psm 6").strip()

        ignore_words = {'stage', 'basic', 'evolves', 'from', 'hp', 'vstar', 'vmax'}
        cleaned_words = []
        for word in raw_title.split():
            clean = re.sub(r'[^a-zA-Z]', '', word)
            if len(clean) >= 2 and clean.lower() not in ignore_words:
                cleaned_words.append(clean)

        cleaned_title = " ".join(cleaned_words)

        # -------------------------------------------------------------
        # 3. COLLECTOR NUMBER (Dual-Zone Normalized Scanner)
        # -------------------------------------------------------------
        detected_number = ""

        # Normalized Modern BL: Y: 980-1035, X: 25-260
        modern_bl = img[980:1035, 25:260]

        # Normalized Vintage/EX BR: Y: 965-1035, X: 520-720
        vintage_br = img[965:1035, 520:720]

        for reg in [modern_bl, vintage_br]:
            gray = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY)
            scale = cv2.resize(gray, (0, 0), fx=3.5, fy=3.5, interpolation=cv2.INTER_CUBIC)

            _, thresh_norm = cv2.threshold(scale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh_inv = cv2.bitwise_not(thresh_norm)

            for candidate in [thresh_norm, thresh_inv]:
                if np.mean(candidate) < 127:
                    candidate = cv2.bitwise_not(candidate)

                padded = cv2.copyMakeBorder(candidate, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)

                raw_corner = pytesseract.image_to_string(
                    padded,
                    config="--psm 7 -c tessedit_char_whitelist=0123456789/"
                ).strip()

                match_slash = re.search(r'(\d{1,4})/\d{1,4}', raw_corner)
                if match_slash:
                    detected_number = match_slash.group(1).lstrip("0")
                    break

            if detected_number:
                break

        return cleaned_title, detected_number
    except Exception as e:
        print(f"OCR Error: {e}")
        return "", ""


# ---------------------------------------------------------
# TCGdex API + Fuzzy Matching
# ---------------------------------------------------------
def search_tcgdex(card_name, detected_num=""):
    clean_name = card_name.strip()
    if not clean_name:
        return []

    seed = clean_name[:4] if len(clean_name) >= 4 else clean_name
    url = f"https://api.tcgdex.net/v2/en/cards?name={seed}"

    try:
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            brief_list = res.json()
            if not isinstance(brief_list, list) or not brief_list:
                return []

            name_choices = {item["id"]: item["name"] for item in brief_list}
            scored_matches = process.extract(
                clean_name,
                name_choices,
                scorer=fuzz.ratio,
                limit=10
            )

            matches = []
            for candidate_name, score, card_id in scored_matches:
                d_res = requests.get(f"https://api.tcgdex.net/v2/en/cards/{card_id}", timeout=5)
                if d_res.status_code == 200:
                    d = d_res.json()
                    pricing = (d.get("pricing") or {}).get("tcgplayer") or {}

                    market_val = 0.0
                    for variant in ["normal", "reverse", "holo"]:
                        var_obj = pricing.get(variant)
                        if isinstance(var_obj, dict) and var_obj.get("marketPrice"):
                            market_val = float(var_obj["marketPrice"])
                            break

                    matches.append({
                        "id": d.get("id"),
                        "name": d.get("name"),
                        "set_name": d.get("set", {}).get("name", "Unknown Set"),
                        "number": str(d.get("localId", "")).strip(),
                        "image_url": f"{d.get('image')}/high.webp" if d.get("image") else None,
                        "market_price": market_val
                    })

            if detected_num:
                matches.sort(
                    key=lambda x: 0 if str(x["number"]).lstrip("0") == str(detected_num).lstrip("0") else 1
                )

            return matches
    except Exception as e:
        print(f"TCGdex error: {e}")
    return []


# ---------------------------------------------------------
# App Session & Batch State
# ---------------------------------------------------------
if "batch" not in st.session_state:
    st.session_state.batch = []

st.sidebar.title("📥 Batch Intake (NM Only)")
uploaded_files = st.sidebar.file_uploader(
    "Drop Front Scans Here",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if st.sidebar.button("⚡ Scan & Auto-Comp Batch", type="primary"):
    if uploaded_files:
        existing = {c["filename"] for c in st.session_state.batch}
        for file in uploaded_files:
            if file.name not in existing:
                raw_bytes = file.read()

                detected_name, detected_number = extract_card_info_ocr(raw_bytes)
                matches = search_tcgdex(detected_name, detected_number)

                st.session_state.batch.append({
                    "id": f"{file.name}_{len(st.session_state.batch)}",
                    "filename": file.name,
                    "local_bytes": raw_bytes,
                    "ocr_name": detected_name,
                    "ocr_number": detected_number,
                    "matched_card": matches[0] if matches else None,
                    "suggestions": matches,
                    "selected_index": 0
                })
        st.rerun()

# ---------------------------------------------------------
# Header & Total Valuation Toolbar
# ---------------------------------------------------------
st.title("🗂️ TCGComping Verification Queue")

total_cards = len(st.session_state.batch)
total_value = sum(
    c["matched_card"]["market_price"]
    for c in st.session_state.batch
    if c.get("matched_card")
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Cards", f"{total_cards}")
c2.metric("Total Market Value", f"${total_value:,.2f}")
if total_cards > 0:
    c3.metric("Average / Card", f"${(total_value / total_cards):,.2f}")

sort_choice = c4.selectbox("Sort By:", ["Scan Order", "Price: High to Low", "Price: Low to High"])
if sort_choice == "Price: High to Low":
    st.session_state.batch.sort(key=lambda x: (x["matched_card"]["market_price"] if x["matched_card"] else 0.0),
                                reverse=True)
elif sort_choice == "Price: Low to High":
    st.session_state.batch.sort(key=lambda x: (x["matched_card"]["market_price"] if x["matched_card"] else 0.0))

st.markdown("---")

# ---------------------------------------------------------
# Verification Queue
# ---------------------------------------------------------
if not st.session_state.batch:
    st.info("Drop your scans in the sidebar and click 'Scan & Auto-Comp Batch'. Cards will comp automatically.")
else:
    for card in st.session_state.batch:
        with st.container():
            col_scan, col_web, col_info, col_del = st.columns([1.5, 1.5, 3.5, 1])

            with col_scan:
                st.caption(f"Scan:")
                st.image(Image.open(io.BytesIO(card["local_bytes"])), width="stretch")

                with st.expander("Show OCR Crop"):
                    nparr = np.frombuffer(card["local_bytes"], np.uint8)
                    raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if raw is not None:
                        aligned = auto_align_card(raw, 750, 1050)
                        title_prev = aligned[40:120, 35:430]
                        modern_bl_prev = aligned[980:1035, 25:260]
                        vintage_br_prev = aligned[965:1035, 520:720]
                        st.image(aligned, caption="Detected & Deskewed Card", width="stretch")
                        st.image(title_prev, caption="Title Region", width="stretch")
                        st.image(modern_bl_prev, caption="Modern BL Number", width="stretch")
                        st.image(vintage_br_prev, caption="Vintage BR Number", width="stretch")

            with col_web:
                st.caption("**Online Match:**")
                if card["matched_card"] and card["matched_card"].get("image_url"):
                    st.image(card["matched_card"]["image_url"], width="stretch")
                else:
                    st.write("*(No visual match found)*")

            with col_info:
                display_ocr = card['ocr_name'] or "Not read"
                if card.get('ocr_number'):
                    display_ocr += f" (#{card['ocr_number']})"
                st.write(f"**OCR Detected:** `{display_ocr}`")

                if card["suggestions"]:
                    opts = [
                        f"{m['name']} — {m['set_name']} (#{m['number']}) | ${m['market_price']:.2f}"
                        for m in card["suggestions"]
                    ]
                    selected = st.selectbox(
                        "Swap match variant:",
                        opts,
                        index=card["selected_index"],
                        key=f"sel_{card['id']}"
                    )
                    idx = opts.index(selected)
                    if idx != card["selected_index"]:
                        card["selected_index"] = idx
                        card["matched_card"] = card["suggestions"][idx]
                        st.rerun()

                    m = card["matched_card"]
                    st.markdown(f"### Market Price: :green[**${m['market_price']:.2f}**]")
                    st.caption(f"{m['set_name']} • #{m['number']} • Near Mint")
                else:
                    new_term = st.text_input("Manual Search (Fallback)", value=card["ocr_name"],
                                             key=f"manual_{card['id']}")
                    if st.button("Retry Search", key=f"retry_{card['id']}"):
                        m_list = search_tcgdex(new_term)
                        card["suggestions"] = m_list
                        if m_list:
                            card["matched_card"] = m_list[0]
                        st.rerun()

            with col_del:
                st.write("###")
                if st.button("🗑️ Exclude", key=f"del_{card['id']}"):
                    st.session_state.batch = [c for c in st.session_state.batch if c["id"] != card["id"]]
                    st.rerun()

            st.divider()

# ---------------------------------------------------------
# CSV Export
# ---------------------------------------------------------
if any(c.get("matched_card") for c in st.session_state.batch):
    csv_rows = ["Product Line,Product Name,Set Name,Number,Condition,Printing,Price,Quantity"]
    for c in st.session_state.batch:
        if c.get("matched_card"):
            m = c["matched_card"]
            csv_rows.append(
                f'Pokemon,"{m["name"]}","{m["set_name"]}","{m["number"]}",Near Mint,Normal,{m["market_price"]:.2f},1')

    st.download_button(
        "📥 Download TCGplayer Ready CSV",
        data="\n".join(csv_rows),
        file_name="tcgplayer_batch.csv",
        mime="text/csv"
    )