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
# Local Computer Vision & OCR
# ---------------------------------------------------------
def extract_card_info_ocr(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return "", ""

        h, w, _ = img.shape

        # 1. TOP TITLE BOX (Cleaned to strip leading stray characters like 'p')
        title_crop = img[int(h * 0.035):int(h * 0.115), int(w * 0.035):int(w * 0.38)]
        t_gray = cv2.cvtColor(title_crop, cv2.COLOR_BGR2GRAY)
        t_scale = cv2.resize(t_gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        t_blur = cv2.GaussianBlur(t_scale, (3, 3), 0)
        _, t_thresh = cv2.threshold(t_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if np.mean(t_thresh) < 127:
            t_thresh = cv2.bitwise_not(t_thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        t_clean = cv2.morphologyEx(t_thresh, cv2.MORPH_OPEN, kernel)

        raw_title = pytesseract.image_to_string(
            t_clean,
            config="--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        ).strip()

        cleaned_title = re.sub(r'[^a-zA-Z]', '', raw_title).strip()
        # Drop stray single leading lowercase letter artifact (e.g. 'pDeoxys' -> 'Deoxys')
        if len(cleaned_title) > 4 and cleaned_title[0].islower() and cleaned_title[1].isupper():
            cleaned_title = cleaned_title[1:]

        # 2. BOTTOM COLLECTOR NUMBER (Covers BOTH bottom-left and bottom-right across eras)
        # Height: 91% to 98.5% | Width: 0% to 100% across the footer
        num_footer = img[int(h * 0.91):int(h * 0.985), 0:w]
        n_gray = cv2.cvtColor(num_footer, cv2.COLOR_BGR2GRAY)
        n_scale = cv2.resize(n_gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

        # Adaptive thresholding isolates tiny numbers regardless of footer color/glare
        n_thresh = cv2.adaptiveThreshold(
            n_scale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
        )

        raw_num = pytesseract.image_to_string(
            n_thresh,
            config="--psm 6 -c tessedit_char_whitelist=0123456789/"
        ).strip()

        detected_number = ""
        # Match XX/YY (e.g., 16/107, 025/198, 151/165)
        match_slash = re.search(r'(\d{1,4})/\d{1,4}', raw_num)
        if match_slash:
            detected_number = match_slash.group(1).lstrip("0")
        else:
            # Fallback: look for 2-3 digit standalone numbers if slash was missed
            digits = re.findall(r'\b\d{1,3}\b', raw_num)
            if digits:
                detected_number = digits[-1].lstrip("0")

        return cleaned_title, detected_number
    except Exception as e:
        print(f"OCR Error: {e}")
        return "", ""


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

            # Fuzzy match candidates against clean title
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

            # IF COLLECTOR NUMBER DETECTED: Sort exact card number to Index 0
            if detected_num:
                matches.sort(
                    key=lambda x: 0 if x["number"].lstrip("0") == str(detected_num) else 1
                )

            return matches
    except Exception as e:
        print(f"TCGdex error: {e}")
    return []

# ---------------------------------------------------------
# TCGdex API + Fuzzy Matching
# ---------------------------------------------------------
def search_tcgdex(card_name, detected_num=""):
    """
    Queries TCGdex and uses RapidFuzz to resolve OCR misspellings
    (e.g., 'Armalda' -> 'Armaldo').
    """
    clean_name = card_name.strip()
    if not clean_name and not detected_num:
        return []

    # Use first 4 characters as seed to fetch candidates
    seed = clean_name[:4] if len(clean_name) >= 4 else clean_name
    url = f"https://api.tcgdex.net/v2/en/cards?name={seed}" if seed else "https://api.tcgdex.net/v2/en/cards"

    try:
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            brief_list = res.json()
            if not isinstance(brief_list, list) or not brief_list:
                return []

            # Fuzzy match candidate names against OCR text
            name_choices = {item["id"]: item["name"] for item in brief_list}
            scored_matches = process.extract(
                clean_name,
                name_choices,
                scorer=fuzz.ratio,
                limit=6
            )

            matches = []
            for candidate_name, score, card_id in scored_matches:
                d_res = requests.get(f"https://api.tcgdex.net/v2/en/cards/{card_id}", timeout=5)
                if d_res.status_code == 200:
                    d = d_res.json()

                    # Safe null-checked extraction of TCGplayer market price
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
                        "number": d.get("localId", ""),
                        "image_url": f"{d.get('image')}/high.webp" if d.get("image") else None,
                        "market_price": market_val
                    })

            # If collector number was detected, prioritize matching number
            if detected_num:
                matches.sort(
                    key=lambda x: 0 if str(x["number"]).lstrip("0") == str(detected_num) else 1
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

                # 1. Automatic OCR extraction
                detected_name, detected_number = extract_card_info_ocr(raw_bytes)

                # 2. Automated fuzzy query
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

            # Local Scan Image
            with col_scan:
                st.caption(f"Scan:")
                st.image(Image.open(io.BytesIO(card["local_bytes"])), width="stretch")

                # Debug crop inspector
                with st.expander("Show OCR Crop"):
                    nparr = np.frombuffer(card["local_bytes"], np.uint8)
                    raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if raw is not None:
                        h, w, _ = raw.shape
                        title_prev = raw[int(h * 0.035):int(h * 0.115), int(w * 0.035):int(w * 0.38)]
                        footer_prev = raw[int(h * 0.91):int(h * 0.985), 0:w]
                        st.image(title_prev, caption="Title Region", width="stretch")
                        st.image(footer_prev, caption="Footer Number Region", width="stretch")

            # High-res Reference Card Image from TCGdex
            with col_web:
                st.caption("**Online Match:**")
                if card["matched_card"] and card["matched_card"].get("image_url"):
                    st.image(card["matched_card"]["image_url"], width="stretch")
                else:
                    st.write("*(No visual match found)*")

            # Comps & Suggestions
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