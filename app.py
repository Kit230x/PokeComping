import streamlit as st
import requests
from PIL import Image
import pytesseract
import cv2
import numpy as np
import io
import re

st.set_page_config(page_title="TCGComping - Automated Zero-Cost", layout="wide")


# ---------------------------------------------------------
# TCGdex API (100% Free - No Keys, No Paywalls)
# ---------------------------------------------------------
def search_tcgdex(card_name):
    """Safely searches TCGdex without crashing on NoneType pricing or missing variants."""
    clean_name = card_name.strip()
    if not clean_name or len(clean_name) < 2:
        return []

    url = f"https://api.tcgdex.net/v2/en/cards?name={clean_name}"
    try:
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            brief_list = res.json()
            if not isinstance(brief_list, list):
                return []

            matches = []
            for item in brief_list[:6]:
                card_id = item.get("id")
                detail_res = requests.get(f"https://api.tcgdex.net/v2/en/cards/{card_id}", timeout=6)
                if detail_res.status_code == 200:
                    d = detail_res.json()

                    # Safely handle NoneType pricing blocks
                    pricing_data = d.get("pricing") or {}
                    tcg_pricing = pricing_data.get("tcgplayer") or {}

                    market_val = 0.0
                    # TCGdex uses 'normal', 'reverse', and 'holo'
                    for variant in ["normal", "reverse", "holo"]:
                        var_obj = tcg_pricing.get(variant)
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
            return matches
    except Exception as e:
        print(f"TCGdex error: {e}")
    return []

# ---------------------------------------------------------
# Local Computer Vision & OCR
# ---------------------------------------------------------
def extract_card_name_ocr(image_bytes):
    """
    Finds the card contour to discard external scanner margins,
    then crops and processes the top title bar for Tesseract.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return ""

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the card contour to ignore white/black margins
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 30, 120)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        card_crop = img
        if contours:
            # Find the largest bounding box (the card itself)
            largest = max(contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(largest)
            # Ensure it is at least 30% of the image size
            if w_box > img.shape[1] * 0.3 and h_box > img.shape[0] * 0.3:
                card_crop = img[y:y + h_box, x:x + w_box]

        h, w, _ = card_crop.shape

        # Standard Modern Pokemon header coordinates relative to the card itself:
        # Top 4% to 15% height, 6% to 75% width (isolates title, skips stage & HP)
        title_region = card_crop[int(h * 0.04):int(h * 0.15), int(w * 0.06):int(w * 0.75)]

        # Preprocessing: Grayscale, Upscaling for Tesseract, Adaptive Threshold
        title_gray = cv2.cvtColor(title_region, cv2.COLOR_BGR2GRAY)
        title_resized = cv2.resize(title_gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        thresh = cv2.threshold(title_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # PSM 7: Treat the image as a single text line
        raw_text = pytesseract.image_to_string(thresh, config="--psm 7")

        # Strip noise, symbols, and artifacts
        cleaned = re.sub(r'[^a-zA-Z\s]', '', raw_text).strip()
        first_line = cleaned.split("\n")[0].strip()
        return first_line
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""


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

                # 1. OCR automatically reads card header
                detected_name = extract_card_name_ocr(raw_bytes)

                # 2. Query TCGdex directly without any keys
                matches = search_tcgdex(detected_name) if detected_name else []

                st.session_state.batch.append({
                    "id": f"{file.name}_{len(st.session_state.batch)}",
                    "filename": file.name,
                    "local_bytes": raw_bytes,
                    "ocr_name": detected_name,
                    "matched_card": matches[0] if matches else None,
                    "suggestions": matches,
                    "selected_index": 0
                })
        st.rerun()

# ---------------------------------------------------------
# Dashboard & Valuation
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
    st.info(
        "Drop your scans in the sidebar and click 'Scan & Auto-Comp Batch'. It will auto-read names and comp cards without keys or subscriptions.")
else:
    for card in st.session_state.batch:
        with st.container():
            col_scan, col_web, col_info, col_del = st.columns([1.5, 1.5, 3.5, 1])

            # Local Scan Image
            with col_scan:
                st.caption(f"Scan:")
                st.image(Image.open(io.BytesIO(card["local_bytes"])), width="stretch")
            # High-res Reference Card Image from TCGdex
            with col_web:
                st.caption("**Online Match:**")
                if card["matched_card"] and card["matched_card"].get("image_url"):
                   st.image(card["matched_card"]["image_url"], width="stretch")
                else:
                    st.write("*(No visual match found)*")

            # Comps & Suggestions
            with col_info:
                st.write(f"**OCR Read:** `{card['ocr_name'] or 'Could not detect text'}`")

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
                    # Fallback input if the scan crop was completely illegible
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