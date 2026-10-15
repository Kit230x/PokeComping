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
    """
    Queries TCGdex open-source API for Pokemon cards and pulls
    card images along with live TCGplayer market prices.
    """
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

            # Take top 6 matching cards and fetch their details/pricing
            matches = []
            for item in brief_list[:6]:
                card_id = item.get("id")
                detail_res = requests.get(f"https://api.tcgdex.net/v2/en/cards/{card_id}", timeout=6)
                if detail_res.status_code == 200:
                    d = detail_res.json()

                    # Extract TCGPlayer Market Price
                    pricing = d.get("pricing", {}).get("tcgplayer", {})
                    market_val = 0.0
                    for variant in ["normal", "holofoil", "reverse-holofoil", "unlimited"]:
                        if variant in pricing and pricing[variant].get("marketPrice"):
                            market_val = float(pricing[variant]["marketPrice"])
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
    Crops the top header area of a centered ScanSnap scan
    and extracts the card name using PyTesseract.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return ""

        h, w, _ = img.shape
        # Top 15% height, 8% to 75% width (isolates title and drops HP/energy icons)
        top_crop = img[int(h * 0.03):int(h * 0.15), int(w * 0.08):int(w * 0.75)]

        # Preprocess: Grayscale and Otsu binarization
        gray = cv2.cvtColor(top_crop, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Extract text via Tesseract (single text line mode)
        text = pytesseract.image_to_string(thresh, config="--psm 7")
        cleaned = re.sub(r'[^a-zA-Z\s]', '', text).strip()
        first_word = cleaned.split("\n")[0].strip()
        return first_word
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