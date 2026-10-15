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
def extract_card_info_ocr(image_bytes):
    """
    Extracts text from both the top header (name) and bottom footer (collector number)
    using bilateral filtering to ignore foil noise.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return "", ""

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Detect card edges to trim scanner margins
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 40, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        card_crop = img
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(largest)
            # Ensure detected contour is reasonably card-sized
            if w_box > img.shape[1] * 0.4 and h_box > img.shape[0] * 0.4:
                card_crop = img[y:y + h_box, x:x + w_box]

        h, w, _ = card_crop.shape

        # --- A. BOTTOM COLLECTOR NUMBER (Most reliable for modern cards) ---
        # Bottom 7% of the card, left 45% (where 064/165 or 125/198 lives)
        num_crop = card_crop[int(h * 0.92):int(h * 0.985), int(w * 0.04):int(w * 0.45)]
        num_gray = cv2.cvtColor(num_crop, cv2.COLOR_BGR2GRAY)

        # Enlarge for Tesseract clarity
        num_resized = cv2.resize(num_gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        # Bilateral filter strips foil glare while keeping sharp letter boundaries
        num_clean = cv2.bilateralFilter(num_resized, 9, 75, 75)
        num_thresh = cv2.adaptiveThreshold(
            num_clean, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
        )

        # Look strictly for digits and slashes (e.g., 064/165 or 25)
        num_text = pytesseract.image_to_string(
            num_thresh,
            config="--psm 6 -c tessedit_char_whitelist=0123456789/"
        ).strip()

        # Extract "XXX/YYY" or pure numbers
        match_slash = re.search(r'(\d{2,4}/\d{2,4})', num_text)
        detected_number = match_slash.group(1).split("/")[0] if match_slash else ""

        # --- B. TOP CARD NAME ---
        # Top 4% to 14% height, 8% to 75% width
        title_crop = card_crop[int(h * 0.04):int(h * 0.14), int(w * 0.08):int(w * 0.72)]
        title_gray = cv2.cvtColor(title_crop, cv2.COLOR_BGR2GRAY)
        title_resized = cv2.resize(title_gray, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        title_clean = cv2.bilateralFilter(title_resized, 9, 75, 75)

        # Test whether text is dark-on-light or light-on-dark (Full Art / Dark cards)
        mean_val = np.mean(title_clean)
        if mean_val < 120:  # Dark background, white text
            _, title_thresh = cv2.threshold(title_clean, 140, 255, cv2.THRESH_BINARY)
        else:  # Light background, dark text
            _, title_thresh = cv2.threshold(title_clean, 120, 255, cv2.THRESH_BINARY_INV)

        raw_title = pytesseract.image_to_string(title_thresh, config="--psm 7").strip()
        cleaned_title = re.sub(r'[^a-zA-Z\s]', '', raw_title).strip()
        # Take the most prominent word (e.g., "Alakazam" or "Charizard")
        words = [w for w in cleaned_title.split() if len(w) >= 3]
        detected_name = words[0] if words else ""

        return detected_name, detected_number
    except Exception as e:
        print(f"OCR Error: {e}")
        return "", ""

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

                # 1. Run the updated OCR (extracts both name and collector number)
                detected_name, detected_number = extract_card_info_ocr(raw_bytes)

                # 2. Pick whichever detected value is strongest
                search_term = detected_name if detected_name else detected_number

                # 3. Query TCGdex
                matches = search_tcgdex(search_term) if search_term else []

                st.session_state.batch.append({
                    "id": f"{file.name}_{len(st.session_state.batch)}",
                    "filename": file.name,
                    "local_bytes": raw_bytes,
                    "ocr_name": f"{detected_name} (#{detected_number})" if detected_number else detected_name,
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