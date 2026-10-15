import io
import pandas as pd
import streamlit as st
from PIL import Image

from tcgcomping.image_processor import decode_image, normalize_card
from tcgcomping.ocr_engine import extract_card_signals
from tcgcomping.card_matcher import rank_candidates
from tcgcomping.card_database import search_cards

st.set_page_config(page_title="TCGComping — Verification Queue", layout="wide")
st.title("🗂️ TCGComping — Verification Queue")
st.caption("Modular Pokémon card identification with confidence-aware suggestions. Review uncertain matches before pricing or export.")

if "batch" not in st.session_state:
    st.session_state.batch = []

with st.sidebar:
    st.header("📥 Batch intake")
    uploads = st.file_uploader(
        "Upload front scans",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    st.subheader("Identification settings")
    game = st.selectbox("Game", ["Pokémon"], help="The pipeline is structured so game-specific adapters can be added later.")
    allow_auto = st.checkbox("Allow high-confidence auto-selection", value=True)
    threshold = st.slider("Auto-selection confidence", 0.50, 0.98, 0.88, 0.01)
    run_scan = st.button("⚡ Scan & identify batch", type="primary", use_container_width=True)

if run_scan:
    if not uploads:
        st.warning("Upload at least one card scan first.")
    else:
        existing = {c["filename"] for c in st.session_state.batch}
        progress = st.progress(0)
        status = st.empty()
        new_items = []
        for i, uploaded in enumerate(uploads):
            if uploaded.name in existing:
                continue
            raw_bytes = uploaded.getvalue()
            try:
                image = decode_image(raw_bytes)
                aligned, alignment_note = normalize_card(image)
                signals = extract_card_signals(aligned)
                candidates = search_cards(signals["name"])
                ranked = rank_candidates(signals, candidates, aligned)
                best = ranked[0] if ranked else None
                auto_ok = bool(best and best["confidence"] >= threshold and best["evidence"]["name_score"] >= 0.72)
                chosen = best if (allow_auto and auto_ok) else None
                new_items.append({
                    "id": f"{uploaded.name}_{len(st.session_state.batch) + len(new_items)}",
                    "filename": uploaded.name,
                    "local_bytes": raw_bytes,
                    "ocr_name": signals["name"],
                    "ocr_number": signals["number"],
                    "ocr_set_hint": signals["set_hint"],
                    "ocr_debug": signals["debug"],
                    "alignment_note": alignment_note,
                    "candidates": ranked,
                    "matched_card": chosen,
                    "confirmed": False,
                    "selected_index": 0,
                    "status": "Auto-suggested" if chosen else ("Needs review" if ranked else "No candidates"),
                })
            except Exception as exc:
                new_items.append({
                    "id": f"{uploaded.name}_{len(st.session_state.batch) + len(new_items)}",
                    "filename": uploaded.name,
                    "local_bytes": raw_bytes,
                    "ocr_name": "",
                    "ocr_number": "",
                    "ocr_set_hint": "",
                    "ocr_debug": f"Processing error: {exc}",
                    "alignment_note": "Image processing failed",
                    "candidates": [],
                    "matched_card": None,
                    "confirmed": False,
                    "selected_index": 0,
                    "status": "Processing error",
                })
            progress.progress((i + 1) / len(uploads))
            status.write(f"Processed {i + 1} of {len(uploads)} uploads")
        st.session_state.batch.extend(new_items)
        st.rerun()

cards = st.session_state.batch
valued = [c for c in cards if c.get("matched_card")]
total_value = sum((c["matched_card"].get("market_price") or 0.0) for c in valued)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Cards in queue", len(cards))
m2.metric("Selected matches", len(valued))
m3.metric("Confirmed matches", sum(bool(c.get("confirmed")) for c in cards))
m4.metric("Selected market value", f"${total_value:,.2f}")

if cards:
    sort_mode = st.selectbox("Sort queue", ["Scan order", "Confidence: low to high", "Price: high to low"])
    if sort_mode == "Confidence: low to high":
        cards.sort(key=lambda c: (c.get("matched_card") or {}).get("confidence", 0))
    elif sort_mode == "Price: high to low":
        cards.sort(key=lambda c: (c.get("matched_card") or {}).get("market_price", 0), reverse=True)

    for card in list(cards):
        with st.container(border=True):
            scan_col, match_col, info_col, action_col = st.columns([1.2, 1.25, 3.1, 0.8])
            with scan_col:
                st.caption(card["filename"])
                st.image(Image.open(io.BytesIO(card["local_bytes"])), use_container_width=True)
                st.caption(card["alignment_note"])
            with match_col:
                st.markdown("**Suggested match**")
                match = card.get("matched_card")
                if match and match.get("image_url"):
                    st.image(match["image_url"], use_container_width=True)
                elif match:
                    st.info("No database image available")
                else:
                    st.warning("No match selected")
            with info_col:
                st.write(f"**OCR name:** `{card.get('ocr_name') or 'Not read'}`")
                st.write(f"**Collector number:** `{card.get('ocr_number') or 'Not read'}`")
                if card.get("ocr_set_hint"):
                    st.write(f"**Set hint:** `{card['ocr_set_hint']}`")
                if card.get("ocr_debug"):
                    with st.expander("OCR diagnostics"):
                        st.code(card["ocr_debug"])
                candidates = card.get("candidates", [])
                if candidates:
                    labels = [
                        f"{c['name']} — {c['set_name']} (#{c['number']}) | {c['confidence']:.0%} | ${c['market_price']:.2f}"
                        for c in candidates
                    ]
                    current_index = next((i for i, c in enumerate(candidates) if match and c["id"] == match["id"]), 0)
                    chosen_label = st.selectbox(
                        "Candidate review",
                        labels,
                        index=min(current_index, len(labels)-1),
                        key=f"candidate_{card['id']}",
                    )
                    selected_idx = labels.index(chosen_label)
                    if not match or candidates[selected_idx]["id"] != match["id"]:
                        card["matched_card"] = candidates[selected_idx]
                        card["confirmed"] = False
                        card["status"] = "Manually selected — not yet confirmed"
                        st.rerun()
                    match = card.get("matched_card")
                    if match:
                        st.markdown(f"**Market price:** ${match.get('market_price', 0.0):.2f}")
                        st.caption(
                            f"{match['set_name']} • #{match['number']} • confidence {match['confidence']:.0%}"
                        )
                        with st.expander("Why this candidate ranked here"):
                            st.json(match["evidence"])
                        if not card.get("confirmed"):
                            if st.button("✓ Confirm identification", key=f"confirm_{card['id']}"):
                                card["confirmed"] = True
                                card["status"] = "Confirmed"
                                st.rerun()
                        else:
                            st.success("Identification confirmed")
                else:
                    st.write("No candidates found. Try improving the scan or enter a name below.")
                manual_term = st.text_input("Search by name", value=card.get("ocr_name", ""), key=f"manual_{card['id']}")
                if st.button("Retry search", key=f"retry_{card['id']}"):
                    fresh = search_cards(manual_term)
                    card["candidates"] = rank_candidates(
                        {"name": manual_term, "number": card.get("ocr_number", ""), "set_hint": card.get("ocr_set_hint", "")},
                        fresh, None
                    )
                    card["matched_card"] = card["candidates"][0] if card["candidates"] else None
                    card["confirmed"] = False
                    card["status"] = "Needs review" if card["candidates"] else "No candidates"
                    st.rerun()
            with action_col:
                st.write("")
                st.caption(card.get("status", "Needs review"))
                if st.button("Exclude", key=f"exclude_{card['id']}"):
                    st.session_state.batch = [c for c in st.session_state.batch if c["id"] != card["id"]]
                    st.rerun()
else:
    st.info("Upload card front scans in the sidebar to start a batch.")

if any(c.get("matched_card") for c in st.session_state.batch):
    st.divider()
    export_mode = st.radio("CSV export rows", ["Confirmed matches only", "All selected matches"], horizontal=True)
    export_cards = [
        c for c in st.session_state.batch
        if c.get("matched_card") and (export_mode == "All selected matches" or c.get("confirmed"))
    ]
    rows = []
    for c in export_cards:
        m = c["matched_card"]
        rows.append({
            "Product Line": "Pokemon",
            "Product Name": m.get("name", ""),
            "Set Name": m.get("set_name", ""),
            "Number": m.get("number", ""),
            "Condition": "Near Mint",
            "Printing": "Normal",
            "Price": f"{float(m.get('market_price') or 0):.2f}",
            "Quantity": 1,
            "Identification Confidence": f"{m.get('confidence', 0):.3f}",
            "Confirmed": bool(c.get("confirmed")),
            "Source Filename": c["filename"],
        })
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download TCGplayer-ready CSV",
        data=csv_bytes,
        file_name="tcgcomping_batch.csv",
        mime="text/csv",
        use_container_width=True,
    )

if cards:
    if st.button("Clear entire queue", type="secondary"):
        st.session_state.batch = []
        st.rerun()
