import io
import uuid
import pandas as pd
import streamlit as st
from PIL import Image

from tcgcomping.image_processor import decode_image, normalize_card
from tcgcomping.ocr_engine import extract_card_signals
from tcgcomping.card_matcher import rank_candidates
from tcgcomping.card_database import search_cards

st.set_page_config(page_title="TCGComping — Verification Queue", layout="wide")
st.title("🗂️ TCGComping — Verification Queue")
st.caption(
    "Modular Pokémon card identification with confidence-aware suggestions. "
    "Review uncertain matches before pricing or export."
)

if "batch" not in st.session_state:
    st.session_state.batch = []


def safe_float(value, default=0.0):
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


with st.sidebar:
    st.header("📥 Batch intake")
    uploads = st.file_uploader(
        "Upload front scans",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    st.subheader("Identification settings")
    game = st.selectbox(
        "Game",
        ["Pokémon"],
        help="The pipeline is structured so game-specific adapters can be added later.",
    )
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
                progress.progress((i + 1) / len(uploads))
                status.write(f"Processed {i + 1} of {len(uploads)} uploads")
                continue

            raw_bytes = uploaded.getvalue()
            try:
                image = decode_image(raw_bytes)
                aligned, alignment_note = normalize_card(image)
                signals = extract_card_signals(aligned)
                candidates = search_cards(signals.get("name") or "")
                ranked = rank_candidates(signals, candidates, aligned)

                best = ranked[0] if ranked else None
                number_score = (best or {}).get("evidence", {}).get("collector_number_score", 0) or 0
                number_present = bool(signals.get("number"))
                auto_ok = bool(
                    best
                    and best.get("confidence") is not None
                    and best["confidence"] >= threshold
                    and best.get("evidence", {}).get("name_score", 0) >= 0.72
                    and (not number_present or number_score >= 0.99)
                )
                chosen = best if (allow_auto and auto_ok) else None

                new_items.append({
                    "id": str(uuid.uuid4()),
                    "filename": uploaded.name,
                    "local_bytes": raw_bytes,
                    "ocr_name": signals.get("name") or "",
                    "ocr_number": signals.get("number") or "",
                    "ocr_set_hint": signals.get("set_hint") or "",
                    "ocr_debug": signals.get("debug") or "",
                    "alignment_note": alignment_note,
                    "candidates": ranked,
                    "matched_card": chosen,
                    "confirmed": False,
                    "status": (
                        "Auto-suggested" if chosen
                        else ("Needs review" if ranked else "No candidates")
                    ),
                })
            except Exception as exc:
                new_items.append({
                    "id": str(uuid.uuid4()),
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
                    "status": "Processing error",
                })

            progress.progress((i + 1) / len(uploads))
            status.write(f"Processed {i + 1} of {len(uploads)} uploads")

        st.session_state.batch.extend(new_items)
        st.rerun()

cards = st.session_state.batch
valued = [c for c in cards if c.get("matched_card")]
total_value = sum(safe_float((c["matched_card"] or {}).get("market_price")) for c in valued)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Cards in queue", len(cards))
m2.metric("Selected matches", len(valued))
m3.metric("Confirmed matches", sum(bool(c.get("confirmed")) for c in cards))
m4.metric("Selected market value", f"${total_value:,.2f}")

if cards:
    sort_mode = st.selectbox(
        "Sort queue",
        ["Scan order", "Confidence: low to high", "Price: high to low"],
    )

    # Work on a display list so we never mutate session state while sorting for the UI
    display_cards = list(cards)
    if sort_mode == "Confidence: low to high":
        display_cards.sort(
            key=lambda c: safe_float((c.get("matched_card") or {}).get("confidence"))
        )
    elif sort_mode == "Price: high to low":
        display_cards.sort(
            key=lambda c: safe_float((c.get("matched_card") or {}).get("market_price")),
            reverse=True,
        )

    for card in display_cards:
        with st.container(border=True):
            scan_col, match_col, info_col, action_col = st.columns([1.2, 1.25, 3.1, 0.8])

            with scan_col:
                st.caption(card["filename"])
                try:
                    st.image(
                        Image.open(io.BytesIO(card["local_bytes"])),
                        use_container_width=True,
                    )
                except Exception:
                    st.warning("Could not display scan")
                st.caption(card.get("alignment_note") or "")

            with match_col:
                st.markdown("**Suggested match**")
                match = card.get("matched_card")
                if match and match.get("image_url"):
                    try:
                        st.image(match["image_url"], use_container_width=True)
                    except Exception:
                        st.info("Could not load database image")
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

                candidates = card.get("candidates") or []
                if candidates:
                    labels = [
                        f"{c.get('name') or 'Unknown card'} — {c.get('set_name') or 'Unknown set'} "
                        f"(#{c.get('number') or '?'}) | {safe_float(c.get('confidence')):.0%} | "
                        f"${safe_float(c.get('market_price')):.2f}"
                        for c in candidates
                    ]

                    select_key = f"candidate_{card['id']}"
                    # Seed session state once so the selectbox stays consistent across reruns
                    if select_key not in st.session_state:
                        idx = next(
                            (
                                i
                                for i, c in enumerate(candidates)
                                if match and c.get("id") == match.get("id")
                            ),
                            0,
                        )
                        st.session_state[select_key] = labels[min(idx, len(labels) - 1)]

                    chosen_label = st.selectbox(
                        "Candidate review",
                        labels,
                        key=select_key,
                    )
                    selected_idx = labels.index(chosen_label)
                    selected = candidates[selected_idx]

                    if not match or selected.get("id") != match.get("id"):
                        card["matched_card"] = selected
                        card["confirmed"] = False
                        card["status"] = "Manually selected — not yet confirmed"
                        st.rerun()

                    match = card.get("matched_card")
                    if match:
                        price = match.get("market_price")
                        confidence = match.get("confidence")
                        price_text = (
                            f"${safe_float(price):.2f}" if price is not None else "Unavailable"
                        )
                        confidence_text = (
                            f"{safe_float(confidence):.0%}"
                            if confidence is not None
                            else "Unavailable"
                        )
                        st.markdown(f"**Market price:** {price_text}")
                        st.caption(
                            f"{match.get('set_name') or 'Unknown set'} • "
                            f"#{match.get('number') or '?'} • confidence {confidence_text}"
                        )
                        with st.expander("Why this candidate ranked here"):
                            st.json(match.get("evidence") or {})

                        if not card.get("confirmed"):
                            if st.button("✓ Confirm identification", key=f"confirm_{card['id']}"):
                                card["confirmed"] = True
                                card["status"] = "Confirmed"
                                st.rerun()
                        else:
                            st.success("Identification confirmed")
                else:
                    st.write("No candidates found. Try improving the scan or enter a name below.")

                manual_term = st.text_input(
                    "Search by name",
                    value=card.get("ocr_name") or "",
                    key=f"manual_{card['id']}",
                )
                if st.button("Retry search", key=f"retry_{card['id']}"):
                    fresh = search_cards(manual_term)
                    # Pass None only if your rank_candidates treats image as optional.
                    # If it does not, decode the local bytes again here.
                    try:
                        aligned_img = None
                        if card.get("local_bytes"):
                            aligned_img = decode_image(card["local_bytes"])
                            aligned_img, _ = normalize_card(aligned_img)
                    except Exception:
                        aligned_img = None

                    card["candidates"] = rank_candidates(
                        {
                            "name": manual_term,
                            "number": card.get("ocr_number") or "",
                            "set_hint": card.get("ocr_set_hint") or "",
                        },
                        fresh,
                        aligned_img,
                    )
                    card["matched_card"] = (
                        card["candidates"][0] if card["candidates"] else None
                    )
                    card["confirmed"] = False
                    card["status"] = (
                        "Needs review" if card["candidates"] else "No candidates"
                    )
                    # Reset the selectbox key so the new candidate list is shown cleanly
                    select_key = f"candidate_{card['id']}"
                    if select_key in st.session_state:
                        del st.session_state[select_key]
                    st.rerun()

            with action_col:
                st.write("")
                st.caption(card.get("status", "Needs review"))
                if st.button("Exclude", key=f"exclude_{card['id']}"):
                    st.session_state.batch = [
                        c for c in st.session_state.batch if c["id"] != card["id"]
                    ]
                    # Clean up related widget keys
                    for k in list(st.session_state.keys()):
                        if card["id"] in str(k):
                            del st.session_state[k]
                    st.rerun()
else:
    st.info("Upload card front scans in the sidebar to start a batch.")

if any(c.get("matched_card") for c in st.session_state.batch):
    st.divider()
    export_mode = st.radio(
        "CSV export rows",
        ["Confirmed matches only", "All selected matches"],
        horizontal=True,
    )
    export_cards = [
        c
        for c in st.session_state.batch
        if c.get("matched_card")
        and (export_mode == "All selected matches" or c.get("confirmed"))
    ]
    rows = []
    for c in export_cards:
        m = c["matched_card"]
        rows.append({
            "Product Line": "Pokemon",
            "Product Name": m.get("name") or "",
            "Set Name": m.get("set_name") or "",
            "Number": m.get("number") or "",
            "Condition": "Near Mint",
            "Printing": "Normal",
            "Price": f"{safe_float(m.get('market_price')):.2f}",
            "Quantity": 1,
            "Identification Confidence": f"{safe_float(m.get('confidence')):.3f}",
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
        # Optional: wipe candidate/manual keys so a fresh batch starts clean
        for k in list(st.session_state.keys()):
            if k.startswith(("candidate_", "manual_", "confirm_", "retry_", "exclude_")):
                del st.session_state[k]
        st.rerun()