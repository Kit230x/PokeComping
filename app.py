import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="TCGComping - Phase 1", layout="wide")


# ---------------------------------------------------------
# API Helper Functions (Free Tier - Robust Querying)
# ---------------------------------------------------------
def search_pokemon_cards(query_term):
    """Searches Pokemontcg.io cleanly by name or collector number."""
    clean_term = query_term.strip()
    if not clean_term:
        return []

    # Clean query: if it contains spaces use quotes without trailing asterisks;
    # if it's a single word or partial, allow wildcard.
    if " " in clean_term:
        q_param = f'name:"{clean_term}"'
    else:
        q_param = f'name:{clean_term}*'

    url = f"https://api.pokemontcg.io/v2/cards?q={q_param}&pageSize=8"
    headers = {"User-Agent": "TCGCompingApp/1.0"}

    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json().get("data", [])
            matches = []
            for item in data:
                prices = item.get("tcgplayer", {}).get("prices", {})
                market_price = 0.0
                # Fallback through price keys
                for p_type in ["normal", "holofoil", "reverseHolofoil", "unlimitedHolofoil"]:
                    if p_type in prices and prices[p_type].get("market"):
                        market_price = float(prices[p_type]["market"])
                        break

                matches.append({
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "set_name": item.get("set", {}).get("name", "Unknown Set"),
                    "number": item.get("number"),
                    "image_url": item.get("images", {}).get("small"),
                    "market_price": market_price
                })
            return matches
        else:
            st.warning(f"API returned status code: {res.status_code}")
    except Exception as e:
        st.error(f"Network error contacting API: {e}")
    return []


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "batch" not in st.session_state:
    st.session_state.batch = []

# ---------------------------------------------------------
# Sidebar: Batch Import
# ---------------------------------------------------------
st.sidebar.title("📥 Batch Intake (NM Only)")
uploaded_files = st.sidebar.file_uploader(
    "Import Front-Facing Scans",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="uploader"
)

if st.sidebar.button("Process New Uploads", type="primary"):
    if uploaded_files:
        added_count = 0
        existing_filenames = {c["filename"] for c in st.session_state.batch}
        for file in uploaded_files:
            if file.name not in existing_filenames:
                img_bytes = file.read()
                st.session_state.batch.append({
                    "id": f"{file.name}_{len(st.session_state.batch)}",
                    "filename": file.name,
                    "local_bytes": img_bytes,
                    "search_query": "Pikachu",  # Sensible default to test immediately
                    "matched_card": None,
                    "suggestions": [],
                    "selected_index": 0
                })
                added_count += 1
        if added_count > 0:
            st.sidebar.success(f"Added {added_count} card(s) to queue!")
            st.rerun()
        else:
            st.sidebar.info("Selected files are already in the batch.")

# ---------------------------------------------------------
# Header & Total Valuation Toolbar
# ---------------------------------------------------------
st.title("🗂️ TCGComping Verification Queue")

total_cards = len(st.session_state.batch)
total_value = sum(
    card["matched_card"]["market_price"]
    for card in st.session_state.batch
    if card.get("matched_card") is not None
)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Total Cards in Batch", f"{total_cards}")
col_m2.metric("Total Market Value", f"${total_value:,.2f}")
if total_cards > 0:
    col_m3.metric("Average / Card", f"${(total_value / total_cards):,.2f}")

sort_option = col_m4.selectbox(
    "Sort Batch By:",
    ["Original Order", "Price: High to Low", "Price: Low to High"]
)

if sort_option == "Price: High to Low":
    st.session_state.batch.sort(
        key=lambda x: (x["matched_card"]["market_price"] if x["matched_card"] else 0.0),
        reverse=True
    )
elif sort_option == "Price: Low to High":
    st.session_state.batch.sort(
        key=lambda x: (x["matched_card"]["market_price"] if x["matched_card"] else 0.0)
    )

st.markdown("---")

# ---------------------------------------------------------
# Verification Queue
# ---------------------------------------------------------
if not st.session_state.batch:
    st.info("No cards in queue. Drop your front scans in the left sidebar and click 'Process New Uploads'.")
else:
    for idx, card in enumerate(st.session_state.batch):
        with st.container():
            c_scan, c_web, c_details, c_actions = st.columns([1.5, 1.5, 3.5, 1.2])

            with c_scan:
                st.caption(f"Local Scan:")
                st.image(Image.open(io.BytesIO(card["local_bytes"])), use_container_width=True)

            with c_web:
                st.caption("**Online Match:**")
                if card["matched_card"] and card["matched_card"].get("image_url"):
                    st.image(card["matched_card"]["image_url"], use_container_width=True)
                else:
                    st.write("*(No match verified yet)*")

            with c_details:
                # Text input with distinct key
                card["search_query"] = st.text_input(
                    "Card Name Search",
                    value=card["search_query"],
                    key=f"query_input_{card['id']}"
                )

                # Search Comps Button
                if st.button("🔍 Search Comps", key=f"search_btn_{card['id']}"):
                    with st.spinner("Comping card..."):
                        matches = search_pokemon_cards(card["search_query"])
                        if matches:
                            card["suggestions"] = matches
                            card["matched_card"] = matches[0]
                            card["selected_index"] = 0
                            st.rerun()
                        else:
                            st.error(f"No cards found matching '{card['search_query']}'. Try another spelling.")

                # Suggestions Selector
                if card["suggestions"]:
                    options = [
                        f"{m['name']} — {m['set_name']} (#{m['number']}) | ${m['market_price']:.2f}"
                        for m in card["suggestions"]
                    ]

                    selected_opt = st.selectbox(
                        "Change match based on suggestions:",
                        options=options,
                        index=card["selected_index"],
                        key=f"select_{card['id']}"
                    )

                    chosen_idx = options.index(selected_opt)
                    if chosen_idx != card["selected_index"]:
                        card["selected_index"] = chosen_idx
                        card["matched_card"] = card["suggestions"][chosen_idx]
                        st.rerun()

                    active = card["matched_card"]
                    st.markdown(f"### Market Price: :green[**${active['market_price']:.2f}**]")
                    st.write(f"**Set:** {active['set_name']} | **Number:** {active['number']}")
                    st.caption("Condition: **Near Mint (NM)**")

            with c_actions:
                st.write("###")
                if st.button("🗑️ Delete", key=f"del_{card['id']}", type="secondary"):
                    st.session_state.batch = [
                        c for c in st.session_state.batch if c["id"] != card["id"]
                    ]
                    st.rerun()

            st.divider()

# ---------------------------------------------------------
# Export Ready Preview
# ---------------------------------------------------------
if any(c.get("matched_card") for c in st.session_state.batch):
    st.subheader("Ready to Export")
    csv_rows = ["Product Line,Product Name,Set Name,Number,Condition,Printing,Price,Quantity"]
    for c in st.session_state.batch:
        if c.get("matched_card"):
            m = c["matched_card"]
            csv_rows.append(
                f'Pokemon,"{m["name"]}","{m["set_name"]}","{m["number"]}",Near Mint,Normal,{m["market_price"]:.2f},1')

    csv_data = "\n".join(csv_rows)
    st.download_button(
        label="📥 Download TCGplayer Ready CSV",
        data=csv_data,
        file_name="tcgplayer_batch_nm.csv",
        mime="text/csv"
    )