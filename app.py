import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="TCGComping - Phase 1", layout="wide")


# ---------------------------------------------------------
# API Helper Functions (Free Tier)
# ---------------------------------------------------------
def search_pokemon_cards(query_term):
    """Searches Pokemontcg.io by name or collector number and returns top matches."""
    url = f"https://api.pokemontcg.io/v2/cards?q=name:\"{query_term}\"*&pageSize=6"
    headers = {"User-Agent": "TCGCompingApp/1.0"}
    try:
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            data = res.json().get("data", [])
            matches = []
            for item in data:
                # Extract TCGPlayer market price (default to normal or holofoil)
                prices = item.get("tcgplayer", {}).get("prices", {})
                market_price = 0.0
                for price_type in ["normal", "holofoil", "reverseHolofoil", "unlimitedHolofoil"]:
                    if price_type in prices and prices[price_type].get("market"):
                        market_price = float(prices[price_type]["market"])
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
    except Exception as e:
        st.error(f"API Error: {e}")
    return []


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "batch" not in st.session_state:
    st.session_state.batch = []

# ---------------------------------------------------------
# Sidebar: Batch Import & Settings
# ---------------------------------------------------------
st.sidebar.title("📥 Batch Intake (NM Only)")
uploaded_files = st.sidebar.file_uploader(
    "Import Front-Facing Scans",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if st.sidebar.button("Process New Uploads"):
    if uploaded_files:
        for file in uploaded_files:
            # Check if file already added to session to prevent duplicate uploads
            if not any(card["filename"] == file.name for card in st.session_state.batch):
                img_bytes = file.read()

                # Default search query fallback from filename or standard prompt
                # (In full pipeline, PyTesseract output would populate this query)
                clean_query = file.name.split(".")[0].replace("_", " ").strip()

                st.session_state.batch.append({
                    "id": file.name,
                    "filename": file.name,
                    "local_bytes": img_bytes,
                    "search_query": clean_query,
                    "matched_card": None,
                    "suggestions": [],
                    "selected_index": 0
                })
        st.sidebar.success(f"Added {len(uploaded_files)} images!")

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

# Metrics Banner
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Total Cards in Batch", f"{total_cards} cards")
col_m2.metric("Total Market Value", f"${total_value:,.2f}")
if total_cards > 0:
    col_m3.metric("Average / Card", f"${(total_value / total_cards):,.2f}")

# Sorting Controls
sort_option = col_m4.selectbox(
    "Sort Batch By:",
    ["Original Order", "Price: High to Low", "Price: Low to High"]
)

# Apply Sorting
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
# Verification Queue (Side-by-Side Skimming)
# ---------------------------------------------------------
if not st.session_state.batch:
    st.info("No cards in queue. Upload your front scans in the left sidebar to start comping.")
else:
    for idx, card in enumerate(st.session_state.batch):
        with st.container():
            c_scan, c_web, c_details, c_actions = st.columns([1.5, 1.5, 3.5, 1.5])

            # 1. Your Uploaded Scan
            with c_scan:
                st.caption(f"Local Scan:")
                st.image(Image.open(io.BytesIO(card["local_bytes"])), use_container_width=True)

            # 2. Online Matched Reference Image
            with c_web:
                st.caption("**Online Match:**")
                if card["matched_card"] and card["matched_card"].get("image_url"):
                    st.image(card["matched_card"]["image_url"], use_container_width=True)
                else:
                    st.markdown("*(No image matched yet)*")

            # 3. Card Metadata, Comps & Suggestion Selector
            with c_details:
                # Query Input for Comping
                new_query = st.text_input(
                    "Card Name / Number Search",
                    value=card["search_query"],
                    key=f"query_{card['id']}"
                )

                # Fetch Button (If not fetched yet or user changes query)
                if st.button("Search Comps", key=f"btn_{card['id']}") or not card["suggestions"]:
                    matches = search_pokemon_cards(new_query)
                    card["suggestions"] = matches
                    if matches:
                        card["matched_card"] = matches[0]
                        card["selected_index"] = 0
                    st.rerun()

                # If matches exist, display suggestions dropdown
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

                    # Update matched card if user picks a different suggestion
                    chosen_idx = options.index(selected_opt)
                    if chosen_idx != card["selected_index"]:
                        card["selected_index"] = chosen_idx
                        card["matched_card"] = card["suggestions"][chosen_idx]
                        st.rerun()

                    # Display Active Comp Details
                    active = card["matched_card"]
                    st.markdown(f"### Market Price: :green[**${active['market_price']:.2f}**]")
                    st.write(f"**Set:** {active['set_name']} | **Number:** {active['number']}")
                    st.caption("Condition: **Near Mint (NM)**")

            # 4. Actions: Delete / Exclude
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
if st.session_state.batch:
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