"""TCGdex API adapter. Future games should implement the same search_cards interface."""
import requests
from rapidfuzz import process, fuzz

BASE_URL = "https://api.tcgdex.net/v2/en"
SESSION = requests.Session()
_CACHE = {}

def _get_json(url, timeout=10):
    try:
        response = SESSION.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        return None
    return None

def search_cards(name, limit=30):
    """Retrieve candidates by full name, then a short prefix fallback if necessary."""
    query = (name or "").strip()
    if len(query) < 2:
        return []
    cache_key = query.casefold()
    if cache_key in _CACHE:
        return _CACHE[cache_key][:limit]

    variants = [query]
    if len(query) > 4:
        variants.append(query[:4])
    raw_items = []
    seen_ids = set()
    for term in variants:
        url = f"{BASE_URL}/cards?name={requests.utils.quote(term)}"
        result = _get_json(url)
        if isinstance(result, list):
            for item in result:
                card_id = item.get("id")
                if card_id and card_id not in seen_ids:
                    seen_ids.add(card_id)
                    raw_items.append(item)
        if len(raw_items) >= limit * 2:
            break

    if not raw_items:
        _CACHE[cache_key] = []
        return []

    # Filter the retrieval pool by name similarity before requesting detailed records.
    names = {item["id"]: item.get("name", "") for item in raw_items}
    ranked = process.extract(query, names, scorer=fuzz.WRatio, limit=max(limit * 2, 20))
    cards = []
    for candidate_name, score, card_id in ranked:
        detail = _get_json(f"{BASE_URL}/cards/{card_id}")
        if not isinstance(detail, dict):
            continue
        set_data = detail.get("set") or {}
        pricing = ((detail.get("pricing") or {}).get("tcgplayer") or {})
        market_price = None
        # Prefer normal price, but don't silently treat missing price as a real $0 market.
        for variant in ("normal", "reverse", "holo"):
            variant_data = pricing.get(variant)
            if isinstance(variant_data, dict) and variant_data.get("marketPrice") is not None:
                try:
                    market_price = float(variant_data["marketPrice"])
                    break
                except (TypeError, ValueError):
                    pass
        image_base = detail.get("image")
        cards.append({
            "id": detail.get("id", card_id),
            "name": detail.get("name", candidate_name),
            "set_name": set_data.get("name", "Unknown Set"),
            "set_id": set_data.get("id", ""),
            "number": str(detail.get("localId", "")).strip(),
            "image_url": f"{image_base}/high.webp" if image_base else None,
            "market_price": market_price,
            "database_name_score": float(score) / 100,
            "game": "Pokemon",
        })
    _CACHE[cache_key] = cards
    return cards[:limit]
