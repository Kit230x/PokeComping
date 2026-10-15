# TCGComping redesign

A modular Streamlit Pokémon card identification prototype based on the original TCGComping app.

## What changed

- Separates image normalization, OCR, database access, candidate ranking, and UI.
- Tries multiple title and collector-number regions instead of assuming one era-specific location.
- Preserves image aspect ratio when boundary detection fails rather than stretching every input.
- Uses collector-number evidence as well as name similarity.
- Displays match evidence and a heuristic confidence score.
- Does not auto-select weak candidates; the user can manually choose and confirm a match.
- CSV export supports confirmed-only or all selected matches.
- Uses a database adapter boundary so other TCG APIs can be added later.

## Requirements

Python 3.10+ recommended. Tesseract OCR must be installed separately because `pytesseract` is only a Python wrapper.

### Install Tesseract

- macOS: `brew install tesseract`
- Debian/Ubuntu: `sudo apt-get install tesseract-ocr`
- Windows: install the Tesseract OCR application and ensure `tesseract.exe` is on PATH.

### Run

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
streamlit run app.py
```

## Important behavior

- Confidence is a ranking heuristic, not a statistically calibrated probability.
- Market prices can be missing or stale depending on API data availability.
- Confirm identity before relying on a price. The app intentionally supports human review.
- Current adapter uses TCGdex's English Pokémon API. Multi-TCG support needs a separate adapter per game and a game-aware schema.
- OCR does not currently infer set symbols reliably. `set_hint` is left as an extension point rather than inventing set evidence.
- Candidate-image similarity is not yet part of the score; it is a sensible next improvement after collecting a small benchmark set of your actual scans.

## Suggested next step

Create a test folder of representative scans from each era, record the correct card IDs, and measure top-1 accuracy, top-5 accuracy, OCR name accuracy, and collector-number accuracy before tuning weights or auto-selection thresholds.
