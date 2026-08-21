# PDF Visual Element & Symbol Matcher (Computer Vision)

A Computer Vision Streamlit application for **visual searching and exact template matching** within multi-page PDF documents. 

Designed specifically for finding predetermined visual elements (such as **playing card suits** [Spade, Heart, Diamond, Club], **car maker logos** like Mercedes/BMW, brand emblems, stamps, icons, or visual symbols) across large PDF catalogs, game rules, and technical manuals.

---

## Key Features

1. **Exact & Multi-Scale Template Matching**:
   - Normalized Cross-Correlation (`cv2.TM_CCOEFF_NORMED`) with multi-scale pyramid search (e.g. `0.3x` to `2.5x` scale factors).
   - Alpha-channel transparency support for PNG icons and logos.
   - Non-Maximum Suppression (NMS) vector deduplication to eliminate duplicate overlapping bounding boxes.
   - Optional multi-angle rotation search (0, 90, 180, 270 degrees).

2. **Multiple Query Input Methods**:
   - **Preset Symbol Library**: 1-click loading for Spades, Hearts, Diamonds, Clubs, Stars, and Mercedes logos.
   - **Upload Query Image**: Upload any custom PNG / JPG logo or icon.
   - **Interactive In-PDF Cropper**: Select any page of the PDF, drag a crop box around a symbol on that page, and search for it across all pages.

3. **Multi-Target Batch Search**:
   - Search for multiple symbols simultaneously (e.g. Spades in green, Hearts in red, Mercedes in blue) with distinct color-coded bounding boxes.

4. **Visual Result Explorer**:
   - **Visual Page Inspector**: High-resolution page viewer with color-coded bounding boxes, confidence badges, and crop details.
   - **Matched Instances Gallery**: Visual card grid of all detected instances across the document with direct "Jump to Page" buttons.
   - **Analytics & Distribution**: Page-by-page hit count bar charts and confidence score histograms.

5. **Full Export Suite**:
   - **Annotated PDF**: Download a vector PDF with permanent highlight bounding boxes on every matched page.
   - **Cropped Matches ZIP Archive**: High-resolution PNG crops of every matched element.
   - **Excel & CSV Reports**: Tabular match coordinates (pixels & PDF points) and similarity scores.
   - **JSON Metadata**: Machine-readable JSON annotations.

---

## Quick Start

### 1. Installation
```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Streamlit Application
```bash
.venv\Scripts\python -m streamlit run app.py
```
or run `main.py`:
```bash
.venv\Scripts\python main.py
```
Open your browser at `http://localhost:8501`.

---

## Project Structure

```
PDFLib/
├── app.py                         # Streamlit Interactive Dashboard
├── main.py                        # Entry point launcher
├── requirements.txt               # Pinned dependencies
├── src/
│   ├── visual_search_engine.py    # Multi-scale template matching, NMS, and multi-page search
│   ├── annotator.py               # Bounding box visualizer, badges, & vector PDF annotator
│   └── exporter.py                # CSV, Excel, JSON, and ZIP bundle exporters
├── sample_assets/
│   ├── create_samples.py          # Generator for sample templates & multi-page test PDF
│   ├── sample_card_game_catalog.pdf # 8-page test PDF with card games and brand logos
│   └── templates/                 # Preset PNG symbols (Spade, Heart, Diamond, Club, Mercedes)
└── tests/
    └── test_visual_matcher.py     # Automated unit test suite
```
