"""
Streamlit Application for Computer Vision PDF Visual Search & Element Matching.
Searches PDF documents visually for predetermined logos, symbols, card suits, icons, and emblems.
"""
import io
import os
import time
from typing import List, Dict, Any, Optional
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2

from src.visual_search_engine import (
    render_entire_pdf,
    prepare_template,
    match_template_multiscale,
    search_document_multi_target
)
from src.annotator import (
    draw_matches_on_image,
    create_annotated_pdf,
    create_match_comparison_card
)
from src.exporter import (
    results_to_dataframe,
    results_to_json,
    create_results_zip_bundle
)

# Page configuration
st.set_page_config(
    page_title="PDF Visual Element & Symbol Matcher",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 6px;
    }
    .match-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session states
if "rendered_pages" not in st.session_state:
    st.session_state.rendered_pages = []
if "pdf_source" not in st.session_state:
    st.session_state.pdf_source = None
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = "sample_card_game_catalog.pdf"
if "targets" not in st.session_state:
    st.session_state.targets = []
if "search_summary" not in st.session_state:
    st.session_state.search_summary = None
if "current_page_idx" not in st.session_state:
    st.session_state.current_page_idx = 0


# Preset templates dictionary
SAMPLE_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "sample_assets", "templates")
PRESET_TEMPLATES = {
    "Spade (Black)": {"file": "spade.png", "color": "#22c55e", "default_name": "Spade"},
    "Heart (Red)": {"file": "heart.png", "color": "#ef4444", "default_name": "Heart"},
    "Diamond (Red)": {"file": "diamond.png", "color": "#f59e0b", "default_name": "Diamond"},
    "Club (Black)": {"file": "club.png", "color": "#8b5cf6", "default_name": "Club"},
    "Star (Gold)": {"file": "star.png", "color": "#eab308", "default_name": "Star"},
    "Mercedes Logo": {"file": "mercedes_logo.png", "color": "#0ea5e9", "default_name": "Mercedes"},
}

# Helper to load preset
def get_preset_image(filename: str) -> Optional[Image.Image]:
    path = os.path.join(SAMPLE_TEMPLATES_DIR, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None


# Top App Header
st.markdown('<div class="main-header">Visual PDF Element & Symbol Matcher</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Search PDF documents visually for logos, playing card suits, stamps, icons, or predetermined visual elements using high-precision Computer Vision template matching.</div>', unsafe_allow_html=True)


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("1. PDF Document")
    
    doc_source_type = st.radio(
        "Select Document Source",
        ["Pre-built Sample PDF", "Upload Custom PDF"],
        index=0
    )
    
    dpi_choice = st.select_slider(
        "Rendering Resolution (DPI)",
        options=[100, 150, 200, 300],
        value=150,
        help="Higher DPI provides sharper rendering and finer sub-pixel matching but takes slightly longer to render."
    )
    
    pdf_to_load = None
    loaded_name = ""
    
    if doc_source_type == "Pre-built Sample PDF":
        sample_choice = st.selectbox(
            "Choose Sample Catalog / Game",
            ["Sample Card Game Catalog (8 Pages)", "Information Dashboards (12 Pages)"]
        )
        if "Card Game" in sample_choice:
            pdf_path = os.path.join(os.path.dirname(__file__), "sample_assets", "sample_card_game_catalog.pdf")
            loaded_name = "sample_card_game_catalog.pdf"
        else:
            pdf_path = os.path.join(os.path.dirname(__file__), "07_Information_Dashboards.pdf")
            loaded_name = "07_Information_Dashboards.pdf"
            
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_to_load = f.read()
    else:
        uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])
        if uploaded_pdf is not None:
            pdf_to_load = uploaded_pdf.read()
            loaded_name = uploaded_pdf.name
            
    # Load and cache PDF if changed
    if pdf_to_load is not None:
        if st.session_state.pdf_source != pdf_to_load or getattr(st.session_state, "cached_dpi", 0) != dpi_choice:
            with st.spinner("Rendering PDF pages..."):
                pages = render_entire_pdf(pdf_to_load, dpi=dpi_choice)
                st.session_state.rendered_pages = pages
                st.session_state.pdf_source = pdf_to_load
                st.session_state.pdf_filename = loaded_name
                st.session_state.cached_dpi = dpi_choice
                st.session_state.search_summary = None
                st.session_state.current_page_idx = 0
            st.success(f"Loaded {len(pages)} pages ({loaded_name})")

    st.markdown("---")
    st.header("2. Query Targets")
    
    target_tab1, target_tab2 = st.tabs(["Preset Library", "Upload Image"])
    
    with target_tab1:
        selected_preset = st.selectbox("Choose Preset Symbol", list(PRESET_TEMPLATES.keys()))
        preset_info = PRESET_TEMPLATES[selected_preset]
        preset_img = get_preset_image(preset_info["file"])
        
        col_p1, col_p2 = st.columns([1, 2])
        if preset_img:
            with col_p1:
                st.image(preset_img, caption="Preset", width=70)
            with col_p2:
                if st.button("Add Preset Target", key=f"add_{selected_preset}"):
                    t_id = preset_info["default_name"].lower()
                    if not any(t["id"] == t_id for t in st.session_state.targets):
                        st.session_state.targets.append({
                            "id": t_id,
                            "name": preset_info["default_name"],
                            "color": preset_info["color"],
                            "image": preset_img
                        })
                        st.success(f"Added {preset_info['default_name']}")
                        st.rerun()
                    else:
                        st.info("Target already in active list.")
                        
    with target_tab2:
        uploaded_target = st.file_uploader("Upload Query Logo / Icon", type=["png", "jpg", "jpeg", "webp"])
        custom_name = st.text_input("Target Name", value="Custom Target")
        custom_color = st.color_picker("Highlight Color", value="#3b82f6")
        
        if uploaded_target is not None:
            col_u1, col_u2 = st.columns([1, 2])
            cust_img = Image.open(uploaded_target)
            with col_u1:
                st.image(cust_img, caption="Upload", width=70)
            with col_u2:
                if st.button("Add Uploaded Target"):
                    t_id = f"custom_{len(st.session_state.targets)+1}"
                    st.session_state.targets.append({
                        "id": t_id,
                        "name": custom_name,
                        "color": custom_color,
                        "image": cust_img
                    })
                    st.success(f"Added {custom_name}")
                    st.rerun()

    # Active targets manager
    if st.session_state.targets:
        st.subheader(f"Active Targets ({len(st.session_state.targets)})")
        for idx, t in enumerate(st.session_state.targets):
            t_col1, t_col2, t_col3 = st.columns([1, 3, 1])
            with t_col1:
                st.image(t["image"], width=36)
            with t_col2:
                st.markdown(f"<span style='color:{t['color']}; font-weight:bold;'>■</span> {t['name']}", unsafe_allow_html=True)
            with t_col3:
                if st.button("Remove", key=f"del_t_{idx}"):
                    st.session_state.targets.pop(idx)
                    st.rerun()
                    
        if st.button("Clear All Targets"):
            st.session_state.targets = []
            st.session_state.search_summary = None
            st.rerun()
    else:
        st.info("No query targets added yet. Add a preset or upload an image above.")

    st.markdown("---")
    st.header("3. Vision Matching Parameters")
    
    confidence_thresh = st.slider(
        "Confidence Threshold (%)",
        min_value=50,
        max_value=99,
        value=78,
        step=1,
        help="Higher values enforce strict 1-to-1 matching. Lower values allow minor visual variations."
    ) / 100.0
    
    with st.expander("Advanced Vision Settings"):
        scale_col1, scale_col2 = st.columns(2)
        with scale_col1:
            min_scale = st.number_input("Min Scale (x)", min_value=0.1, max_value=1.5, value=0.4, step=0.1)
        with scale_col2:
            max_scale = st.number_input("Max Scale (x)", min_value=1.0, max_value=4.0, value=2.2, step=0.1)
            
        num_scales = st.slider("Scale Steps Count", min_value=10, max_value=40, value=20)
        
        rot_choice = st.checkbox("Enable Multi-Angle Rotation Search (0, 90, 180, 270 deg)", value=False)
        angles_list = [0.0, 90.0, 180.0, 270.0] if rot_choice else [0.0]
        
        match_mode = st.radio("Matching Mode", ["Normalized Cross-Correlation (Grayscale)", "Color-Preserving", "Edge / Canny Contours"], index=0)
        grayscale_mode = (match_mode == "Normalized Cross-Correlation (Grayscale)")
        edge_mode = (match_mode == "Edge / Canny Contours")
        
        nms_iou = st.slider("NMS Overlap Threshold (IoU)", min_value=0.1, max_value=0.8, value=0.3, step=0.05)

    st.markdown("---")
    
    # Run search button
    search_disabled = (len(st.session_state.rendered_pages) == 0 or len(st.session_state.targets) == 0)
    if st.button("Run Visual Search Across PDF", type="primary", use_container_width=True, disabled=search_disabled):
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        def progress_cb(pct, msg):
            progress_bar.progress(pct)
            status_text.text(msg)
            
        summary = search_document_multi_target(
            pages=st.session_state.rendered_pages,
            targets=st.session_state.targets,
            min_scale=min_scale,
            max_scale=max_scale,
            num_scales=num_scales,
            angles=angles_list,
            threshold=confidence_thresh,
            nms_iou_thresh=nms_iou,
            grayscale=grayscale_mode,
            edge_mode=edge_mode,
            progress_callback=progress_cb
        )
        
        st.session_state.search_summary = summary
        progress_bar.empty()
        status_text.empty()
        st.success(f"Search complete! Found {summary['total_matches']} matches in {summary['execution_time_sec']}s")
        st.rerun()


# ==========================================
# MAIN DASHBOARD CONTENT AREA
# ==========================================
if not st.session_state.rendered_pages:
    st.info("Please load or upload a PDF document from the sidebar to begin.")
    st.stop()

# Default preset setup if user opens app with 0 targets
if not st.session_state.targets:
    col_demo1, col_demo2 = st.columns([3, 1])
    with col_demo1:
        st.info("Quick Start: Click the button to the right to load the Spade template and search the sample catalog.")
    with col_demo2:
        if st.button("Demo: Search Spades"):
            spade_img = get_preset_image("spade.png")
            if spade_img:
                st.session_state.targets = [{
                    "id": "spade",
                    "name": "Spade",
                    "color": "#22c55e",
                    "image": spade_img
                }]
                st.rerun()

# Top Metrics Banner if search has been run
if st.session_state.search_summary is not None:
    summary = st.session_state.search_summary
    results = summary["results"]
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary['total_matches']}</div>
            <div class="metric-label">Total Matches Found</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        num_pages_hit = len(summary['pages_with_matches'])
        total_p = len(st.session_state.rendered_pages)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{num_pages_hit} / {total_p}</div>
            <div class="metric-label">Pages with Matches</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary['execution_time_sec']}s</div>
            <div class="metric-label">Search Duration</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        top_conf = max([r["confidence"] for r in results]) * 100 if results else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{top_conf:.1f}%</div>
            <div class="metric-label">Highest Match Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# Main Tabs Navigation
tab_inspector, tab_gallery, tab_analytics, tab_cropper, tab_export = st.tabs([
    "Visual Page Inspector",
    "Matched Instances Gallery",
    "Analytics & Distribution",
    "In-PDF Crop & Search",
    "Export Center"
])


# ==========================================
# TAB 1: VISUAL PAGE INSPECTOR
# ==========================================
with tab_inspector:
    pages = st.session_state.rendered_pages
    summary = st.session_state.search_summary
    results = summary["results"] if summary else []
    
    # Page Navigation Controls
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([2, 2, 2, 2])
    
    with nav_col1:
        only_matches = st.checkbox("Show only pages with matches", value=bool(summary and summary["pages_with_matches"]))
        
    available_indices = []
    if only_matches and summary and summary["pages_with_matches"]:
        available_indices = [p_num - 1 for p_num in summary["pages_with_matches"]]
    else:
        available_indices = list(range(len(pages)))
        
    if not available_indices:
        st.warning("No pages match the current filter.")
        available_indices = list(range(len(pages)))
        
    with nav_col2:
        page_options = [f"Page {idx+1}" for idx in available_indices]
        current_sel_idx = 0
        if st.session_state.current_page_idx in available_indices:
            current_sel_idx = available_indices.index(st.session_state.current_page_idx)
            
        selected_page_str = st.selectbox("Select Page", page_options, index=current_sel_idx)
        selected_page_idx = available_indices[page_options.index(selected_page_str)]
        st.session_state.current_page_idx = selected_page_idx

    with nav_col3:
        btn_prev, btn_next = st.columns(2)
        with btn_prev:
            if st.button("Previous Page", use_container_width=True):
                cur_pos = available_indices.index(selected_page_idx)
                if cur_pos > 0:
                    st.session_state.current_page_idx = available_indices[cur_pos - 1]
                    st.rerun()
        with btn_next:
            if st.button("Next Page", use_container_width=True):
                cur_pos = available_indices.index(selected_page_idx)
                if cur_pos < len(available_indices) - 1:
                    st.session_state.current_page_idx = available_indices[cur_pos + 1]
                    st.rerun()

    with nav_col4:
        show_labels_opt = st.checkbox("Show Labels & Scores", value=True)
        show_conf_opt = st.checkbox("Show Confidence %", value=True)
        box_thickness_opt = st.slider("Box Thickness", min_value=1, max_value=6, value=3)

    # Render current page
    curr_page_data = pages[selected_page_idx]
    page_img = curr_page_data["image"]
    page_num = curr_page_data["page_num"]
    
    # Filter matches on this page
    matches_on_page = [r for r in results if r["page_idx"] == selected_page_idx]
    
    if matches_on_page:
        annotated_img = draw_matches_on_image(
            page_rgb=page_img,
            matches_for_page=matches_on_page,
            show_labels=show_labels_opt,
            show_confidence=show_conf_opt,
            box_thickness=box_thickness_opt
        )
    else:
        annotated_img = page_img

    # Main view: Left column = Rendered page, Right column = Matches panel
    page_col, detail_col = st.columns([7, 3])
    
    with page_col:
        st.markdown(f"**Viewing: Page {page_num} of {len(pages)}** ({len(matches_on_page)} match{'es' if len(matches_on_page) != 1 else ''} found)")
        st.image(annotated_img, use_container_width=True)

    with detail_col:
        st.markdown("### Detected on this Page")
        if matches_on_page:
            for idx, m in enumerate(matches_on_page):
                with st.expander(f"Match #{idx+1}: {m['target_name']} ({m['confidence']*100:.1f}%)", expanded=(idx < 3)):
                    c_col1, c_col2 = st.columns([1, 1])
                    with c_col1:
                        st.caption("Detected Region Crop")
                        if m.get("crop") is not None:
                            st.image(m["crop"], use_container_width=True)
                    with c_col2:
                        st.markdown(f"**Target:** {m['target_name']}")
                        st.markdown(f"**Confidence:** `{m['confidence']*100:.2f}%`")
                        st.markdown(f"**Scale:** `{m['scale']}x`")
                        st.markdown(f"**Angle:** `{m['angle']} deg`")
                        st.markdown(f"**Bounds (px):** `[{m['bbox'][0]}, {m['bbox'][1]}, {m['bbox'][2]}, {m['bbox'][3]}]`")
        else:
            st.info("No matches on this page.")
            if not summary:
                st.caption("Click 'Run Visual Search Across PDF' in the sidebar to search.")


# ==========================================
# TAB 2: MATCHED INSTANCES GALLERY
# ==========================================
with tab_gallery:
    summary = st.session_state.search_summary
    results = summary["results"] if summary else []
    
    if not results:
        st.info("No matched instances to display. Run a visual search from the sidebar to populate the gallery.")
    else:
        gal_f1, gal_f2, gal_f3 = st.columns([2, 2, 2])
        with gal_f1:
            target_filter = st.selectbox("Filter by Target", ["All Targets"] + list(set(r["target_name"] for r in results)))
        with gal_f2:
            min_score_filter = st.slider("Filter Minimum Confidence (%)", min_value=50, max_value=99, value=int(confidence_thresh * 100)) / 100.0
        with gal_f3:
            sort_by = st.selectbox("Sort By", ["Highest Confidence", "Page Number", "Target Name"])

        filtered_results = results.copy()
        if target_filter != "All Targets":
            filtered_results = [r for r in filtered_results if r["target_name"] == target_filter]
        filtered_results = [r for r in filtered_results if r["confidence"] >= min_score_filter]

        if sort_by == "Highest Confidence":
            filtered_results.sort(key=lambda x: x["confidence"], reverse=True)
        elif sort_by == "Page Number":
            filtered_results.sort(key=lambda x: (x["page_num"], -x["confidence"]))
        elif sort_by == "Target Name":
            filtered_results.sort(key=lambda x: (x["target_name"], -x["confidence"]))

        st.markdown(f"**Displaying {len(filtered_results)} matched instances:**")
        
        # Grid layout with cards
        num_cols = 4
        for i in range(0, len(filtered_results), num_cols):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                if i + j < len(filtered_results):
                    m = filtered_results[i + j]
                    with cols[j]:
                        st.markdown(f"**Page {m['page_num']} - {m['target_name']}**")
                        if m.get("crop") is not None and m["crop"].size > 0:
                            st.image(m["crop"], use_container_width=True)
                        st.markdown(f"Score: **{m['confidence']*100:.1f}%** | Scale: `{m['scale']}x`")
                        if st.button(f"Jump to P.{m['page_num']}", key=f"jump_{m['match_id']}"):
                            st.session_state.current_page_idx = m["page_idx"]
                            st.rerun()


# ==========================================
# TAB 3: ANALYTICS & DISTRIBUTION
# ==========================================
with tab_analytics:
    summary = st.session_state.search_summary
    results = summary["results"] if summary else []
    
    if not results:
        st.info("Run a visual search to view distribution analytics.")
    else:
        df = results_to_dataframe(results)
        
        an_col1, an_col2 = st.columns(2)
        
        with an_col1:
            st.markdown("### Matches per Page")
            page_counts = df.groupby(["Page", "Target"]).size().unstack(fill_value=0)
            all_page_nums = [p["page_num"] for p in st.session_state.rendered_pages]
            page_counts = page_counts.reindex(all_page_nums, fill_value=0)
            st.bar_chart(page_counts)
            
        with an_col2:
            st.markdown("### Confidence Score Distribution")
            hist_values, hist_edges = np.histogram(df["Confidence (%)"], bins=10, range=(50, 100))
            hist_df = pd.DataFrame({
                "Confidence Range (%)": [f"{int(hist_edges[i])}-{int(hist_edges[i+1])}%" for i in range(len(hist_values))],
                "Match Count": hist_values
            }).set_index("Confidence Range (%)")
            st.bar_chart(hist_df)
            
        st.markdown("### Full Match Records Table")
        st.dataframe(df, use_container_width=True)


# ==========================================
# TAB 4: IN-PDF CROP & SEARCH
# ==========================================
with tab_cropper:
    st.markdown("### Crop Visual Element Directly from a PDF Page")
    st.markdown("Select a page, adjust the bounding box over any symbol, logo, or icon, preview the crop, and search for it across all pages.")
    
    crop_page_idx = st.selectbox(
        "Select Page to Crop From",
        range(len(st.session_state.rendered_pages)),
        format_func=lambda idx: f"Page {idx+1}",
        key="crop_page_select"
    )
    
    src_page_info = st.session_state.rendered_pages[crop_page_idx]
    src_img = src_page_info["image"]
    img_h, img_w = src_img.shape[:2]
    
    crop_c1, crop_c2 = st.columns([1, 1])
    
    with crop_c1:
        st.markdown("**1. Adjust Crop Coordinates (Pixels)**")
        x_min_slider = st.slider("Left X (px)", 0, img_w - 20, int(img_w * 0.15))
        y_min_slider = st.slider("Top Y (px)", 0, img_h - 20, int(img_h * 0.20))
        box_w_slider = st.slider("Width (px)", 10, img_w - x_min_slider, min(140, img_w - x_min_slider))
        box_h_slider = st.slider("Height (px)", 10, img_h - y_min_slider, min(140, img_h - y_min_slider))
        
        x2_crop = x_min_slider + box_w_slider
        y2_crop = y_min_slider + box_h_slider
        
        preview_page_img = src_img.copy()
        cv2.rectangle(preview_page_img, (x_min_slider, y_min_slider), (x2_crop, y2_crop), (0, 255, 0), 3)
        st.image(preview_page_img, caption="Page with Crop Selection Box", use_container_width=True)
        
    with crop_c2:
        st.markdown("**2. Cropped Element Preview**")
        cropped_sample = src_img[y_min_slider:y2_crop, x_min_slider:x2_crop]
        if cropped_sample.size > 0:
            pil_cropped = Image.fromarray(cropped_sample)
            st.image(pil_cropped, caption=f"Cropped Template ({box_w_slider}x{box_h_slider} px)", width=160)
            
            crop_target_name = st.text_input("Name for Cropped Element", value=f"Cropped_P{crop_page_idx+1}")
            crop_target_color = st.color_picker("Highlight Color", value="#ec4899", key="crop_color_picker")
            
            if st.button("Add Cropped Target & Search Document Now", type="primary"):
                t_id = f"crop_p{crop_page_idx+1}_{int(time.time())}"
                st.session_state.targets.append({
                    "id": t_id,
                    "name": crop_target_name,
                    "color": crop_target_color,
                    "image": pil_cropped
                })
                st.success(f"Added target '{crop_target_name}'. Ready to search.")
                st.rerun()


# ==========================================
# TAB 5: EXPORT CENTER
# ==========================================
with tab_export:
    summary = st.session_state.search_summary
    results = summary["results"] if summary else []
    
    if not results:
        st.info("Run a visual search to generate and download reports.")
    else:
        st.markdown("### Download Visual Search Results & Highlighted PDF")
        st.markdown("Export findings in standard formats:")
        
        ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)
        
        with ex_col1:
            st.markdown("#### Full ZIP Bundle")
            st.caption("Contains Annotated PDF, CSV/Excel reports, JSON metadata, and high-res cropped images.")
            zip_bytes = create_results_zip_bundle(
                st.session_state.pdf_source,
                summary,
                st.session_state.pdf_filename
            )
            st.download_button(
                label="Download ZIP Bundle",
                data=zip_bytes,
                file_name=f"visual_matches_{st.session_state.pdf_filename.replace('.pdf', '')}.zip",
                mime="application/zip",
                use_container_width=True
            )
            
        with ex_col2:
            st.markdown("#### Annotated PDF")
            st.caption("Original PDF with permanent vector bounding box highlights on every matched page.")
            annot_pdf = create_annotated_pdf(st.session_state.pdf_source, results)
            st.download_button(
                label="Download Annotated PDF",
                data=annot_pdf,
                file_name=f"annotated_{st.session_state.pdf_filename}",
                mime="application/pdf",
                use_container_width=True
            )
            
        with ex_col3:
            st.markdown("#### Excel / CSV Report")
            st.caption("Tabular match data with coordinates, confidence scores, and page numbers.")
            df = results_to_dataframe(results)
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV Report",
                data=csv_data,
                file_name="matches_report.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with ex_col4:
            st.markdown("#### JSON Metadata")
            st.caption("Machine-readable JSON schema with exact pixel and PDF point coordinates.")
            json_str = results_to_json(results, metadata={"execution_time_sec": summary.get("execution_time_sec", 0)})
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="matches_metadata.json",
                mime="application/json",
                use_container_width=True
            )
