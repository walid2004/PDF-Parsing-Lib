"""
Visual Annotator & PDF Highlighter for Visual Search Results.
Draws high-contrast bounding boxes, confidence badges, side-by-side comparisons,
and creates vector annotated downloadable PDFs.
"""
import io
import os
from typing import List, Dict, Tuple, Optional, Any
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pymupdf as fitz


def hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    """Convert hex color string like #22c55e to BGR tuple (B, G, R)."""
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 6:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return (b, g, r)
    return (0, 255, 0)


def hex_to_rgb_norm(hex_str: str) -> Tuple[float, float, float]:
    """Convert hex color string to normalized RGB tuple (0..1, 0..1, 0..1) for PyMuPDF."""
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 6:
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        return (r, g, b)
    return (0.1, 0.8, 0.3)


def draw_matches_on_image(
    page_rgb: np.ndarray,
    matches_for_page: List[Dict[str, Any]],
    show_labels: bool = True,
    show_confidence: bool = True,
    box_thickness: int = 3,
    alpha_fill: float = 0.15
) -> np.ndarray:
    """
    Draw visual annotations for all matches found on a page image.
    Includes semi-transparent highlighted fill, bounding border, and badge labels.
    """
    annotated = page_rgb.copy()
    overlay = annotated.copy()
    h, w = annotated.shape[:2]
    
    # 1. Draw semi-transparent fills
    for m in matches_for_page:
        x1, y1, x2, y2 = m["bbox"]
        color_hex = m.get("target_color", "#22c55e")
        bgr_color = hex_to_bgr(color_hex)
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        
        cv2.rectangle(overlay, (x1, y1), (x2, y2), rgb_color, -1)
    
    cv2.addWeighted(overlay, alpha_fill, annotated, 1.0 - alpha_fill, 0, annotated)
    
    # 2. Draw solid border rectangles and badges
    for idx, m in enumerate(matches_for_page):
        x1, y1, x2, y2 = m["bbox"]
        color_hex = m.get("target_color", "#22c55e")
        bgr_color = hex_to_bgr(color_hex)
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        
        # Border
        cv2.rectangle(annotated, (x1, y1), (x2, y2), rgb_color, box_thickness)
        
        # Corner brackets for modern CV UI look
        corner_len = min(15, (x2 - x1) // 3, (y2 - y1) // 3)
        if corner_len > 3:
            # Top-left
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), box_thickness + 1)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), box_thickness + 1)
            # Bottom-right
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), box_thickness + 1)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), box_thickness + 1)
            
        if show_labels:
            target_name = m.get("target_name", "Match")
            conf_val = m.get("confidence", 1.0) * 100
            scale_val = m.get("scale", 1.0)
            
            if show_confidence:
                label_text = f"#{idx+1} {target_name} ({conf_val:.1f}%)"
            else:
                label_text = f"#{idx+1} {target_name}"
                
            font_scale = 0.55
            font_thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            
            # Badge coordinates (draw above box if space, otherwise inside)
            bx1 = x1
            by1 = max(0, y1 - th - 8) if (y1 - th - 8 >= 0) else y1
            bx2 = min(w, bx1 + tw + 10)
            by2 = by1 + th + 8
            
            # Badge background
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), rgb_color, -1)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
            
            # Badge text (white or dark text depending on luminance)
            luminance = 0.299 * rgb_color[0] + 0.587 * rgb_color[1] + 0.114 * rgb_color[2]
            text_color = (15, 23, 42) if luminance > 160 else (255, 255, 255)
            
            cv2.putText(annotated, label_text, (bx1 + 5, by1 + th + 3), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            
    return annotated


def create_annotated_pdf(
    pdf_source: Any,
    search_results: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> bytes:
    """
    Generate an annotated vector PDF with visual highlight rectangles and labels
    permanently embedded for all matches.
    """
    if isinstance(pdf_source, (bytes, io.BytesIO)):
        if isinstance(pdf_source, io.BytesIO):
            pdf_bytes = pdf_source.getvalue()
        else:
            pdf_bytes = pdf_source
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    else:
        doc = fitz.open(pdf_source)
    
    # Group results by page_idx
    page_matches = {}
    for r in search_results:
        p_idx = r["page_idx"]
        if p_idx not in page_matches:
            page_matches[p_idx] = []
        page_matches[p_idx].append(r)
        
    for p_idx, matches in page_matches.items():
        if p_idx >= len(doc):
            continue
        page = doc[p_idx]
        
        for m in matches:
            px1, py1, px2, py2 = m["bbox_pdf_pts"]
            rect = fitz.Rect(px1, py1, px2, py2)
            
            color_rgb = hex_to_rgb_norm(m.get("target_color", "#22c55e"))
            
            # Draw vector highlight rectangle with stroke
            page.draw_rect(rect, color=color_rgb, width=2.0, overlay=True)
            
            # Semi-transparent highlight fill annotation
            annot = page.add_rect_annot(rect)
            annot.set_colors(stroke=color_rgb, fill=color_rgb)
            annot.set_opacity(0.2)
            annot.update()
            
            # Badge text
            target_name = m.get("target_name", "Match")
            conf_str = f"{m.get('confidence', 1.0)*100:.1f}%"
            badge_text = f"[{target_name}: {conf_str}]"
            
            # Insert badge text right above bounding box
            text_pos = fitz.Point(px1, max(12, py1 - 4))
            page.insert_text(text_pos, badge_text, fontsize=8, color=color_rgb, fontname="helv")
            
    if output_path:
        doc.save(output_path)
        doc.close()
        with open(output_path, "rb") as f:
            return f.read()
    else:
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes


def create_match_comparison_card(
    template_img: np.ndarray,
    crop_img: np.ndarray,
    target_name: str,
    confidence: float,
    scale: float,
    page_num: int,
    card_size: Tuple[int, int] = (300, 180)
) -> np.ndarray:
    """
    Create a crisp side-by-side visual comparison image between the query template
    and the detected match region with metadata metrics.
    """
    cw, ch = card_size
    card = np.ones((ch, cw, 3), dtype=np.uint8) * 248  # Light gray background
    
    # Border
    cv2.rectangle(card, (0, 0), (cw - 1, ch - 1), (220, 225, 230), 1)
    
    # Target and Match panels
    panel_w = 110
    panel_h = 110
    
    # Resize template to fit panel
    th, tw = template_img.shape[:2]
    aspect_t = tw / max(1, th)
    if aspect_t > 1.0:
        nw = panel_w
        nh = int(panel_w / aspect_t)
    else:
        nh = panel_h
        nw = int(panel_h * aspect_t)
    nw, nh = max(4, nw), max(4, nh)
    resized_tpl = cv2.resize(template_img, (nw, nh), interpolation=cv2.INTER_AREA)
    
    # Resize crop to fit panel
    ch_h, ch_w = crop_img.shape[:2]
    aspect_c = ch_w / max(1, ch_h)
    if aspect_c > 1.0:
        cnw = panel_w
        cnh = int(panel_w / aspect_c)
    else:
        cnh = panel_h
        cnw = int(panel_h * aspect_c)
    cnw, cnh = max(4, cnw), max(4, cnh)
    resized_crop = cv2.resize(crop_img, (cnw, cnh), interpolation=cv2.INTER_AREA)
    
    # Paste template at (20, 40)
    tx = 20 + (panel_w - nw) // 2
    ty = 40 + (panel_h - nh) // 2
    card[ty:ty+nh, tx:tx+nw] = resized_tpl[:, :, :3]
    
    # Paste crop at (170, 40)
    cx = 170 + (panel_w - cnw) // 2
    cy = 40 + (panel_h - cnh) // 2
    card[cy:cy+cnh, cx:cx+cnw] = resized_crop[:, :, :3]
    
    # Draw panel borders
    cv2.rectangle(card, (20, 40), (20 + panel_w, 40 + panel_h), (200, 205, 215), 1)
    cv2.rectangle(card, (170, 40), (170 + panel_w, 40 + panel_h), (34, 197, 94), 2)
    
    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(card, "Query Template", (20, 30), font, 0.45, (100, 116, 139), 1, cv2.LINE_AA)
    cv2.putText(card, f"Match on P.{page_num}", (170, 30), font, 0.45, (16, 185, 129), 1, cv2.LINE_AA)
    
    # Match metrics at bottom
    info_str = f"Score: {confidence*100:.1f}% | Scale: {scale:.2f}x"
    cv2.putText(card, info_str, (20, ch - 12), font, 0.45, (30, 41, 59), 1, cv2.LINE_AA)
    
    return card
