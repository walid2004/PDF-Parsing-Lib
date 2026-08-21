"""
Exporter module for visual search results.
Generates CSV, Excel, JSON metadata, and comprehensive ZIP bundles containing cropped match images.
"""
import io
import json
import zipfile
from typing import List, Dict, Any, Optional
import pandas as pd
import cv2
from PIL import Image

from src.annotator import create_annotated_pdf


def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert search results into a clean Pandas DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "Match ID": r.get("match_id", ""),
            "Page": r.get("page_num", 0),
            "Target": r.get("target_name", ""),
            "Target ID": r.get("target_id", ""),
            "Confidence (%)": round(r.get("confidence", 0.0) * 100, 2),
            "Scale": r.get("scale", 1.0),
            "Angle (deg)": r.get("angle", 0.0),
            "Pixel X1": r.get("bbox", [0,0,0,0])[0],
            "Pixel Y1": r.get("bbox", [0,0,0,0])[1],
            "Pixel X2": r.get("bbox", [0,0,0,0])[2],
            "Pixel Y2": r.get("bbox", [0,0,0,0])[3],
            "Width (px)": r.get("width", 0),
            "Height (px)": r.get("height", 0),
            "PDF X1 (pt)": r.get("bbox_pdf_pts", [0,0,0,0])[0],
            "PDF Y1 (pt)": r.get("bbox_pdf_pts", [0,0,0,0])[1],
            "PDF X2 (pt)": r.get("bbox_pdf_pts", [0,0,0,0])[2],
            "PDF Y2 (pt)": r.get("bbox_pdf_pts", [0,0,0,0])[3],
        })
    return pd.DataFrame(rows)


def results_to_json(results: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> str:
    """Serialize search results to formatted JSON (excluding raw image arrays)."""
    clean_results = []
    for r in results:
        item = {k: v for k, v in r.items() if k != "crop"}
        clean_results.append(item)
        
    payload = {
        "metadata": metadata or {},
        "total_matches": len(clean_results),
        "matches": clean_results
    }
    return json.dumps(payload, indent=2)


def create_results_zip_bundle(
    pdf_source: Any,
    search_summary: Dict[str, Any],
    pdf_filename: str = "document.pdf"
) -> bytes:
    """
    Create a complete ZIP archive bundle containing:
    1. Annotated PDF with highlighted match rectangles
    2. matches_report.csv
    3. matches_report.xlsx
    4. matches_metadata.json
    5. cropped_matches/ directory with high-res PNGs of every match
    """
    results = search_summary.get("results", [])
    df = results_to_dataframe(results)
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Annotated PDF
        try:
            annotated_pdf_bytes = create_annotated_pdf(pdf_source, results)
            annot_name = f"annotated_{pdf_filename}" if not pdf_filename.startswith("annotated_") else pdf_filename
            zf.writestr(annot_name, annotated_pdf_bytes)
        except Exception as e:
            print(f"Warning: Could not create annotated PDF in zip: {e}")
            
        # 2. CSV Report
        csv_data = df.to_csv(index=False).encode("utf-8")
        zf.writestr("matches_report.csv", csv_data)
        
        # 3. Excel Report
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Matches")
        zf.writestr("matches_report.xlsx", excel_buf.getvalue())
        
        # 4. JSON metadata
        json_data = results_to_json(results, metadata={"execution_time_sec": search_summary.get("execution_time_sec", 0)}).encode("utf-8")
        zf.writestr("matches_metadata.json", json_data)
        
        # 5. Cropped images
        for r in results:
            crop_arr = r.get("crop")
            if crop_arr is not None and crop_arr.size > 0:
                match_id = r.get("match_id", "crop")
                target_name = r.get("target_name", "target").replace(" ", "_")
                p_num = r.get("page_num", 1)
                
                # Encode crop to PNG
                pil_crop = Image.fromarray(crop_arr)
                crop_buf = io.BytesIO()
                pil_crop.save(crop_buf, format="PNG")
                
                crop_filename = f"cropped_matches/page_{p_num:03d}_{target_name}_{match_id}.png"
                zf.writestr(crop_filename, crop_buf.getvalue())
                
    return zip_buffer.getvalue()
