"""
Visual Search & Template Matching Engine for PDFs using Computer Vision.
Supports Multi-scale Normalized Cross-Correlation, Alpha Masking, Rotation Invariance,
Color Verification, Feature Matching (SIFT/ORB), and Non-Maximum Suppression.
"""
import io
import os
import time
from typing import List, Dict, Tuple, Optional, Any
import cv2
import numpy as np
from PIL import Image
import pymupdf as fitz


def render_pdf_page(page: fitz.Page, dpi: int = 150) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Render a PyMuPDF page to an RGB numpy array at specified DPI.
    Returns: (rgb_image_array, (pdf_width_pts, pdf_height_pts))
    """
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    
    # Convert pixmap to numpy RGB array
    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
    page_rect = page.rect
    pdf_size = (page_rect.width, page_rect.height)
    return img_data, pdf_size


def render_entire_pdf(pdf_source: Any, dpi: int = 150) -> List[Dict[str, Any]]:
    """
    Render all pages of a PDF into a list of page dicts.
    pdf_source can be a filepath (str) or bytes.
    """
    if isinstance(pdf_source, (bytes, io.BytesIO)):
        if isinstance(pdf_source, io.BytesIO):
            pdf_bytes = pdf_source.getvalue()
        else:
            pdf_bytes = pdf_source
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    else:
        doc = fitz.open(pdf_source)
    
    pages = []
    for i, page in enumerate(doc):
        img_rgb, (pdf_w, pdf_h) = render_pdf_page(page, dpi=dpi)
        pages.append({
            "page_idx": i,
            "page_num": i + 1,
            "image": img_rgb,
            "width": img_rgb.shape[1],
            "height": img_rgb.shape[0],
            "pdf_width": pdf_w,
            "pdf_height": pdf_h,
            "dpi": dpi
        })
    doc.close()
    return pages


def prepare_template(template_input: Any) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
    """
    Prepare a query template image.
    Handles PIL Image, file path, numpy array, bytes.
    Returns: (bgr_template, alpha_mask, is_transparent)
    """
    if isinstance(template_input, str):
        pil_img = Image.open(template_input)
    elif isinstance(template_input, bytes):
        pil_img = Image.open(io.BytesIO(template_input))
    elif isinstance(template_input, Image.Image):
        pil_img = template_input
    elif isinstance(template_input, np.ndarray):
        if len(template_input.shape) == 2:
            return cv2.cvtColor(template_input, cv2.COLOR_GRAY2BGR), None, False
        elif template_input.shape[2] == 4:
            pil_img = Image.fromarray(template_input)
        else:
            return template_input, None, False
    else:
        raise ValueError(f"Unsupported template input type: {type(template_input)}")
    
    # Check for alpha channel
    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        pil_rgba = pil_img.convert("RGBA")
        np_rgba = np.array(pil_rgba)
        bgr = cv2.cvtColor(np_rgba, cv2.COLOR_RGBA2BGR)
        alpha = np_rgba[:, :, 3]
        
        # Binarize mask
        _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        is_transparent = np.any(alpha < 250)
        return bgr, mask if is_transparent else None, is_transparent
    else:
        pil_rgb = pil_img.convert("RGB")
        np_rgb = np.array(pil_rgb)
        bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
        return bgr, None, False


def rotate_image(image: np.ndarray, angle: float, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Rotate an image (and optional mask) by angle degrees without cropping."""
    if angle == 0:
        return image, mask
    
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    rot_mat[0, 2] += (new_w / 2) - cx
    rot_mat[1, 2] += (new_h / 2) - cy
    
    rotated_img = cv2.warpAffine(image, rot_mat, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    rotated_mask = None
    if mask is not None:
        rotated_mask = cv2.warpAffine(mask, rot_mat, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return rotated_img, rotated_mask


def non_max_suppression_fast(boxes: np.ndarray, scores: np.ndarray, overlap_thresh: float = 0.3) -> List[int]:
    """
    Fast vector-based Non-Maximum Suppression.
    boxes: numpy array of shape (N, 4) in format [x1, y1, x2, y2]
    scores: numpy array of shape (N,)
    overlap_thresh: IoU overlap threshold
    Returns: list of integer indices of kept bounding boxes
    """
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0].astype(float)
    y1 = boxes[:, 1].astype(float)
    x2 = boxes[:, 2].astype(float)
    y2 = boxes[:, 3].astype(float)
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        inds = np.where(iou <= overlap_thresh)[0]
        order = order[inds + 1]
        
    return keep


def match_template_multiscale(
    page_rgb: np.ndarray,
    template_bgr: np.ndarray,
    template_mask: Optional[np.ndarray] = None,
    min_scale: float = 0.3,
    max_scale: float = 2.5,
    num_scales: int = 25,
    angles: Optional[List[float]] = None,
    threshold: float = 0.80,
    nms_iou_thresh: float = 0.30,
    grayscale: bool = True,
    edge_mode: bool = False,
    max_results_per_page: int = 100
) -> List[Dict[str, Any]]:
    """
    Search for a visual template across multiple scales and angles on a PDF page image.
    Uses Normalized Cross-Correlation (`cv2.TM_CCOEFF_NORMED`) with Non-Max Suppression.
    """
    if angles is None:
        angles = [0.0]
    
    page_bgr = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2BGR)
    page_h, page_w = page_bgr.shape[:2]
    
    orig_th, orig_tw = template_bgr.shape[:2]
    if orig_th <= 0 or orig_tw <= 0:
        return []
    
    # Convert page image for matching
    if grayscale or edge_mode:
        page_gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
        if edge_mode:
            page_proc = cv2.Canny(page_gray, 50, 200)
        else:
            page_proc = page_gray
    else:
        page_proc = page_bgr
    
    scales = np.linspace(min_scale, max_scale, num_scales)
    all_boxes = []
    all_scores = []
    all_scales = []
    all_angles = []
    
    for angle in angles:
        rot_tpl, rot_mask = rotate_image(template_bgr, angle, template_mask)
        r_th, r_tw = rot_tpl.shape[:2]
        
        for scale in scales:
            sw = int(r_tw * scale)
            sh = int(r_th * scale)
            
            # Check boundaries: template must be smaller than page and >= 8px
            if sw >= page_w or sh >= page_h or sw < 8 or sh < 8:
                continue
            
            # Resize template
            scaled_tpl = cv2.resize(rot_tpl, (sw, sh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
            scaled_mask = None
            if rot_mask is not None:
                scaled_mask = cv2.resize(rot_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
            
            # Process template
            if grayscale or edge_mode:
                scaled_gray = cv2.cvtColor(scaled_tpl, cv2.COLOR_BGR2GRAY)
                if edge_mode:
                    tpl_proc = cv2.Canny(scaled_gray, 50, 200)
                else:
                    tpl_proc = scaled_gray
            else:
                tpl_proc = scaled_tpl
            
            # Perform Template Matching
            try:
                if scaled_mask is not None and not edge_mode and not grayscale:
                    # Masked template matching in OpenCV (supports TM_CCORR_NORMED or TM_SQDIFF)
                    res = cv2.matchTemplate(page_proc, tpl_proc, cv2.TM_CCORR_NORMED, mask=scaled_mask)
                else:
                    res = cv2.matchTemplate(page_proc, tpl_proc, cv2.TM_CCOEFF_NORMED)
            except Exception:
                # Fallback to unmasked TM_CCOEFF_NORMED
                res = cv2.matchTemplate(page_proc, tpl_proc, cv2.TM_CCOEFF_NORMED)
            
            # Find locations exceeding threshold
            loc = np.where(res >= threshold)
            for pt_y, pt_x in zip(loc[0], loc[1]):
                score = float(res[pt_y, pt_x])
                
                # Bounding box [x1, y1, x2, y2]
                x1 = int(pt_x)
                y1 = int(pt_y)
                x2 = int(pt_x + sw)
                y2 = int(pt_y + sh)
                
                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(score)
                all_scales.append(scale)
                all_angles.append(angle)
    
    if len(all_boxes) == 0:
        return []
    
    boxes_np = np.array(all_boxes)
    scores_np = np.array(all_scores)
    
    # Run Non-Maximum Suppression to eliminate duplicate overlapping boxes
    keep_indices = non_max_suppression_fast(boxes_np, scores_np, overlap_thresh=nms_iou_thresh)
    keep_indices = keep_indices[:max_results_per_page]
    
    matches = []
    for idx in keep_indices:
        b = boxes_np[idx]
        sc = float(scores_np[idx])
        scale_val = float(all_scales[idx])
        ang_val = float(all_angles[idx])
        
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        # Ensure inside page boundary
        x1 = max(0, min(x1, page_w - 1))
        y1 = max(0, min(y1, page_h - 1))
        x2 = max(x1 + 1, min(x2, page_w))
        y2 = max(y1 + 1, min(y2, page_h))
        
        # Crop matched image
        crop_rgb = page_rgb[y1:y2, x1:x2].copy()
        
        matches.append({
            "bbox": [x1, y1, x2, y2],
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "confidence": sc,
            "scale": round(scale_val, 3),
            "angle": ang_val,
            "crop": crop_rgb
        })
        
    # Sort matches by confidence descending
    matches.sort(key=lambda m: m["confidence"], reverse=True)
    return matches


def match_features_sift(
    page_rgb: np.ndarray,
    template_bgr: np.ndarray,
    min_match_count: int = 6,
    ratio_thresh: float = 0.75
) -> Optional[Dict[str, Any]]:
    """
    Feature-based matching using SIFT and RANSAC Homography.
    Computes perspective transformation and returns detected polygon & keypoints.
    """
    page_bgr = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2BGR)
    page_gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(tpl_gray, None)
    kp2, des2 = sift.detectAndCompute(page_gray, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None
    
    # FLANN matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except Exception:
        return None
    
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
            
    if len(good_matches) < min_match_count:
        return None
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None:
        return None
    
    h, w = tpl_gray.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    
    # Bounding rect
    xs = dst[:, 0, 0]
    ys = dst[:, 0, 1]
    x1, y1, x2, y2 = int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))
    
    # Boundary clamp
    ph, pw = page_gray.shape
    x1 = max(0, min(x1, pw - 1))
    y1 = max(0, min(y1, ph - 1))
    x2 = max(x1 + 1, min(x2, pw))
    y2 = max(y1 + 1, min(y2, ph))
    
    inliers = int(np.sum(mask)) if mask is not None else len(good_matches)
    confidence = min(1.0, inliers / (len(good_matches) + 1e-5))
    
    crop_rgb = page_rgb[y1:y2, x1:x2].copy() if (x2 > x1 and y2 > y1) else None
    
    return {
        "bbox": [x1, y1, x2, y2],
        "polygon": dst.reshape(-1, 2).tolist(),
        "confidence": confidence,
        "inliers": inliers,
        "good_matches": len(good_matches),
        "crop": crop_rgb
    }


def search_document_multi_target(
    pages: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    min_scale: float = 0.3,
    max_scale: float = 2.5,
    num_scales: int = 25,
    angles: Optional[List[float]] = None,
    threshold: float = 0.80,
    nms_iou_thresh: float = 0.30,
    grayscale: bool = True,
    edge_mode: bool = False,
    progress_callback: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Search all pages of a PDF document for one or more target visual templates.
    Returns:
      {
        "total_matches": int,
        "matches_by_target": {target_id: count},
        "pages_with_matches": [page_nums],
        "results": [
           {
             "page_idx": int,
             "page_num": int,
             "target_id": str,
             "target_name": str,
             "target_color": str,
             "bbox": [x1, y1, x2, y2],
             "bbox_norm": [x1_norm, y1_norm, x2_norm, y2_norm],
             "bbox_pdf_pts": [px1, py1, px2, py2],
             "confidence": float,
             "scale": float,
             "angle": float,
             "crop": np.ndarray
           }, ...
        ],
        "execution_time_sec": float
      }
    """
    start_time = time.time()
    all_results = []
    total_steps = len(pages) * len(targets)
    current_step = 0
    
    # Pre-process templates
    prepared_targets = []
    for t in targets:
        bgr, mask, is_trans = prepare_template(t["image"])
        prepared_targets.append({
            "id": t.get("id", t.get("name", "target")),
            "name": t.get("name", "Target"),
            "color": t.get("color", "#22c55e"),
            "bgr": bgr,
            "mask": mask,
            "is_transparent": is_trans
        })
    
    matches_by_target = {t["id"]: 0 for t in prepared_targets}
    pages_with_matches = set()
    
    for page_info in pages:
        p_idx = page_info["page_idx"]
        p_num = page_info["page_num"]
        p_img = page_info["image"]
        pw = page_info["width"]
        ph = page_info["height"]
        pdf_w = page_info["pdf_width"]
        pdf_h = page_info["pdf_height"]
        
        # Scale factors from pixel coordinates back to PDF points
        sx = pdf_w / pw
        sy = pdf_h / ph
        
        for t in prepared_targets:
            matches = match_template_multiscale(
                page_rgb=p_img,
                template_bgr=t["bgr"],
                template_mask=t["mask"],
                min_scale=min_scale,
                max_scale=max_scale,
                num_scales=num_scales,
                angles=angles,
                threshold=threshold,
                nms_iou_thresh=nms_iou_thresh,
                grayscale=grayscale,
                edge_mode=edge_mode
            )
            
            for m in matches:
                x1, y1, x2, y2 = m["bbox"]
                pdf_pts = [round(x1 * sx, 2), round(y1 * sy, 2), round(x2 * sx, 2), round(y2 * sy, 2)]
                norm_coords = [round(x1 / pw, 4), round(y1 / ph, 4), round(x2 / pw, 4), round(y2 / ph, 4)]
                
                res_obj = {
                    "match_id": f"P{p_num}_T{t['id']}_{len(all_results)+1}",
                    "page_idx": p_idx,
                    "page_num": p_num,
                    "target_id": t["id"],
                    "target_name": t["name"],
                    "target_color": t["color"],
                    "bbox": [x1, y1, x2, y2],
                    "bbox_norm": norm_coords,
                    "bbox_pdf_pts": pdf_pts,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "confidence": m["confidence"],
                    "scale": m["scale"],
                    "angle": m["angle"],
                    "crop": m["crop"]
                }
                all_results.append(res_obj)
                matches_by_target[t["id"]] += 1
                pages_with_matches.add(p_num)
            
            current_step += 1
            if progress_callback:
                progress_callback(current_step / max(1, total_steps), f"Searching Page {p_num}/{len(pages)} for {t['name']}...")
                
    exec_time = round(time.time() - start_time, 3)
    
    return {
        "total_matches": len(all_results),
        "matches_by_target": matches_by_target,
        "pages_with_matches": sorted(list(pages_with_matches)),
        "results": all_results,
        "execution_time_sec": exec_time
    }
