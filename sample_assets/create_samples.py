"""
Script to create sample visual templates (Spades, Hearts, Diamonds, Clubs, Mercedes-style logo, etc.)
and a multi-page test PDF representing a card game / brand catalog.
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pymupdf as fitz

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "templates")
PDF_DIR = os.path.dirname(__file__)

def ensure_dirs():
    os.makedirs(SAMPLE_DIR, exist_ok=True)

def create_spade_icon(size=120):
    """Draw a clean, crisp spade symbol with transparent background."""
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    w, h = size, size
    cx = w // 2
    
    # Lobes of the spade
    r = int(w * 0.22)
    cy_lobes = int(h * 0.48)
    
    draw.ellipse([cx - 2 * r + int(w * 0.04), cy_lobes - r, cx + int(w * 0.04), cy_lobes + r], fill=(15, 23, 42, 255))
    draw.ellipse([cx - int(w * 0.04), cy_lobes - r, cx + 2 * r - int(w * 0.04), cy_lobes + r], fill=(15, 23, 42, 255))
    
    # Top point triangle
    tip = (cx, int(h * 0.12))
    left_pt = (int(w * 0.12), cy_lobes)
    right_pt = (int(w * 0.88), cy_lobes)
    draw.polygon([tip, left_pt, right_pt], fill=(15, 23, 42, 255))
    
    # Base/stem of the spade
    stem_top = (cx, int(h * 0.45))
    stem_bl = (int(w * 0.30), int(h * 0.88))
    stem_br = (int(w * 0.70), int(h * 0.88))
    draw.polygon([stem_top, stem_bl, stem_br], fill=(15, 23, 42, 255))
    
    path = os.path.join(SAMPLE_DIR, "spade.png")
    img.save(path, "PNG")
    return path

def create_heart_icon(size=120):
    """Draw a clean heart symbol."""
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    w, h = size, size
    cx = w // 2
    r = int(w * 0.23)
    cy_lobes = int(h * 0.35)
    
    draw.ellipse([cx - 2 * r + int(w*0.04), cy_lobes - r, cx + int(w*0.04), cy_lobes + r], fill=(220, 38, 38, 255))
    draw.ellipse([cx - int(w*0.04), cy_lobes - r, cx + 2 * r - int(w*0.04), cy_lobes + r], fill=(220, 38, 38, 255))
    
    tip = (cx, int(h * 0.88))
    left_pt = (int(w * 0.10), cy_lobes + int(h * 0.05))
    right_pt = (int(w * 0.90), cy_lobes + int(h * 0.05))
    draw.polygon([tip, left_pt, right_pt], fill=(220, 38, 38, 255))
    
    path = os.path.join(SAMPLE_DIR, "heart.png")
    img.save(path, "PNG")
    return path

def create_diamond_icon(size=120):
    """Draw a clean diamond symbol."""
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    w, h = size, size
    pts = [
        (w // 2, int(h * 0.10)),
        (int(w * 0.88), h // 2),
        (w // 2, int(h * 0.90)),
        (int(w * 0.12), h // 2)
    ]
    draw.polygon(pts, fill=(220, 38, 38, 255))
    path = os.path.join(SAMPLE_DIR, "diamond.png")
    img.save(path, "PNG")
    return path

def create_club_icon(size=120):
    """Draw a clean club symbol."""
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    w, h = size, size
    cx = w // 2
    r = int(w * 0.20)
    
    draw.ellipse([cx - r, int(h * 0.12), cx + r, int(h * 0.12) + 2 * r], fill=(15, 23, 42, 255))
    draw.ellipse([int(w * 0.12), int(h * 0.38), int(w * 0.12) + 2 * r, int(h * 0.38) + 2 * r], fill=(15, 23, 42, 255))
    draw.ellipse([int(w * 0.88) - 2 * r, int(h * 0.38), int(w * 0.88), int(h * 0.38) + 2 * r], fill=(15, 23, 42, 255))
    draw.rectangle([int(w * 0.35), int(h * 0.35), int(w * 0.65), int(h * 0.55)], fill=(15, 23, 42, 255))
    draw.polygon([(cx, int(h * 0.40)), (int(w * 0.30), int(h * 0.88)), (int(w * 0.70), int(h * 0.88))], fill=(15, 23, 42, 255))
    
    path = os.path.join(SAMPLE_DIR, "club.png")
    img.save(path, "PNG")
    return path

def create_star_icon(size=120):
    """Draw a clean star icon."""
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    w, h = size, size
    cx, cy = w / 2, h / 2
    points = []
    for i in range(10):
        r = (w * 0.45) if i % 2 == 0 else (w * 0.20)
        angle = i * np.pi / 5 - np.pi / 2
        x = cx + r * np.cos(angle)
        y = cy + r * np.sin(angle)
        points.append((x, y))
    draw.polygon(points, fill=(234, 179, 8, 255))
    path = os.path.join(SAMPLE_DIR, "star.png")
    img.save(path, "PNG")
    return path

def create_mercedes_logo(size=140):
    """Draw a clean Mercedes three-pointed star logo."""
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    w, h = size, size
    cx, cy = w / 2, h / 2
    outer_r = int(w * 0.44)
    line_w = max(3, int(w * 0.04))
    
    draw.ellipse([cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r], outline=(30, 41, 59, 255), width=line_w)
    
    angles = [-np.pi/2, np.pi/6, 5*np.pi/6]
    star_r = outer_r - line_w / 2
    center_hub_r = int(w * 0.08)
    
    for ang in angles:
        tip_x = cx + star_r * np.cos(ang)
        tip_y = cy + star_r * np.sin(ang)
        
        left_ang = ang + np.pi/2
        right_ang = ang - np.pi/2
        b1_x = cx + center_hub_r * np.cos(left_ang)
        b1_y = cy + center_hub_r * np.sin(left_ang)
        b2_x = cx + center_hub_r * np.cos(right_ang)
        b2_y = cy + center_hub_r * np.sin(right_ang)
        
        draw.polygon([(tip_x, tip_y), (b1_x, b1_y), (cx, cy)], fill=(30, 41, 59, 255))
        draw.polygon([(tip_x, tip_y), (b2_x, b2_y), (cx, cy)], fill=(71, 85, 105, 255))
    
    path = os.path.join(SAMPLE_DIR, "mercedes_logo.png")
    img.save(path, "PNG")
    return path

def generate_sample_card_game_pdf(output_path=None):
    """Generate a multi-page card game / catalog PDF with various suits, cards, and layouts."""
    if output_path is None:
        output_path = os.path.join(PDF_DIR, "sample_card_game_catalog.pdf")
    
    ensure_dirs()
    create_spade_icon()
    create_heart_icon()
    create_diamond_icon()
    create_club_icon()
    create_star_icon()
    create_mercedes_logo()
    
    doc = fitz.open()
    
    suits = {
        "spade": os.path.join(SAMPLE_DIR, "spade.png"),
        "heart": os.path.join(SAMPLE_DIR, "heart.png"),
        "diamond": os.path.join(SAMPLE_DIR, "diamond.png"),
        "club": os.path.join(SAMPLE_DIR, "club.png"),
        "star": os.path.join(SAMPLE_DIR, "star.png"),
        "mercedes": os.path.join(SAMPLE_DIR, "mercedes_logo.png")
    }
    
    cards_data = [
        {"title": "Royal Card Game - Deck Overview (Page 1)", 
         "cards": [
             {"name": "Ace of Spades", "suit": "spade", "rank": "A", "rect": (80, 160, 220, 360), "scale": 1.0},
             {"name": "King of Spades", "suit": "spade", "rank": "K", "rect": (240, 160, 380, 360), "scale": 0.85},
             {"name": "Queen of Hearts", "suit": "heart", "rank": "Q", "rect": (400, 160, 540, 360), "scale": 0.85},
         ]},
        {"title": "Suit Section: The Kingdom of Hearts (Page 2)",
         "cards": [
             {"name": "10 of Hearts", "suit": "heart", "rank": "10", "rect": (80, 160, 220, 360), "scale": 0.9},
             {"name": "Jack of Hearts", "suit": "heart", "rank": "J", "rect": (240, 160, 380, 360), "scale": 0.9},
             {"name": "Ace of Hearts", "suit": "heart", "rank": "A", "rect": (400, 160, 540, 360), "scale": 1.0},
         ]},
        {"title": "Mastering the Spades Trump (Page 3)",
         "cards": [
             {"name": "2 of Spades", "suit": "spade", "rank": "2", "rect": (80, 140, 200, 300), "scale": 0.75},
             {"name": "7 of Spades", "suit": "spade", "rank": "7", "rect": (220, 140, 340, 300), "scale": 0.75},
             {"name": "Jack of Spades", "suit": "spade", "rank": "J", "rect": (360, 140, 480, 300), "scale": 0.9},
             {"name": "Queen of Spades", "suit": "spade", "rank": "Q", "rect": (150, 360, 290, 540), "scale": 1.0},
             {"name": "10 of Diamonds", "suit": "diamond", "rank": "10", "rect": (330, 360, 470, 540), "scale": 0.85},
         ]},
        {"title": "Suit Section: Diamonds & Gold (Page 4)",
         "cards": [
             {"name": "Ace of Diamonds", "suit": "diamond", "rank": "A", "rect": (80, 180, 220, 380), "scale": 1.0},
             {"name": "King of Diamonds", "suit": "diamond", "rank": "K", "rect": (240, 180, 380, 380), "scale": 0.9},
             {"name": "Queen of Diamonds", "suit": "diamond", "rank": "Q", "rect": (400, 180, 540, 380), "scale": 0.9},
         ]},
        {"title": "The Dark Suits: Clubs & Spades Clash (Page 5)",
         "cards": [
             {"name": "Ace of Clubs", "suit": "club", "rank": "A", "rect": (80, 160, 220, 360), "scale": 0.95},
             {"name": "King of Clubs", "suit": "club", "rank": "K", "rect": (240, 160, 380, 360), "scale": 0.95},
             {"name": "10 of Spades", "suit": "spade", "rank": "10", "rect": (400, 160, 540, 360), "scale": 0.95},
         ]},
        {"title": "Official Sponsors & Automakers Catalog (Page 6)",
         "cards": [
             {"name": "Mercedes-Benz Luxury Fleet", "suit": "mercedes", "rank": "MB", "rect": (80, 180, 260, 400), "scale": 1.2},
             {"name": "Star Championship Award", "suit": "star", "rank": "STAR", "rect": (320, 180, 500, 400), "scale": 1.1},
         ]},
        {"title": "The All-Spade High Stakes Table (Page 7)",
         "cards": [
             {"name": "3 of Spades", "suit": "spade", "rank": "3", "rect": (60, 150, 180, 310), "scale": 0.75},
             {"name": "4 of Spades", "suit": "spade", "rank": "4", "rect": (200, 150, 320, 310), "scale": 0.75},
             {"name": "5 of Spades", "suit": "spade", "rank": "5", "rect": (340, 150, 460, 310), "scale": 0.75},
             {"name": "6 of Spades", "suit": "spade", "rank": "6", "rect": (480, 150, 600, 310), "scale": 0.75},
             {"name": "Ace of Spades (Gold Edition)", "suit": "spade", "rank": "A", "rect": (200, 350, 380, 580), "scale": 1.3},
         ]},
        {"title": "Dealer Hand Rotation - Round 4 (Page 8)",
         "cards": [
             {"name": "9 of Hearts", "suit": "heart", "rank": "9", "rect": (100, 180, 230, 370), "scale": 0.85},
             {"name": "8 of Clubs", "suit": "club", "rank": "8", "rect": (250, 180, 380, 370), "scale": 0.85},
             {"name": "8 of Diamonds", "suit": "diamond", "rank": "8", "rect": (400, 180, 530, 370), "scale": 0.85},
         ]}
    ]
    
    for p_idx, page_info in enumerate(cards_data):
        page = doc.new_page(width=612, height=792)
        
        page.draw_rect(fitz.Rect(0, 0, 612, 70), color=None, fill=(0.08, 0.12, 0.20))
        page.insert_text((30, 45), page_info["title"], fontsize=17, color=(1, 1, 1), fontname="helv")
        
        page.draw_line((30, 750), (582, 750), color=(0.7, 0.7, 0.7), width=0.8)
        page.insert_text((30, 770), f"Visual Card Game Catalog - Page {p_idx+1} of {len(cards_data)}", fontsize=9, color=(0.4, 0.4, 0.4))
        
        for card in page_info["cards"]:
            rx0, ry0, rx1, ry1 = card["rect"]
            card_rect = fitz.Rect(rx0, ry0, rx1, ry1)
            
            page.draw_rect(card_rect, color=(0.75, 0.75, 0.8), fill=(0.98, 0.98, 1.0), width=1.5)
            
            rank_color = (0.86, 0.15, 0.15) if card["suit"] in ["heart", "diamond"] else (0.06, 0.09, 0.16)
            page.insert_text((rx0 + 12, ry0 + 26), card["rank"], fontsize=16, color=rank_color, fontname="helv")
            
            page.insert_text((rx0 + 10, ry1 - 12), card["name"], fontsize=9, color=(0.2, 0.2, 0.3), fontname="helv")
            
            suit_img_path = suits.get(card["suit"])
            if suit_img_path and os.path.exists(suit_img_path):
                cx = (rx0 + rx1) / 2
                cy = (ry0 + ry1) / 2
                img_size = min(rx1 - rx0, ry1 - ry0) * 0.45 * card.get("scale", 1.0)
                img_rect = fitz.Rect(cx - img_size/2, cy - img_size/2, cx + img_size/2, cy + img_size/2)
                page.insert_image(img_rect, filename=suit_img_path)
                
                corner_size = 14
                corner_rect = fitz.Rect(rx0 + 12, ry0 + 32, rx0 + 12 + corner_size, ry0 + 32 + corner_size)
                page.insert_image(corner_rect, filename=suit_img_path)
    
    doc.save(output_path)
    doc.close()
    print(f"Generated sample card game PDF with {len(cards_data)} pages at: {output_path}")
    return output_path

if __name__ == "__main__":
    ensure_dirs()
    generate_sample_card_game_pdf()
