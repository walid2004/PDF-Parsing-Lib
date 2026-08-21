"""
Automated unit tests for Computer Vision Visual Search Engine and Highlighters.
"""
import os
import unittest
import numpy as np
import cv2
from PIL import Image

from src.visual_search_engine import (
    render_entire_pdf,
    prepare_template,
    match_template_multiscale,
    search_document_multi_target
)
from src.annotator import (
    draw_matches_on_image,
    create_annotated_pdf
)
from src.exporter import (
    results_to_dataframe,
    create_results_zip_bundle
)

class TestVisualSearchEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_pdf_path = os.path.join(os.path.dirname(__file__), "..", "sample_assets", "sample_card_game_catalog.pdf")
        cls.templates_dir = os.path.join(os.path.dirname(__file__), "..", "sample_assets", "templates")
        
        # Ensure sample PDF exists
        if not os.path.exists(cls.sample_pdf_path):
            from sample_assets.create_samples import generate_sample_card_game_pdf
            generate_sample_card_game_pdf(cls.sample_pdf_path)

    def test_render_pdf(self):
        pages = render_entire_pdf(self.sample_pdf_path, dpi=120)
        self.assertGreater(len(pages), 0)
        self.assertEqual(pages[0]["page_num"], 1)
        self.assertEqual(len(pages[0]["image"].shape), 3)

    def test_spade_search(self):
        pages = render_entire_pdf(self.sample_pdf_path, dpi=120)
        spade_path = os.path.join(self.templates_dir, "spade.png")
        self.assertTrue(os.path.exists(spade_path))
        
        spade_img = Image.open(spade_path)
        targets = [{"id": "spade", "name": "Spade", "color": "#22c55e", "image": spade_img}]
        
        summary = search_document_multi_target(
            pages=pages,
            targets=targets,
            min_scale=0.4,
            max_scale=1.8,
            num_scales=20,
            threshold=0.75,
            nms_iou_thresh=0.3
        )
        
        # Spades appear on Page 1, Page 3, Page 5, Page 7
        self.assertGreater(summary["total_matches"], 5)
        self.assertIn(1, summary["pages_with_matches"])
        self.assertIn(3, summary["pages_with_matches"])
        self.assertIn(7, summary["pages_with_matches"])
        
        # Test annotator
        p1_matches = [m for m in summary["results"] if m["page_num"] == 1]
        p1_annotated = draw_matches_on_image(pages[0]["image"], p1_matches)
        self.assertEqual(p1_annotated.shape, pages[0]["image"].shape)
        
        # Test PDF annotation
        annot_pdf_bytes = create_annotated_pdf(self.sample_pdf_path, summary["results"])
        self.assertGreater(len(annot_pdf_bytes), 1000)
        
        # Test Export
        df = results_to_dataframe(summary["results"])
        self.assertEqual(len(df), summary["total_matches"])
        
        zip_bytes = create_results_zip_bundle(self.sample_pdf_path, summary, "sample_card_game.pdf")
        self.assertGreater(len(zip_bytes), 1000)

    def test_mercedes_logo_search(self):
        pages = render_entire_pdf(self.sample_pdf_path, dpi=120)
        mb_path = os.path.join(self.templates_dir, "mercedes_logo.png")
        self.assertTrue(os.path.exists(mb_path))
        
        mb_img = Image.open(mb_path)
        targets = [{"id": "mercedes", "name": "Mercedes", "color": "#3b82f6", "image": mb_img}]
        
        summary = search_document_multi_target(
            pages=pages,
            targets=targets,
            min_scale=0.5,
            max_scale=1.5,
            num_scales=15,
            threshold=0.75
        )
        
        # Mercedes appears on Page 6
        self.assertIn(6, summary["pages_with_matches"])
        mb_matches = [m for m in summary["results"] if m["page_num"] == 6]
        self.assertGreaterEqual(len(mb_matches), 1)

if __name__ == "__main__":
    unittest.main()
