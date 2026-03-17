"""
Scene 05 — Embeddings
Run: manim -pql 05_embeddings.py EmbeddingsScene
"""

from manim import *
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class EmbeddingsScene(LLMScene):
    def construct(self):
        title = self.show_title("Embeddings", "How Numbers Carry Meaning")
        self.wait(0.5)

        # ── 1. Token ID → vector lookup ───────────────────────────────────────
        token_id_box = rounded_box(1.5, 0.65, fill_color=GREY_DARK,
                                   stroke_color=GREEN_MED,
                                   label='"king"  → 1547', label_color=GREEN_MED)
        token_id_box.shift(LEFT * 3.5 + UP * 0.5)

        arrow = Arrow(token_id_box.get_right(),
                      token_id_box.get_right() + RIGHT * 1.2,
                      color=WHITE, stroke_width=2, buff=0.1)

        lookup_label = label_text("Embedding\nLookup Table", color=GREY_LIGHT)
        lookup_label.next_to(arrow, UP, buff=0.1)

        vec_vals = [0.21, -0.48, 0.83, 0.07, -0.31, 0.60]
        vec_display = make_vector_display(vec_vals, color=BLUE_LIGHT)
        vec_display.next_to(arrow.get_end(), RIGHT, buff=0.2)
        vec_title = label_text("512-dim vector", color=BLUE_LIGHT)
        vec_title.next_to(vec_display, UP, buff=0.15)

        dots = label_text("⋮", color=BLUE_LIGHT)
        dots.next_to(vec_display, DOWN, buff=0.05)

        self.play(FadeIn(token_id_box), run_time=0.6)
        self.play(GrowArrow(arrow), FadeIn(lookup_label), run_time=0.6)
        self.play(FadeIn(vec_display), FadeIn(vec_title), FadeIn(dots), run_time=0.7)
        self.wait(0.8)

        # ── 2. 2D semantic space — word clusters ──────────────────────────────
        self.fade_all(token_id_box, arrow, lookup_label,
                      vec_display, vec_title, dots)

        axes = Axes(
            x_range=[-3.5, 3.5, 1], y_range=[-2.5, 2.5, 1],
            x_length=7, y_length=5,
            axis_config={"color": GREY_MED, "include_numbers": False},
            tips=False,
        )
        axes.shift(DOWN * 0.2)

        x_lbl = label_text("← meaning dimension 1 →", color=GREY_MED)
        x_lbl.next_to(axes, DOWN, buff=0.2)
        y_lbl = label_text("dim 2", color=GREY_MED)
        y_lbl.next_to(axes, LEFT, buff=0.2)

        # Word positions in 2D
        words = {
            "king":    (1.8,  1.5,  YELLOW_MED),
            "queen":   (1.5,  0.9,  YELLOW_MED),
            "prince":  (2.1,  0.3,  YELLOW_MED),
            "dog":     (-1.5, 0.8,  GREEN_MED),
            "cat":     (-1.8, 0.4,  GREEN_MED),
            "puppy":   (-1.2, -0.2, GREEN_MED),
            "Paris":   (-0.2, -1.5, ORANGE_MED),
            "London":  ( 0.6, -1.8, ORANGE_MED),
            "Berlin":  (-0.8, -2.0, ORANGE_MED),
        }

        self.play(Create(axes), FadeIn(x_lbl), FadeIn(y_lbl), run_time=0.8)

        dot_group = VGroup()
        for word, (x, y, col) in words.items():
            dot = Dot(axes.c2p(x, y), color=col, radius=0.10)
            lbl = label_text(word, color=col)
            lbl.next_to(dot, UR, buff=0.05)
            dot_group.add(VGroup(dot, lbl))

        self.play(LaggedStart(*[FadeIn(d) for d in dot_group], lag_ratio=0.1),
                  run_time=1.5)
        self.wait(0.8)

        # Draw cluster circles
        royal_ellipse = Ellipse(width=3.0, height=2.5, color=YELLOW_MED,
                                stroke_width=1.5, fill_opacity=0.06,
                                fill_color=YELLOW_MED)
        royal_ellipse.move_to(axes.c2p(1.8, 0.9))

        animal_ellipse = Ellipse(width=2.2, height=2.0, color=GREEN_MED,
                                 stroke_width=1.5, fill_opacity=0.06,
                                 fill_color=GREEN_MED)
        animal_ellipse.move_to(axes.c2p(-1.5, 0.3))

        city_ellipse = Ellipse(width=2.8, height=1.5, color=ORANGE_MED,
                               stroke_width=1.5, fill_opacity=0.06,
                               fill_color=ORANGE_MED)
        city_ellipse.move_to(axes.c2p(-0.1, -1.8))

        cluster_labels = VGroup(
            label_text("Royalty", color=YELLOW_MED).move_to(axes.c2p(2.4, 2.1)),
            label_text("Animals", color=GREEN_MED).move_to(axes.c2p(-2.4, 1.3)),
            label_text("Cities",  color=ORANGE_MED).move_to(axes.c2p(0.4, -2.3)),
        )

        self.play(Create(royal_ellipse), Create(animal_ellipse),
                  Create(city_ellipse), run_time=0.8)
        self.play(LaggedStart(*[Write(l) for l in cluster_labels], lag_ratio=0.2),
                  run_time=0.7)
        self.wait(1)

        # ── 3. The famous analogy: king - man + woman = queen ─────────────────
        analogy_txt = body_text(
            "king − man + woman  ≈  queen     (embedding arithmetic!)",
            color=WHITE,
        )
        analogy_txt.to_edge(DOWN, buff=0.4)
        self.play(Write(analogy_txt), run_time=1.0)

        # Draw an arrow king→queen
        king_pt  = axes.c2p(1.8, 1.5)
        queen_pt = axes.c2p(1.5, 0.9)
        analogy_arrow = Arrow(king_pt, queen_pt, color=WHITE,
                              buff=0.12, stroke_width=2)
        self.play(GrowArrow(analogy_arrow), run_time=0.5)
        self.wait(1.5)

        self.fade_all(analogy_arrow, analogy_txt, royal_ellipse,
                      animal_ellipse, city_ellipse, cluster_labels,
                      dot_group, axes, x_lbl, y_lbl, title)

        # ── 4. Summary ────────────────────────────────────────────────────────
        summary_lines = [
            "Token ID → dense vector (embedding)",
            "Similar meanings → nearby vectors",
            "Arithmetic on vectors = arithmetic on meaning",
            "Learned from context during training",
        ]
        summary = VGroup(*[body_text(f"• {l}", color=WHITE) for l in summary_lines])
        summary.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        summary.move_to(ORIGIN)
        box = SurroundingRectangle(summary, color=BLUE_MED,
                                   buff=0.35, corner_radius=0.15)
        self.play(Create(box), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(l) for l in summary], lag_ratio=0.15),
                  run_time=1.2)
        self.wait(2)
