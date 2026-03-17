"""
Scene 11 — Layer Normalization
Run: manim -pql 11_layer_normalization.py LayerNormScene
"""

from manim import *
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class LayerNormScene(LLMScene):
    def construct(self):
        title = self.show_title("Layer Normalization", "Keeping Values Stable")
        self.wait(0.5)
        self.fade_all(title)

        # ── 1. The problem: exploding / vanishing values ───────────────────────
        problem_title = body_text("Without normalisation — values explode or collapse:",
                                  color=WHITE)
        problem_title.to_edge(UP, buff=0.6)
        self.play(Write(problem_title), run_time=0.7)

        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[-10, 10000, 2000],
            x_length=6, y_length=4,
            axis_config={"color": GREY_MED, "include_numbers": False},
            tips=False,
        )
        axes.shift(DOWN * 0.2)
        x_lbl = label_text("Layer depth →", color=GREY_MED)
        x_lbl.next_to(axes, DOWN, buff=0.2)

        explode_curve = axes.plot(lambda x: np.exp(x * 0.8) - 1,
                                  color=RED_MED, stroke_width=2.5)
        vanish_curve  = axes.plot(lambda x: 9000 * np.exp(-x * 0.9),
                                  color=BLUE_MED, stroke_width=2.5)

        exp_lbl = label_text("Exploding activations", color=RED_MED)
        exp_lbl.next_to(axes, RIGHT, buff=0.3).shift(UP * 0.5)
        van_lbl = label_text("Vanishing activations", color=BLUE_MED)
        van_lbl.next_to(axes, RIGHT, buff=0.3).shift(DOWN * 0.3)

        self.play(Create(axes), FadeIn(x_lbl), run_time=0.6)
        self.play(Create(explode_curve), FadeIn(exp_lbl), run_time=0.7)
        self.play(Create(vanish_curve), FadeIn(van_lbl), run_time=0.7)
        self.wait(1)
        self.fade_all(problem_title, axes, x_lbl,
                      explode_curve, vanish_curve, exp_lbl, van_lbl)

        # ── 2. What normalisation does — visual ───────────────────────────────
        norm_title = body_text("LayerNorm: rescale each token's vector to mean≈0, std≈1",
                               color=WHITE)
        norm_title.to_edge(UP, buff=0.6)
        self.play(Write(norm_title), run_time=0.7)

        # Before
        before_vals = [8.5, -12.3, 45.1, 3.2, -0.8, 22.4]
        after_vals  = [(v - np.mean(before_vals)) / np.std(before_vals)
                       for v in before_vals]

        before_bars = VGroup()
        after_bars  = VGroup()
        bar_w = 0.5
        for i, (bv, av) in enumerate(zip(before_vals, after_vals)):
            x = i * (bar_w + 0.15)
            # Before bar
            b_height = min(abs(bv) * 0.05, 2.5) * np.sign(bv)
            b_col = RED_MED if bv > 0 else BLUE_MED
            b_bar = Rectangle(width=bar_w, height=abs(b_height),
                               fill_color=b_col, fill_opacity=0.7,
                               stroke_color=b_col, stroke_width=1)
            b_bar.shift(RIGHT * x + UP * (b_height / 2 if bv > 0 else b_height / 2))
            before_bars.add(b_bar)
            # After bar
            a_height = av * 0.5
            a_col = GREEN_MED if av > 0 else ORANGE_MED
            a_bar = Rectangle(width=bar_w, height=abs(a_height),
                               fill_color=a_col, fill_opacity=0.7,
                               stroke_color=a_col, stroke_width=1)
            a_bar.shift(RIGHT * x + UP * (a_height / 2 if av > 0 else a_height / 2))
            after_bars.add(a_bar)

        before_bars.move_to(LEFT * 3)
        after_bars.move_to(RIGHT * 3)

        b_label = label_text("Before LayerNorm\n(wild range)", color=RED_MED)
        b_label.next_to(before_bars, DOWN, buff=0.3)
        a_label = label_text("After LayerNorm\n(mean≈0, std≈1)", color=GREEN_MED)
        a_label.next_to(after_bars, DOWN, buff=0.3)

        zero_line_b = DashedLine(before_bars.get_left() + LEFT * 0.2,
                                 before_bars.get_right() + RIGHT * 0.2,
                                 color=GREY_MED, stroke_width=1)
        zero_line_b.move_to(before_bars.get_center())
        zero_line_a = DashedLine(after_bars.get_left() + LEFT * 0.2,
                                 after_bars.get_right() + RIGHT * 0.2,
                                 color=GREY_MED, stroke_width=1)
        zero_line_a.move_to(after_bars.get_center())

        arrow_transform = Arrow(LEFT * 0.8, RIGHT * 0.8, color=WHITE,
                                stroke_width=2, buff=0.1)
        norm_txt = label_text("LayerNorm", color=YELLOW_MED)
        norm_txt.next_to(arrow_transform, UP, buff=0.1)

        self.play(FadeIn(before_bars), FadeIn(zero_line_b), FadeIn(b_label),
                  run_time=0.8)
        self.play(GrowArrow(arrow_transform), Write(norm_txt), run_time=0.5)
        self.play(FadeIn(after_bars), FadeIn(zero_line_a), FadeIn(a_label),
                  run_time=0.8)
        self.wait(1.2)
        self.fade_all(norm_title, before_bars, zero_line_b, b_label,
                      arrow_transform, norm_txt, after_bars, zero_line_a, a_label)

        # ── 3. Pre-norm vs post-norm and RMSNorm ──────────────────────────────
        comp_title = body_text("Modern models use Pre-Norm with RMSNorm:", color=WHITE)
        comp_title.to_edge(UP, buff=0.6)
        self.play(Write(comp_title), run_time=0.7)

        rows_data = [
            ("Post-Norm (original)", GREY_LIGHT,
             "Norm AFTER sublayer + residual",    "Original Transformer"),
            ("Pre-Norm (modern)",    YELLOW_MED,
             "Norm BEFORE sublayer",              "GPT, LLaMA, most LLMs"),
            ("RMSNorm",              GREEN_MED,
             "Simplified: no mean subtraction",   "LLaMA, Mistral, Gemma"),
        ]

        rows = VGroup()
        for name, col, desc, used in rows_data:
            n = body_text(name, color=col)
            d = label_text(desc, color=WHITE)
            u = label_text(f"→ {used}", color=GREY_LIGHT)
            row = VGroup(n, d, u)
            row.arrange(RIGHT, buff=0.45)
            rows.add(row)

        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        rows.move_to(ORIGIN + DOWN * 0.2)
        box = SurroundingRectangle(rows, color=BLUE_MED, buff=0.35, corner_radius=0.15)

        self.play(Create(box), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.2),
                  run_time=1.2)

        rmsnorm_note = label_text(
            "RMSNorm skips mean subtraction — ~20% faster, same quality.",
            color=GREY_LIGHT,
        )
        rmsnorm_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(rmsnorm_note), run_time=0.6)
        self.wait(2)
