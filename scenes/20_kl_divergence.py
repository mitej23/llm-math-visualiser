"""
Scene 20 — KL Divergence
Run: manim -pql 20_kl_divergence.py KLDivergenceScene
"""

from manim import *
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class KLDivergenceScene(LLMScene):
    def construct(self):
        title = self.show_title("KL Divergence", "Don't Stray Too Far")
        self.wait(0.5)
        self.fade_all(title)

        # ── 1. Two distributions — similar vs different ────────────────────────
        dist_title = body_text("KL Divergence measures how different two distributions are:",
                               color=WHITE)
        dist_title.to_edge(UP, buff=0.6)
        self.play(Write(dist_title), run_time=0.7)

        tokens = ["cat", "dog", "fish", "bird"]
        ref_probs  = [0.45, 0.30, 0.15, 0.10]   # reference policy
        close_probs = [0.42, 0.32, 0.16, 0.10]  # small KL
        far_probs   = [0.10, 0.05, 0.70, 0.15]  # large KL

        bar_w = 0.6
        y_scale = 3.0

        def make_dist_bars(probs, col, x_offset):
            bars = VGroup()
            for i, (tok, p) in enumerate(zip(tokens, probs)):
                h = p * y_scale
                bar = Rectangle(width=bar_w, height=h,
                                 fill_color=col, fill_opacity=0.8,
                                 stroke_color=col, stroke_width=1)
                bar.shift(RIGHT * (i * (bar_w + 0.2) + x_offset) + UP * (h / 2 - 1.0))
                lbl = label_text(tok, color=col)
                lbl.next_to(bar, DOWN, buff=0.1)
                bars.add(VGroup(bar, lbl))
            return bars

        ref_bars   = make_dist_bars(ref_probs,   BLUE_MED, -5.5)
        close_bars = make_dist_bars(close_probs, GREEN_MED, -1.2)
        far_bars   = make_dist_bars(far_probs,   RED_MED,   3.0)

        ref_label   = label_text("Reference\n(frozen SFT)", color=BLUE_MED)
        close_label = label_text("RL policy\n(small KL ✅)", color=GREEN_MED)
        far_label   = label_text("RL policy\n(large KL ❌)", color=RED_MED)

        ref_label.next_to(ref_bars, UP, buff=0.2)
        close_label.next_to(close_bars, UP, buff=0.2)
        far_label.next_to(far_bars, UP, buff=0.2)

        self.play(LaggedStart(*[FadeIn(b) for b in ref_bars],   lag_ratio=0.1),
                  FadeIn(ref_label), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(b) for b in close_bars], lag_ratio=0.1),
                  FadeIn(close_label), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(b) for b in far_bars],   lag_ratio=0.1),
                  FadeIn(far_label), run_time=0.8)

        kl_note = label_text(
            "Small KL → model improved but still 'itself'     "
            "Large KL → model has changed too dramatically",
            color=GREY_LIGHT,
        )
        kl_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(kl_note), run_time=0.5)
        self.wait(1.2)
        self.fade_all(dist_title, ref_bars, ref_label,
                      close_bars, close_label, far_bars, far_label, kl_note)

        # ── 2. KL penalty in the reward formula ───────────────────────────────
        formula_title = body_text("The RLHF reward formula:", color=WHITE)
        formula_title.to_edge(UP, buff=0.6)
        self.play(Write(formula_title), run_time=0.6)

        parts = [
            ("Total reward", WHITE),
            ("  =  ", GREY_LIGHT),
            ("Reward model score", GREEN_MED),
            ("  −  λ ×  ", GREY_LIGHT),
            ("KL( RL policy ‖ Reference )", ORANGE_MED),
        ]
        formula_row = VGroup(*[body_text(t, color=c) for t, c in parts])
        formula_row.arrange(RIGHT, buff=0.05)
        formula_row.move_to(ORIGIN + UP * 0.5)

        reward_note = label_text("↑ be helpful",       color=GREEN_MED)
        kl_note2    = label_text("↑ don't change too much", color=ORANGE_MED)
        reward_note.next_to(formula_row[2], DOWN, buff=0.3)
        kl_note2.next_to(formula_row[4], DOWN, buff=0.3)

        self.play(LaggedStart(*[FadeIn(p) for p in formula_row], lag_ratio=0.1),
                  run_time=1.0)
        self.play(FadeIn(reward_note), FadeIn(kl_note2), run_time=0.6)
        self.wait(1.2)

        source_note = label_text(
            "Source: Lambert et al., HuggingFace RLHF Blog (2022)  &  "
            "Huang et al., N+ Implementation Details (2023)",
            color=GREY_MED,
        )
        source_note.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(source_note), run_time=0.5)
        self.wait(1)
        self.fade_all(formula_title, formula_row, reward_note, kl_note2, source_note)

        # ── 3. Adaptive KL control ────────────────────────────────────────────
        adapt_title = body_text("Adaptive KL: auto-tune the leash tightness", color=WHITE)
        adapt_title.to_edge(UP, buff=0.6)
        self.play(Write(adapt_title), run_time=0.7)

        cases = [
            ("KL > target",  RED_MED,   "Tighten leash  (increase λ) 🐕‍🦺"),
            ("KL ≈ target",  GREEN_MED, "Keep as-is  ✅"),
            ("KL < target",  BLUE_MED,  "Loosen leash  (decrease λ) 🐕"),
        ]
        case_rows = VGroup()
        for cond, col, action in cases:
            c = body_text(cond, color=col)
            a = label_text(action, color=WHITE)
            a.next_to(c, RIGHT, buff=0.5)
            case_rows.add(VGroup(c, a))

        case_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        case_rows.move_to(ORIGIN + DOWN * 0.2)
        box = SurroundingRectangle(case_rows, color=GREY_MED, buff=0.3, corner_radius=0.12)

        target_note = label_text("Target KL ≈ 6.0 (typical default)", color=GREY_LIGHT)
        target_note.next_to(box, UP, buff=0.25)

        self.play(Write(target_note), Create(box), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(r) for r in case_rows], lag_ratio=0.25),
                  run_time=1.0)
        self.wait(2)
