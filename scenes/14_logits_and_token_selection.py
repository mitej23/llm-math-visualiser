"""
Scene 14 — Logits & Token Selection
Run: manim -pql 14_logits_and_token_selection.py LogitsTokenSelectionScene
"""

from manim import *
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


def softmax(x):
    e = np.exp(np.array(x) - max(x))
    return (e / e.sum()).tolist()


class LogitsTokenSelectionScene(LLMScene):
    def construct(self):
        title = self.show_title("Logits & Token Selection", "Picking the Next Word")
        self.wait(0.5)
        self.fade_all(title)

        # ── 1. From hidden state to logits ─────────────────────────────────────
        pipeline_title = body_text("End of decode step: vector → logits → token",
                                   color=WHITE)
        pipeline_title.to_edge(UP, buff=0.6)
        self.play(Write(pipeline_title), run_time=0.7)

        stages = [
            ("Hidden state\n4096-dim vector",   BLUE_MED),
            ("Linear projection\n4096 → 32,000", PURPLE_MED),
            ("Logits\n32,000 raw scores",         ORANGE_MED),
            ("Softmax\n→ probabilities",           GREEN_MED),
            ("Sample\none token",                  YELLOW_MED),
        ]
        stage_boxes = VGroup()
        for lbl, col in stages:
            b = rounded_box(2.2, 0.85,
                            fill_color=col + "22", stroke_color=col,
                            label=lbl, label_color=col)
            stage_boxes.add(b)

        stage_boxes.arrange(RIGHT, buff=0.45)
        stage_boxes.scale_to_fit_width(13.5)
        stage_boxes.move_to(ORIGIN)

        stage_arrows = VGroup(*[
            Arrow(stage_boxes[i].get_right(), stage_boxes[i + 1].get_left(),
                  color=GREY_MED, buff=0.04, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.18)
            for i in range(len(stage_boxes) - 1)
        ])

        self.play(LaggedStart(*[FadeIn(b) for b in stage_boxes], lag_ratio=0.15),
                  run_time=1.5)
        self.play(LaggedStart(*[GrowArrow(a) for a in stage_arrows], lag_ratio=0.1),
                  run_time=0.8)
        self.wait(1.0)
        self.fade_all(pipeline_title, stage_boxes, stage_arrows)

        # ── 2. Probability bar chart ───────────────────────────────────────────
        chart_title = body_text('After "The cat sat on the ___":  probability distribution',
                                color=WHITE)
        chart_title.to_edge(UP, buff=0.6)
        self.play(Write(chart_title), run_time=0.7)

        raw_tokens = ["mat",  "floor", "chair", "roof", "sun",  "cloud"]
        raw_logits = [ 4.2,    3.1,     2.8,     0.5,   -1.2,   -2.5 ]
        probs      = softmax(raw_logits)
        bar_colors = [GREEN_MED, BLUE_MED, BLUE_MED, GREY_MED, GREY_MED, GREY_MED]

        bars = make_prob_bars(raw_tokens, probs,
                              max_height=3.0, bar_width=0.65,
                              colors=bar_colors)
        bars.move_to(ORIGIN + DOWN * 0.5)

        self.play(LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.1),
                  run_time=1.2)
        self.wait(0.8)

        # Greedy highlight
        greedy_rect = SurroundingRectangle(bars[0], color=YELLOW_MED,
                                           buff=0.08, stroke_width=2, corner_radius=0.1)
        greedy_lbl = label_text("Greedy: always pick the highest", color=YELLOW_MED)
        greedy_lbl.next_to(greedy_rect, UP, buff=0.25)

        self.play(Create(greedy_rect), FadeIn(greedy_lbl), run_time=0.6)
        self.wait(0.8)
        self.fade_all(chart_title, bars, greedy_rect, greedy_lbl)

        # ── 3. Greedy vs. sampling ─────────────────────────────────────────────
        comparison_title = body_text("Greedy vs. Sampling — pros and cons:", color=WHITE)
        comparison_title.to_edge(UP, buff=0.6)
        self.play(Write(comparison_title), run_time=0.6)

        rows_data = [
            ("Greedy",      YELLOW_MED, "Always top token",   "Consistent", "Repetitive"),
            ("Temperature", ORANGE_MED, "Reshape distribution","Creative",   "May be incoherent"),
            ("Top-k",       BLUE_MED,   "Sample from top k",  "Controlled", "Fixed cutoff"),
            ("Top-p",       GREEN_MED,  "Dynamic nucleus",    "Adaptive",   "Best for most tasks"),
        ]

        rows = VGroup()
        for name, col, how, pro, con in rows_data:
            n   = body_text(name, color=col)
            h   = label_text(how, color=WHITE)
            p   = label_text(f"✓ {pro}", color=GREEN_MED)
            c   = label_text(f"✗ {con}", color=RED_MED)
            row = VGroup(n, h, p, c)
            row.arrange(RIGHT, buff=0.45)
            rows.add(row)

        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        rows.move_to(ORIGIN + DOWN * 0.3)
        box = SurroundingRectangle(rows, color=GREY_MED, buff=0.3, corner_radius=0.12)

        self.play(Create(box), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.2),
                  run_time=1.3)
        self.wait(2)
