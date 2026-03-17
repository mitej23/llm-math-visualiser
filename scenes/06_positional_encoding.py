"""
Scene 06 — Positional Encoding
Run: manim -pql 06_positional_encoding.py PositionalEncodingScene
"""

from manim import *
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class PositionalEncodingScene(LLMScene):
    def construct(self):
        title = self.show_title("Positional Encoding", "Where in the Sentence?")
        self.wait(0.5)

        # ── 1. The problem: word order matters ────────────────────────────────
        sentence_a = body_text('"Dog bites man"', color=GREEN_MED)
        sentence_b = body_text('"Man bites dog"', color=RED_MED)
        sentence_a.shift(UP * 1.2 + LEFT * 2.5)
        sentence_b.shift(UP * 1.2 + RIGHT * 2.5)

        note = label_text(
            "Same words, same embeddings — but very different meanings!\n"
            "Without position info, the model can't tell them apart.",
            color=GREY_LIGHT,
        )
        note.next_to(sentence_a, DOWN, buff=0.5)
        note.shift(RIGHT * 2.5)

        self.play(Write(sentence_a), Write(sentence_b), run_time=0.8)
        self.play(FadeIn(note), run_time=0.7)
        self.wait(1.2)
        self.fade_all(sentence_a, sentence_b, note)

        # ── 2. Adding position embedding to word embedding ────────────────────
        words    = ["The", "cat", "sat"]
        pos_nums = [0,     1,     2   ]
        word_col = BLUE_LIGHT
        pos_col  = ORANGE_MED

        col_groups = VGroup()
        for i, (word, pos) in enumerate(zip(words, pos_nums)):
            w_box = rounded_box(1.3, 0.6, fill_color=BLUE_DARK,
                                stroke_color=word_col, label=word, label_color=word_col)
            plus   = body_text("+", color=WHITE)
            p_box  = rounded_box(1.3, 0.6, fill_color=GREY_DARK,
                                 stroke_color=pos_col,
                                 label=f"pos {pos}", label_color=pos_col)
            eq     = body_text("=", color=WHITE)
            r_box  = rounded_box(1.3, 0.6, fill_color=PURPLE_MED + "33",
                                 stroke_color=PURPLE_MED,
                                 label="vector", label_color=PURPLE_MED)
            row = VGroup(w_box, plus, p_box, eq, r_box)
            row.arrange(RIGHT, buff=0.2)
            col_groups.add(row)

        col_groups.arrange(DOWN, buff=0.35)
        col_groups.shift(DOWN * 0.2)

        formula = body_text("Final embedding  =  Word embedding  +  Position embedding",
                            color=WHITE)
        formula.next_to(col_groups, UP, buff=0.45)

        self.play(Write(formula), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(r) for r in col_groups], lag_ratio=0.2),
                  run_time=1.2)
        self.wait(1.2)

        # ── 3. Sinusoidal waves visual ────────────────────────────────────────
        self.fade_all(formula, col_groups)

        wave_title = body_text("Sinusoidal Encoding — each position gets a unique wave pattern",
                               color=WHITE)
        wave_title.to_edge(UP, buff=0.8)
        self.play(Write(wave_title), run_time=0.7)

        axes = Axes(
            x_range=[0, 30, 5], y_range=[-1.2, 1.2, 0.5],
            x_length=9, y_length=3.5,
            axis_config={"color": GREY_MED, "include_numbers": False},
            tips=False,
        )
        axes.shift(DOWN * 0.3)
        x_lbl = label_text("Position in sequence →", color=GREY_MED)
        x_lbl.next_to(axes, DOWN, buff=0.2)

        # Plot 3 different frequency dimensions
        frequencies = [
            (0.5,  BLUE_MED,   "dim 0 (low freq)"),
            (2.0,  GREEN_MED,  "dim 1 (mid freq)"),
            (6.0,  ORANGE_MED, "dim 2 (high freq)"),
        ]
        curves = VGroup()
        leg_items = VGroup()
        for freq, col, lbl in frequencies:
            curve = axes.plot(lambda x, f=freq: np.sin(f * x / 5),
                              color=col, stroke_width=2.5)
            curves.add(curve)
            leg_dot  = Dot(color=col, radius=0.08)
            leg_text = label_text(lbl, color=col)
            leg_text.next_to(leg_dot, RIGHT, buff=0.15)
            leg_items.add(VGroup(leg_dot, leg_text))

        leg_items.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        leg_items.to_edge(RIGHT, buff=0.3)

        self.play(Create(axes), FadeIn(x_lbl), run_time=0.6)
        self.play(LaggedStart(*[Create(c) for c in curves], lag_ratio=0.3),
                  LaggedStart(*[FadeIn(l) for l in leg_items], lag_ratio=0.3),
                  run_time=1.8)
        self.wait(1.2)

        # Highlight two positions
        pos5_line  = DashedLine(axes.c2p(5, -1.2), axes.c2p(5, 1.2),
                                color=YELLOW_MED, stroke_width=2)
        pos20_line = DashedLine(axes.c2p(20, -1.2), axes.c2p(20, 1.2),
                                color=RED_MED, stroke_width=2)
        pos5_lbl  = label_text("pos 5",  color=YELLOW_MED)
        pos20_lbl = label_text("pos 20", color=RED_MED)
        pos5_lbl.next_to(pos5_line, UP,  buff=0.1)
        pos20_lbl.next_to(pos20_line, UP, buff=0.1)

        self.play(Create(pos5_line), FadeIn(pos5_lbl),
                  Create(pos20_line), FadeIn(pos20_lbl), run_time=0.7)

        explanation = label_text("Each position has a unique 'fingerprint' of wave values",
                                 color=GREY_LIGHT)
        explanation.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(explanation), run_time=0.5)
        self.wait(1.5)

        # ── 4. RoPE callout ───────────────────────────────────────────────────
        self.fade_all(wave_title, axes, x_lbl, curves, leg_items,
                      pos5_line, pos5_lbl, pos20_line, pos20_lbl,
                      explanation, title)

        methods = [
            ("Sinusoidal", "Formula-based waves",      "Original Transformer"),
            ("Learned",    "Lookup table per position", "GPT-2, BERT"),
            ("RoPE",       "Rotate embedding vectors",  "LLaMA, Mistral"),
            ("ALiBi",      "Penalise by distance",      "Bloom, MPT"),
        ]

        rows = VGroup()
        for method, how, used_in in methods:
            m = body_text(method, color=BLUE_LIGHT)
            h = label_text(how, color=WHITE)
            u = label_text(f"→ {used_in}", color=GREY_LIGHT)
            row = VGroup(m, h, u)
            row.arrange(RIGHT, buff=0.5)
            rows.add(row)

        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        rows.move_to(ORIGIN)
        box = SurroundingRectangle(rows, color=BLUE_MED, buff=0.35, corner_radius=0.15)
        heading = body_text("Positional Encoding Methods", color=WHITE)
        heading.next_to(box, UP, buff=0.2)

        self.play(Write(heading), Create(box), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.2),
                  run_time=1.2)
        self.wait(2)
