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
                                stroke_color=word_col, label=word, label_color=WHITE)
            plus   = body_text("+", color=WHITE)
            p_box  = rounded_box(1.3, 0.6, fill_color=GREY_DARK,
                                 stroke_color=pos_col,
                                 label=f"pos {pos}", label_color=WHITE)
            eq     = body_text("=", color=WHITE)
            r_box  = rounded_box(1.3, 0.6, fill_color=str(PURPLE_MED) + "33",
                                 stroke_color=PURPLE_MED,
                                 label="vector", label_color=WHITE)
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

        self.fade_all(heading, box, rows)

        # ── 5. Why attention is order-blind ───────────────────────────────────
        blind_title = body_text("Without PE — attention is completely order-blind", color=WHITE)
        blind_title.to_edge(UP, buff=0.8)
        self.play(Write(blind_title), run_time=0.7)

        # Show two sentences side by side
        sent1 = body_text('"The dog ate the bone"', color=GREEN_MED)
        sent2 = body_text('"bone the ate dog The"', color=RED_MED)
        sent1.shift(UP * 1.8 + LEFT * 2.5)
        sent2.shift(UP * 1.8 + RIGHT * 2.5)
        label1 = label_text("Original", color=GREEN_MED)
        label2 = label_text("Shuffled!", color=RED_MED)
        label1.next_to(sent1, UP, buff=0.15)
        label2.next_to(sent2, UP, buff=0.15)

        self.play(FadeIn(label1), Write(sent1), run_time=0.6)
        self.play(FadeIn(label2), Write(sent2), run_time=0.6)
        self.wait(0.5)

        # Two identical attention grids
        tokens_blind = ["dog", "ate", "bone"]
        scores_blind = [
            [0.8, 0.5, 0.3],
            [0.5, 0.9, 0.6],
            [0.3, 0.6, 0.7],
        ]

        grid1 = make_attention_grid(tokens_blind, scores_blind, cell_size=0.5)
        grid2 = make_attention_grid(tokens_blind, scores_blind, cell_size=0.5)
        grid1.shift(LEFT * 2.8 + DOWN * 0.5)
        grid2.shift(RIGHT * 2.8 + DOWN * 0.5)

        self.play(FadeIn(grid1), FadeIn(grid2), run_time=0.9)

        same_note = label_text("Both grids are IDENTICAL — attention alone can't tell order!", color=YELLOW_MED)
        same_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(same_note), run_time=0.5)
        self.wait(1.5)

        self.fade_all(blind_title, sent1, sent2, label1, label2, grid1, grid2, same_note)

        # ── 6. Adding PE to embeddings — element-wise addition ────────────────
        add_title = body_text("Adding PE to embeddings — element-wise addition", color=WHITE)
        add_title.to_edge(UP, buff=0.8)
        self.play(Write(add_title), run_time=0.7)

        # Show three bar charts: embedding + position = final
        np.random.seed(7)
        embed_vals = np.array([0.6, -0.4, 0.8, 0.2, -0.7, 0.5])
        pos_vals   = np.array([0.3,  0.1, -0.2, 0.4, 0.6, -0.3])
        final_vals = embed_vals + pos_vals

        def make_bar_group(vals, col, title_str):
            group = VGroup()
            bw = 0.28
            for v in vals:
                h = abs(v) * 1.2
                bar = Rectangle(width=bw, height=max(h, 0.02),
                                fill_color=col, fill_opacity=0.8,
                                stroke_color=WHITE, stroke_width=0.5)
                group.add(bar)
            group.arrange(RIGHT, buff=0.05)
            for i, (bar, v) in enumerate(zip(group, vals)):
                if v < 0:
                    bar.align_to(group[0], DOWN)
            ttl = label_text(title_str, color=col)
            ttl.next_to(group, UP, buff=0.2)
            return VGroup(ttl, group)

        emb_grp = make_bar_group(embed_vals, BLUE_MED, "Word Embedding")
        pos_grp = make_bar_group(pos_vals,   ORANGE_MED, "Position Vector")
        fin_grp = make_bar_group(final_vals, GREEN_MED,  "Final Input Vector")

        plus_sign  = body_text("+", color=WHITE)
        eq_sign    = body_text("=", color=WHITE)

        add_row = VGroup(emb_grp, plus_sign, pos_grp, eq_sign, fin_grp)
        add_row.arrange(RIGHT, buff=0.4)
        add_row.move_to(ORIGIN)

        self.play(FadeIn(emb_grp), run_time=0.6)
        self.play(FadeIn(plus_sign), FadeIn(pos_grp), run_time=0.5)
        self.play(FadeIn(eq_sign), FadeIn(fin_grp), run_time=0.6)

        elem_note = label_text("Each dimension gets: word_value + position_value", color=GREY_LIGHT)
        elem_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(elem_note), run_time=0.5)
        self.wait(1.5)

        self.fade_all(add_title, add_row, plus_sign, eq_sign, elem_note)

        # ── 7. Relative vs absolute position ─────────────────────────────────
        rel_title = body_text("Absolute vs Relative positional encoding", color=WHITE)
        rel_title.to_edge(UP, buff=0.8)
        self.play(Write(rel_title), run_time=0.7)

        # Absolute row
        abs_label = label_text("Absolute:", color=BLUE_MED)
        abs_label.shift(LEFT * 4.5 + UP * 1.2)
        abs_boxes = VGroup()
        for i in range(6):
            b = rounded_box(0.7, 0.5, fill_color=str(BLUE_MED) + "22",
                            stroke_color=BLUE_MED,
                            label=f"pos {i}", label_color=WHITE)
            abs_boxes.add(b)
        abs_boxes.arrange(RIGHT, buff=0.1)
        abs_boxes.next_to(abs_label, RIGHT, buff=0.25)
        abs_note = label_text("Token knows its global index — GPT-2, BERT", color=BLUE_MED)
        abs_note.next_to(abs_boxes, DOWN, buff=0.2)

        # Relative row
        rel_label = label_text("Relative:", color=ORANGE_MED)
        rel_label.shift(LEFT * 4.5 + DOWN * 0.8)
        rel_tokens = ["The", "cat", "sat", "on", "mat", "."]
        rel_boxes2 = VGroup()
        for tok in rel_tokens:
            b = rounded_box(0.7, 0.5, fill_color=str(ORANGE_MED) + "22",
                            stroke_color=ORANGE_MED, label=tok, label_color=WHITE)
            rel_boxes2.add(b)
        rel_boxes2.arrange(RIGHT, buff=0.1)
        rel_boxes2.next_to(rel_label, RIGHT, buff=0.25)

        # Arrow showing relative distance
        dist_arrow = Arrow(rel_boxes2[0].get_top() + UP * 0.3,
                           rel_boxes2[3].get_top() + UP * 0.3,
                           color=ORANGE_MED, stroke_width=2, buff=0)
        dist_lbl = label_text("+3", color=ORANGE_MED)
        dist_lbl.next_to(dist_arrow, UP, buff=0.1)
        rel_note = label_text("Tokens know distance between each other — RoPE, T5", color=ORANGE_MED)
        rel_note.next_to(rel_boxes2, DOWN, buff=0.2)

        self.play(FadeIn(abs_label), FadeIn(abs_boxes), run_time=0.6)
        self.play(FadeIn(abs_note), run_time=0.4)
        self.play(FadeIn(rel_label), FadeIn(rel_boxes2), run_time=0.6)
        self.play(Create(dist_arrow), FadeIn(dist_lbl), FadeIn(rel_note), run_time=0.7)
        self.wait(1.5)

        self.fade_all(rel_title, abs_label, abs_boxes, abs_note,
                      rel_label, rel_boxes2, dist_arrow, dist_lbl, rel_note)

        # ── 8. RoPE — rotating a 2D vector ────────────────────────────────────
        rope_title = body_text("RoPE — Rotary Position Embedding", color=WHITE)
        rope_title.to_edge(UP, buff=0.8)
        self.play(Write(rope_title), run_time=0.7)

        # Draw a circle with rotating vectors
        circle = Circle(radius=1.8, color=GREY_MED, stroke_width=1.5)
        circle.shift(LEFT * 1.5 + DOWN * 0.3)
        self.play(Create(circle), run_time=0.5)

        origin_pt = circle.get_center()
        angles = [0, PI / 6, PI / 3, PI / 2]
        pos_labels_rope = ["pos 0", "pos 1", "pos 2", "pos 3"]
        vec_colors_rope = [BLUE_MED, GREEN_MED, ORANGE_MED, PURPLE_MED]

        vectors_rope = VGroup()
        pos_lbls_rope = VGroup()
        for angle, pos_lbl_str, col in zip(angles, pos_labels_rope, vec_colors_rope):
            end_pt = origin_pt + np.array([1.8 * np.cos(angle), 1.8 * np.sin(angle), 0])
            vec = Arrow(origin_pt, end_pt, color=col, stroke_width=2.5,
                        buff=0, max_tip_length_to_length_ratio=0.18)
            lbl = label_text(pos_lbl_str, color=col)
            lbl.next_to(end_pt, UR if angle < PI / 2 else UP, buff=0.1)
            vectors_rope.add(vec)
            pos_lbls_rope.add(lbl)

        self.play(LaggedStart(*[Create(v) for v in vectors_rope], lag_ratio=0.2),
                  LaggedStart(*[FadeIn(l) for l in pos_lbls_rope], lag_ratio=0.2),
                  run_time=1.2)

        rope_explanation = VGroup(
            label_text("Each position rotates the vector by a fixed angle", color=WHITE),
            label_text("Attention between two tokens depends only on", color=GREY_LIGHT),
            label_text("the ANGLE DIFFERENCE — relative distance!", color=YELLOW_MED),
            label_text("Used in: LLaMA 2/3, Mistral, Qwen, Falcon", color=BLUE_LIGHT),
        )
        rope_explanation.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        rope_explanation.shift(RIGHT * 2.5 + DOWN * 0.3)

        self.play(LaggedStart(*[FadeIn(l) for l in rope_explanation], lag_ratio=0.2),
                  run_time=1.0)
        self.wait(1.5)

        self.fade_all(rope_title, circle, vectors_rope, pos_lbls_rope, rope_explanation)

        # ── 9. Context length timeline ────────────────────────────────────────
        ctx_title2 = body_text("Context length evolution — PE makes this possible", color=WHITE)
        ctx_title2.to_edge(UP, buff=0.8)
        self.play(Write(ctx_title2), run_time=0.7)

        ctx_data = [
            ("GPT-2\n(2019)",    1024,    GREY_MED),
            ("GPT-3\n(2020)",    2048,    BLUE_MED),
            ("GPT-4\n(2023)",    32768,   GREEN_MED),
            ("LLaMA 3\n(2024)",  131072,  ORANGE_MED),
            ("Gemini\n(2024)",   1000000, YELLOW_MED),
        ]

        max_val = 1000000
        bar_max_h = 3.2
        ctx_bars = VGroup()
        for label_str, tokens_val, col in ctx_data:
            h = max(bar_max_h * (tokens_val / max_val) ** 0.4, 0.05)
            bar = Rectangle(width=0.8, height=h,
                            fill_color=col, fill_opacity=0.85,
                            stroke_color=WHITE, stroke_width=1)
            lbl = label_text(label_str, color=col)
            lbl.next_to(bar, DOWN, buff=0.15)
            val_lbl = label_text(f"{tokens_val // 1000}k" if tokens_val >= 1000 else str(tokens_val), color=col)
            val_lbl.next_to(bar, UP, buff=0.1)
            group = VGroup(bar, lbl, val_lbl)
            ctx_bars.add(group)

        ctx_bars.arrange(RIGHT, buff=0.5)
        # Align bars to bottom
        for g in ctx_bars:
            g[0].align_to(ctx_bars[0][0], DOWN)
            g[1].next_to(g[0], DOWN, buff=0.15)
            g[2].next_to(g[0], UP, buff=0.1)

        ctx_bars.move_to(ORIGIN + DOWN * 0.3)

        self.play(LaggedStart(*[FadeIn(g) for g in ctx_bars], lag_ratio=0.15), run_time=1.2)

        ctx_note = label_text("RoPE and scaling tricks enable context windows to grow dramatically", color=GREY_LIGHT)
        ctx_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(ctx_note), run_time=0.5)
        self.wait(2)

        self.fade_all(ctx_title2, ctx_bars, ctx_note)
