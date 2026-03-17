"""
Scene 10 — Feed-Forward Networks
Run: manim -pql 10_feed_forward_networks.py FeedForwardScene
"""

from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class FeedForwardScene(LLMScene):
    def construct(self):
        title = self.show_title("Feed-Forward Networks", "The 'Thinking' Layers")
        self.wait(0.5)
        self.fade_all(title)

        # ── 1. FFN in context ─────────────────────────────────────────────────
        context_title = body_text("FFN comes after attention in every block:", color=WHITE)
        context_title.to_edge(UP, buff=0.6)
        self.play(Write(context_title), run_time=0.6)

        block_parts = [
            ("Attention\n(context sharing)", BLUE_MED),
            ("FFN\n(private thinking)", PURPLE_MED),
        ]
        block_boxes = VGroup()
        for lbl, col in block_parts:
            b = rounded_box(3.5, 1.1,
                            fill_color=col + "22",
                            stroke_color=col,
                            label=lbl, label_color=col)
            block_boxes.add(b)

        block_boxes.arrange(RIGHT, buff=0.8)
        block_boxes.move_to(ORIGIN)
        block_arr = Arrow(block_boxes[0].get_right(), block_boxes[1].get_left(),
                          color=WHITE, buff=0.05, stroke_width=2)

        attn_note = label_text("All tokens talk to each other", color=BLUE_MED)
        attn_note.next_to(block_boxes[0], DOWN, buff=0.2)
        ffn_note  = label_text("Each token processes alone", color=PURPLE_MED)
        ffn_note.next_to(block_boxes[1], DOWN, buff=0.2)

        self.play(FadeIn(block_boxes[0]), FadeIn(attn_note), run_time=0.6)
        self.play(GrowArrow(block_arr), run_time=0.4)
        self.play(FadeIn(block_boxes[1]), FadeIn(ffn_note), run_time=0.6)
        self.wait(1)
        self.fade_all(context_title, block_boxes, block_arr, attn_note, ffn_note)

        # ── 2. Expand → activate → compress ──────────────────────────────────
        arch_title = body_text("FFN structure: expand wide, then compress", color=WHITE)
        arch_title.to_edge(UP, buff=0.6)
        self.play(Write(arch_title), run_time=0.6)

        stage_data = [
            ("Input\n4096 dim",  BLUE_MED,   1.0, "token embedding"),
            ("Expand\n16384 dim", GREEN_MED,  2.5, "\"brainstorm wide\""),
            ("Activate\n(SiLU)",  ORANGE_MED, 2.5, "filter — keep strong signals"),
            ("Compress\n4096 dim", BLUE_MED,  1.0, "\"distil the insight\""),
        ]

        bars = VGroup()
        for lbl, col, height_scale, note in stage_data:
            bar = Rectangle(width=0.8, height=height_scale,
                            fill_color=col + "55", fill_opacity=1,
                            stroke_color=col, stroke_width=2)
            lbl_txt = label_text(lbl, color=col)
            lbl_txt.next_to(bar, DOWN, buff=0.15)
            note_txt = label_text(note, color=GREY_LIGHT)
            note_txt.next_to(bar, UP, buff=0.15)
            bars.add(VGroup(bar, lbl_txt, note_txt))

        bars.arrange(RIGHT, buff=1.0)
        bars.move_to(ORIGIN + DOWN * 0.3)

        # Arrows between bars
        bar_arrows = VGroup()
        for i in range(len(bars) - 1):
            arr = Arrow(bars[i][0].get_right(), bars[i + 1][0].get_left(),
                        color=GREY_MED, buff=0.05, stroke_width=1.5,
                        max_tip_length_to_length_ratio=0.2)
            bar_arrows.add(arr)

        self.play(LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.2),
                  run_time=1.5)
        self.play(LaggedStart(*[GrowArrow(a) for a in bar_arrows], lag_ratio=0.2),
                  run_time=0.8)
        self.wait(1.2)
        self.fade_all(arch_title, bars, bar_arrows)

        # ── 3. SwiGLU gate visual ─────────────────────────────────────────────
        swiglu_title = body_text("SwiGLU — the gated FFN used in LLaMA, Gemma, PaLM",
                                  color=WHITE)
        swiglu_title.to_edge(UP, buff=0.6)
        self.play(Write(swiglu_title), run_time=0.7)

        input_box = rounded_box(1.4, 0.6, stroke_color=BLUE_MED, label="Input x")
        input_box.shift(LEFT * 4)

        w1_box = rounded_box(1.6, 0.6, stroke_color=GREEN_MED,
                             label="W1 × x\n(main)", label_color=GREEN_MED)
        w1_box.shift(LEFT * 1.5 + UP * 0.8)

        w3_box = rounded_box(1.6, 0.6, stroke_color=ORANGE_MED,
                             label="W3 × x\n(gate)", label_color=ORANGE_MED)
        w3_box.shift(LEFT * 1.5 + DOWN * 0.8)

        silu_box = rounded_box(1.4, 0.6, stroke_color=ORANGE_MED,
                               label="SiLU( ·)", label_color=ORANGE_MED)
        silu_box.next_to(w1_box, RIGHT, buff=0.6)

        mult_dot = Dot(color=WHITE, radius=0.18)
        mult_dot.next_to(silu_box, RIGHT, buff=0.8)
        mult_label = body_text("⊙", color=WHITE).move_to(mult_dot)

        w2_box = rounded_box(1.6, 0.6, stroke_color=BLUE_LIGHT,
                             label="W2 × ( ·)", label_color=BLUE_LIGHT)
        w2_box.next_to(mult_dot, RIGHT, buff=0.8)

        out_box = rounded_box(1.4, 0.6, stroke_color=BLUE_MED, label="Output")
        out_box.next_to(w2_box, RIGHT, buff=0.6)

        # Arrows
        def mk_arrow(s, e, col=GREY_MED):
            return Arrow(s.get_right(), e.get_left(), color=col,
                         buff=0.05, stroke_width=1.5,
                         max_tip_length_to_length_ratio=0.18)

        a1 = Arrow(input_box.get_right(), w1_box.get_left(), buff=0.05,
                   color=GREY_MED, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        a2 = Arrow(input_box.get_right(), w3_box.get_left(), buff=0.05,
                   color=GREY_MED, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        a3 = mk_arrow(w1_box, silu_box)
        a4 = Arrow(w3_box.get_right(),
                   mult_label.get_left() + DOWN * 0.4,
                   buff=0.05, color=ORANGE_MED, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        a5 = mk_arrow(silu_box, mult_label.copy().shift(LEFT * 0.2))
        a6 = mk_arrow(mult_dot, w2_box)
        a7 = mk_arrow(w2_box, out_box)

        diagram = VGroup(input_box, w1_box, w3_box, silu_box,
                         mult_label, w2_box, out_box,
                         a1, a2, a3, a4, a5, a6, a7)
        diagram.scale_to_fit_width(13)
        diagram.move_to(ORIGIN + DOWN * 0.3)

        self.play(LaggedStart(*[FadeIn(m) for m in [
            input_box, a1, a2, w1_box, w3_box, a3, silu_box,
            a4, a5, mult_label, a6, w2_box, a7, out_box
        ]], lag_ratio=0.08), run_time=2.0)

        gate_note = label_text(
            "The gate (W3 path) acts like a spotlight:\n"
            "it decides how much of W1's output to let through.",
            color=GREY_LIGHT,
        )
        gate_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(gate_note), run_time=0.7)
        self.wait(2)
