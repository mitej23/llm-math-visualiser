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
                            fill_color=str(col) + "22",
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
        self.play(Create(block_arr), run_time=0.4)
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
                            fill_color=str(col) + "55", fill_opacity=1,
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
        self.play(LaggedStart(*[Create(a) for a in bar_arrows], lag_ratio=0.2),
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
                             label="W1 x x\n(main)", label_color=GREEN_MED)
        w1_box.shift(LEFT * 1.5 + UP * 0.8)

        w3_box = rounded_box(1.6, 0.6, stroke_color=ORANGE_MED,
                             label="W3 x x\n(gate)", label_color=ORANGE_MED)
        w3_box.shift(LEFT * 1.5 + DOWN * 0.8)

        silu_box = rounded_box(1.4, 0.6, stroke_color=ORANGE_MED,
                               label="SiLU( . )", label_color=ORANGE_MED)
        silu_box.next_to(w1_box, RIGHT, buff=0.6)

        mult_label = body_text("x", color=WHITE)
        mult_label.next_to(silu_box, RIGHT, buff=0.8)

        w2_box = rounded_box(1.6, 0.6, stroke_color=BLUE_LIGHT,
                             label="W2 x ( . )", label_color=BLUE_LIGHT)
        w2_box.next_to(mult_label, RIGHT, buff=0.8)

        out_box = rounded_box(1.4, 0.6, stroke_color=BLUE_MED, label="Output")
        out_box.next_to(w2_box, RIGHT, buff=0.6)

        a1 = Arrow(input_box.get_right(), w1_box.get_left(), buff=0.05,
                   color=GREY_MED, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        a2 = Arrow(input_box.get_right(), w3_box.get_left(), buff=0.05,
                   color=GREY_MED, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        a3 = Arrow(w1_box.get_right(), silu_box.get_left(), buff=0.05,
                   color=GREY_MED, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        a4 = Arrow(w3_box.get_right(), mult_label.get_left() + DOWN * 0.3,
                   buff=0.05, color=ORANGE_MED, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        a5 = Arrow(silu_box.get_right(), mult_label.get_left(), buff=0.05,
                   color=GREY_MED, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        a6 = Arrow(mult_label.get_right(), w2_box.get_left(), buff=0.05,
                   color=GREY_MED, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        a7 = Arrow(w2_box.get_right(), out_box.get_left(), buff=0.05,
                   color=GREY_MED, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)

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
        self.fade_all(swiglu_title, diagram, gate_note)

        # ── 4. FFN position in the full transformer block ─────────────────────
        block_title = body_text("Full transformer block — FFN is the second half:", color=WHITE)
        block_title.to_edge(UP, buff=0.6)
        self.play(Write(block_title), run_time=0.7)

        block_steps = [
            ("Input x", GREY_LIGHT, False),
            ("LayerNorm", YELLOW_MED, False),
            ("Multi-Head\nAttention", BLUE_MED, False),
            ("Add + Residual", GREY_LIGHT, False),
            ("LayerNorm", YELLOW_MED, False),
            ("FFN", PURPLE_MED, True),
            ("Add + Residual", GREY_LIGHT, False),
            ("Output x", GREY_LIGHT, False),
        ]

        step_boxes = VGroup()
        for lbl, col, is_highlight in block_steps:
            stroke = col
            fill = str(PURPLE_MED) + "44" if is_highlight else str(col) + "22"
            b = rounded_box(2.0, 0.55, fill_color=fill,
                            stroke_color=stroke, label=lbl, label_color=col)
            step_boxes.add(b)

        step_boxes.arrange(DOWN, buff=0.12)
        step_boxes.scale_to_fit_height(6.5)
        step_boxes.move_to(ORIGIN + DOWN * 0.2)

        step_arrows = VGroup()
        for i in range(len(step_boxes) - 1):
            arr = Arrow(step_boxes[i].get_bottom(), step_boxes[i + 1].get_top(),
                        color=GREY_MED, buff=0.05, stroke_width=1.5,
                        max_tip_length_to_length_ratio=0.2)
            step_arrows.add(arr)

        self.play(LaggedStart(*[FadeIn(b) for b in step_boxes], lag_ratio=0.1),
                  run_time=1.5)
        self.play(LaggedStart(*[Create(a) for a in step_arrows], lag_ratio=0.1),
                  run_time=0.8)

        ffn_label = label_text("FFN is here — knowledge retrieval", color=PURPLE_MED)
        ffn_label.next_to(step_boxes[5], RIGHT, buff=0.4)
        ffn_brace = Arrow(ffn_label.get_left(), step_boxes[5].get_right(),
                          color=PURPLE_MED, buff=0.05, stroke_width=1.5,
                          max_tip_length_to_length_ratio=0.2)
        self.play(FadeIn(ffn_label), Create(ffn_brace), run_time=0.7)
        self.wait(1.5)
        self.fade_all(block_title, step_boxes, step_arrows, ffn_label, ffn_brace)

        # ── 5. Expand and contract — d_model vs d_ff ─────────────────────────
        expand_title = body_text("Expand 4x wide, then contract back down:", color=WHITE)
        expand_title.to_edge(UP, buff=0.6)
        self.play(Write(expand_title), run_time=0.7)

        narrow_in = Rectangle(width=1.0, height=4.0,
                              fill_color=str(BLUE_MED) + "44", fill_opacity=1,
                              stroke_color=BLUE_MED, stroke_width=2)
        narrow_in.shift(LEFT * 4.5)
        narrow_in_lbl = label_text("d_model\n512", color=BLUE_MED)
        narrow_in_lbl.next_to(narrow_in, DOWN, buff=0.2)

        wide = Rectangle(width=1.0, height=4.0,
                         fill_color=str(GREEN_MED) + "44", fill_opacity=1,
                         stroke_color=GREEN_MED, stroke_width=2)
        wide.stretch_to_fit_width(4.0)
        wide.move_to(ORIGIN)
        wide_lbl = label_text("d_ff\n2048\n(4x wider)", color=GREEN_MED)
        wide_lbl.next_to(wide, DOWN, buff=0.2)

        narrow_out = Rectangle(width=1.0, height=4.0,
                               fill_color=str(BLUE_MED) + "44", fill_opacity=1,
                               stroke_color=BLUE_MED, stroke_width=2)
        narrow_out.shift(RIGHT * 4.5)
        narrow_out_lbl = label_text("d_model\n512", color=BLUE_MED)
        narrow_out_lbl.next_to(narrow_out, DOWN, buff=0.2)

        exp_arr = Arrow(narrow_in.get_right(), wide.get_left(),
                        color=GREEN_MED, buff=0.05, stroke_width=2)
        con_arr = Arrow(wide.get_right(), narrow_out.get_left(),
                        color=BLUE_MED, buff=0.05, stroke_width=2)

        exp_lbl = label_text("W1: expand", color=GREEN_MED)
        exp_lbl.next_to(exp_arr, UP, buff=0.1)
        con_lbl = label_text("W2: compress", color=BLUE_MED)
        con_lbl.next_to(con_arr, UP, buff=0.1)

        self.play(FadeIn(narrow_in), FadeIn(narrow_in_lbl), run_time=0.5)
        self.play(Create(exp_arr), FadeIn(exp_lbl), run_time=0.5)
        self.play(FadeIn(wide), FadeIn(wide_lbl), run_time=0.6)
        self.play(Create(con_arr), FadeIn(con_lbl), run_time=0.5)
        self.play(FadeIn(narrow_out), FadeIn(narrow_out_lbl), run_time=0.5)

        room_note = label_text(
            "More width = more 'room' to detect complex features\nbefore distilling the answer",
            color=GREY_LIGHT,
        )
        room_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(room_note), run_time=0.6)
        self.wait(1.5)
        self.fade_all(expand_title, narrow_in, narrow_in_lbl, wide, wide_lbl,
                      narrow_out, narrow_out_lbl, exp_arr, exp_lbl,
                      con_arr, con_lbl, room_note)

        # ── 6. FFN as key-value memory ────────────────────────────────────────
        kv_title = body_text("FFN acts as a key-value memory store:", color=WHITE)
        kv_title.to_edge(UP, buff=0.6)
        self.play(Write(kv_title), run_time=0.7)

        w1_mem = rounded_box(3.2, 1.0,
                             fill_color=str(GREEN_MED) + "22",
                             stroke_color=GREEN_MED,
                             label="W1 (key matrix)\nLooks for patterns in input",
                             label_color=GREEN_MED)
        w1_mem.shift(LEFT * 2.5 + UP * 0.5)

        act_mem = rounded_box(2.8, 0.8,
                              fill_color=str(ORANGE_MED) + "22",
                              stroke_color=ORANGE_MED,
                              label="Activation\nFires for matching patterns",
                              label_color=ORANGE_MED)
        act_mem.shift(ORIGIN + UP * 0.5)

        w2_mem = rounded_box(3.2, 1.0,
                             fill_color=str(BLUE_MED) + "22",
                             stroke_color=BLUE_MED,
                             label="W2 (value matrix)\nRetrieves stored facts",
                             label_color=BLUE_MED)
        w2_mem.shift(RIGHT * 2.5 + UP * 0.5)

        arr_kv1 = Arrow(w1_mem.get_right(), act_mem.get_left(),
                        color=GREY_MED, buff=0.05, stroke_width=1.5,
                        max_tip_length_to_length_ratio=0.2)
        arr_kv2 = Arrow(act_mem.get_right(), w2_mem.get_left(),
                        color=GREY_MED, buff=0.05, stroke_width=1.5,
                        max_tip_length_to_length_ratio=0.2)

        example_note = label_text(
            "Example: Input context 'Paris is the capital of ___'\n"
            "W1 fires a 'capital city lookup' key -> W2 outputs 'France'",
            color=GREY_LIGHT,
        )
        example_note.to_edge(DOWN, buff=0.4)

        self.play(FadeIn(w1_mem), run_time=0.5)
        self.play(Create(arr_kv1), FadeIn(act_mem), run_time=0.6)
        self.play(Create(arr_kv2), FadeIn(w2_mem), run_time=0.6)
        self.play(FadeIn(example_note), run_time=0.7)
        self.wait(2)
        self.fade_all(kv_title, w1_mem, act_mem, w2_mem,
                      arr_kv1, arr_kv2, example_note)

        # ── 7. FFN vs attention — side-by-side ────────────────────────────────
        compare_title = body_text("Attention vs FFN — different jobs:", color=WHITE)
        compare_title.to_edge(UP, buff=0.6)
        self.play(Write(compare_title), run_time=0.7)

        attn_box = rounded_box(5.0, 3.5,
                               fill_color=str(BLUE_MED) + "22",
                               stroke_color=BLUE_MED,
                               label="", label_color=BLUE_MED)
        attn_box.shift(LEFT * 3.3)

        ffn_box = rounded_box(5.0, 3.5,
                              fill_color=str(PURPLE_MED) + "22",
                              stroke_color=PURPLE_MED,
                              label="", label_color=PURPLE_MED)
        ffn_box.shift(RIGHT * 3.3)

        attn_heading = body_text("Attention", color=BLUE_MED)
        attn_heading.move_to(attn_box.get_top() + DOWN * 0.35)

        attn_facts = VGroup(
            label_text("Mixes tokens together", color=WHITE),
            label_text("Cross-token communication", color=WHITE),
            label_text("'What is relevant here?'", color=GREY_LIGHT),
            label_text("Varies per sequence", color=GREY_LIGHT),
        )
        attn_facts.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        attn_facts.move_to(attn_box.get_center() + DOWN * 0.2)

        ffn_heading = body_text("FFN", color=PURPLE_MED)
        ffn_heading.move_to(ffn_box.get_top() + DOWN * 0.35)

        ffn_facts = VGroup(
            label_text("Processes each token alone", color=WHITE),
            label_text("No cross-token communication", color=WHITE),
            label_text("'What do I know about this?'", color=GREY_LIGHT),
            label_text("Same weights for all tokens", color=GREY_LIGHT),
        )
        ffn_facts.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        ffn_facts.move_to(ffn_box.get_center() + DOWN * 0.2)

        self.play(FadeIn(attn_box), FadeIn(ffn_box), run_time=0.5)
        self.play(FadeIn(attn_heading), FadeIn(ffn_heading), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(f) for f in attn_facts], lag_ratio=0.15),
                  run_time=0.8)
        self.play(LaggedStart(*[FadeIn(f) for f in ffn_facts], lag_ratio=0.15),
                  run_time=0.8)

        together_note = label_text(
            "Together: attention gathers context, FFN applies knowledge to it",
            color=YELLOW_MED,
        )
        together_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(together_note), run_time=0.6)
        self.wait(2)
        self.fade_all(compare_title, attn_box, ffn_box, attn_heading, ffn_heading,
                      attn_facts, ffn_facts, together_note)
