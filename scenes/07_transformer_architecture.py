"""
Scene 07 — Transformer Architecture
Run: manim -pql 07_transformer_architecture.py TransformerArchitectureScene
"""

from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class TransformerArchitectureScene(LLMScene):
    def construct(self):
        title = self.show_title("Transformer Architecture", "The Big Picture")
        self.wait(0.5)
        self.fade_all(title)

        # ── Full pipeline — left to right ─────────────────────────────────────
        stages = [
            ("Input\nText",       GREY_MED,    "\"The cat...\""),
            ("Tokenize",          BLUE_MED,    "words → IDs"),
            ("Embed\n+ Position", GREEN_MED,   "IDs → vectors"),
            ("Transformer\nBlocks×N", PURPLE_MED, "attention + FFN"),
            ("Output\nHead",      ORANGE_MED,  "project to vocab"),
            ("Next\nToken",       YELLOW_MED,  "sample one token"),
        ]

        boxes = VGroup()
        for label, col, sub in stages:
            box = rounded_box(1.5, 1.0,
                              fill_color=str(col) + "22",
                              stroke_color=col,
                              label=label, label_color=col)
            sub_txt = label_text(sub, color=GREY_LIGHT)
            sub_txt.next_to(box, DOWN, buff=0.15)
            boxes.add(VGroup(box, sub_txt))

        boxes.arrange(RIGHT, buff=0.6)
        boxes.scale_to_fit_width(13)
        boxes.move_to(ORIGIN)

        arrows = VGroup()
        for i in range(len(boxes) - 1):
            start = boxes[i][0].get_right()
            end   = boxes[i + 1][0].get_left()
            arr = Arrow(start, end, color=GREY_MED,
                        buff=0.05, stroke_width=2,
                        max_tip_length_to_length_ratio=0.2)
            arrows.add(arr)

        self.play(LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.15),
                  run_time=1.8)
        self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.1),
                  run_time=1.0)
        self.wait(1)

        # ── Highlight the transformer block ───────────────────────────────────
        block_box = boxes[3][0]
        hl = SurroundingRectangle(block_box, color=YELLOW_MED,
                                  buff=0.12, corner_radius=0.1, stroke_width=2)
        hl_lbl = label_text("This repeats 32–96×", color=YELLOW_MED)
        hl_lbl.next_to(hl, UP, buff=0.2)
        self.play(Create(hl), FadeIn(hl_lbl), run_time=0.7)
        self.wait(0.8)
        self.fade_all(hl, hl_lbl)

        # ── Zoom into transformer block internals ─────────────────────────────
        self.fade_all(boxes, arrows)

        block_title = body_text("Inside One Transformer Block", color=WHITE)
        block_title.to_edge(UP, buff=0.6)
        self.play(Write(block_title), run_time=0.6)

        internals = [
            ("Layer\nNorm",          GREY_LIGHT,  0.0),
            ("Multi-Head\nAttention",BLUE_MED,    0.0),
            ("Residual  +",          GREEN_LIGHT, 0.0),
            ("Layer\nNorm",          GREY_LIGHT,  0.0),
            ("Feed-Forward\nNetwork",PURPLE_MED,  0.0),
            ("Residual  +",          GREEN_LIGHT, 0.0),
        ]

        stack = VGroup()
        for lbl, col, _ in internals:
            b = rounded_box(3.2, 0.7,
                            fill_color=str(col) + "22",
                            stroke_color=col,
                            label=lbl, label_color=col)
            stack.add(b)

        stack.arrange(DOWN, buff=0.22)
        stack.move_to(ORIGIN + DOWN * 0.2)

        v_arrows = VGroup()
        for i in range(len(stack) - 1):
            arr = Arrow(stack[i].get_bottom(), stack[i + 1].get_top(),
                        color=GREY_MED, buff=0.04, stroke_width=1.5,
                        max_tip_length_to_length_ratio=0.2)
            v_arrows.add(arr)

        # Residual bypass arrows
        res1 = CurvedArrow(stack[0].get_left() + LEFT * 0.1,
                           stack[2].get_left() + LEFT * 0.1,
                           angle=-TAU / 6, color=GREEN_LIGHT, stroke_width=1.5)
        res2 = CurvedArrow(stack[3].get_left() + LEFT * 0.1,
                           stack[5].get_left() + LEFT * 0.1,
                           angle=-TAU / 6, color=GREEN_LIGHT, stroke_width=1.5)
        res_lbl1 = label_text("skip", color=GREEN_LIGHT)
        res_lbl1.next_to(res1, LEFT, buff=0.1)
        res_lbl2 = label_text("skip", color=GREEN_LIGHT)
        res_lbl2.next_to(res2, LEFT, buff=0.1)

        self.play(LaggedStart(*[FadeIn(b) for b in stack], lag_ratio=0.1),
                  run_time=1.2)
        self.play(LaggedStart(*[Create(a) for a in v_arrows], lag_ratio=0.08),
                  run_time=0.8)
        self.play(Create(res1), Create(res2),
                  FadeIn(res_lbl1), FadeIn(res_lbl2), run_time=0.7)
        self.wait(1.5)

        # ── Residual connection callout ────────────────────────────────────────
        res_note = body_text(
            "Residual connections let gradients skip straight through —\n"
            "each layer learns only the correction, not the whole thing.",
            color=GREY_LIGHT,
        )
        res_note.to_edge(RIGHT, buff=0.4)
        res_note.shift(UP * 0.5)
        self.play(Write(res_note), run_time=1.0)
        self.wait(2)
        self.fade_all(block_title, stack, v_arrows, res1, res2,
                      res_lbl1, res_lbl2, res_note)

        # ── Section: Encoder vs Decoder vs Encoder-Decoder ───────────────────
        enc_dec_title = body_text("Three Transformer Flavours", color=WHITE)
        enc_dec_title.to_edge(UP, buff=0.6)
        self.play(Write(enc_dec_title), run_time=0.6)

        flavours = [
            ("BERT\nEncoder-Only",    BLUE_MED,
             "Bidirectional\nReads full context",
             "Classification\nEmbeddings"),
            ("GPT\nDecoder-Only",     GREEN_MED,
             "Causal (left-to-right)\nGenerates token by token",
             "Text generation\nChat, code"),
            ("T5\nEncoder-Decoder",   ORANGE_MED,
             "Encoder reads input\nDecoder generates output",
             "Translation\nSummarisation"),
        ]

        flavour_boxes = VGroup()
        for name, col, desc, use in flavours:
            title_t = body_text(name, color=col)
            desc_t  = label_text(desc, color=GREY_LIGHT)
            use_t   = label_text(use, color=WHITE)
            desc_t.next_to(title_t, DOWN, buff=0.15)
            use_t.next_to(desc_t, DOWN, buff=0.15)
            content = VGroup(title_t, desc_t, use_t)
            bg = SurroundingRectangle(content, color=str(col) + "55",
                                      fill_color=str(col) + "11",
                                      fill_opacity=1, buff=0.3,
                                      corner_radius=0.15)
            flavour_boxes.add(VGroup(bg, content))

        flavour_boxes.arrange(RIGHT, buff=0.5)
        flavour_boxes.move_to(ORIGIN + DOWN * 0.2)

        self.play(LaggedStart(*[FadeIn(fb) for fb in flavour_boxes], lag_ratio=0.25),
                  run_time=1.5)
        self.wait(1.5)

        trend_note = label_text(
            "Modern chat models (GPT-4, Claude, LLaMA) are all Decoder-Only",
            color=YELLOW_MED,
        )
        trend_note.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(trend_note), run_time=0.6)
        self.wait(1.5)
        self.fade_all(enc_dec_title, flavour_boxes, trend_note)

        # ── Section: Single transformer block — detailed stack ────────────────
        detail_title = body_text("One Block, Step by Step", color=WHITE)
        detail_title.to_edge(UP, buff=0.6)
        self.play(Write(detail_title), run_time=0.6)

        steps = [
            ("Pre-LayerNorm",       GREY_LIGHT,  "Stabilise values before attention"),
            ("Multi-Head Attention",BLUE_MED,    "Each token gathers context"),
            ("Residual Add  +",     GREEN_LIGHT, "x = x + Attention(x)"),
            ("Pre-LayerNorm",       GREY_LIGHT,  "Stabilise before FFN"),
            ("Feed-Forward Network",PURPLE_MED,  "Think: expand 4x, then compress"),
            ("Residual Add  +",     GREEN_LIGHT, "x = x + FFN(x)"),
        ]

        detail_stack = VGroup()
        step_notes = VGroup()
        for lbl, col, note in steps:
            b = rounded_box(3.0, 0.65,
                            fill_color=str(col) + "22",
                            stroke_color=col,
                            label=lbl, label_color=col)
            n = label_text(note, color=GREY_MED)
            detail_stack.add(b)
            step_notes.add(n)

        detail_stack.arrange(DOWN, buff=0.2)
        detail_stack.shift(LEFT * 1.5 + DOWN * 0.2)

        for i, (b, n) in enumerate(zip(detail_stack, step_notes)):
            n.next_to(b, RIGHT, buff=0.4)

        self.play(LaggedStart(*[FadeIn(b) for b in detail_stack], lag_ratio=0.1),
                  run_time=1.2)
        self.play(LaggedStart(*[FadeIn(n) for n in step_notes], lag_ratio=0.1),
                  run_time=1.0)
        self.wait(2)
        self.fade_all(detail_title, detail_stack, step_notes)

        # ── Section: Residual connections — signal flow diagram ───────────────
        res_title = body_text("Residual Connections — The Skip Highway", color=WHITE)
        res_title.to_edge(UP, buff=0.6)
        self.play(Write(res_title), run_time=0.6)

        # Show: input → block → output, with bypass
        input_box  = rounded_box(2.2, 0.75, fill_color=str(GREY_MED) + "22",
                                  stroke_color=GREY_MED, label="Input  x", label_color=GREY_LIGHT)
        block_box2 = rounded_box(2.2, 0.75, fill_color=str(BLUE_MED) + "22",
                                  stroke_color=BLUE_MED, label="Layer (e.g. Attention)",
                                  label_color=BLUE_MED)
        add_box    = rounded_box(2.2, 0.75, fill_color=str(GREEN_MED) + "22",
                                  stroke_color=GREEN_MED, label="Add  (+)", label_color=GREEN_MED)
        output_box = rounded_box(2.2, 0.75, fill_color=str(GREY_MED) + "22",
                                  stroke_color=GREY_MED, label="Output  x'", label_color=GREY_LIGHT)

        main_stack = VGroup(input_box, block_box2, add_box, output_box)
        main_stack.arrange(DOWN, buff=0.4)
        main_stack.shift(LEFT * 0.5)

        skip_arrow = CurvedArrow(
            input_box.get_right() + RIGHT * 0.1,
            add_box.get_right() + RIGHT * 0.1,
            angle=TAU / 5, color=YELLOW_MED, stroke_width=2.5
        )
        skip_lbl = label_text("Skip connection\n(original x)", color=YELLOW_MED)
        skip_lbl.next_to(skip_arrow, RIGHT, buff=0.15)

        main_arrows = VGroup()
        for i in range(len(main_stack) - 1):
            a = Arrow(main_stack[i].get_bottom(), main_stack[i+1].get_top(),
                      color=GREY_MED, buff=0.05, stroke_width=1.5,
                      max_tip_length_to_length_ratio=0.2)
            main_arrows.add(a)

        self.play(LaggedStart(*[FadeIn(b) for b in main_stack], lag_ratio=0.15),
                  run_time=1.0)
        self.play(LaggedStart(*[Create(a) for a in main_arrows], lag_ratio=0.1),
                  run_time=0.7)
        self.play(Create(skip_arrow), FadeIn(skip_lbl), run_time=0.8)

        res_note2 = body_text(
            "Even if the block learns nothing useful,\n"
            "the skip preserves the original signal.",
            color=GREY_LIGHT,
        )
        res_note2.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(res_note2), run_time=0.7)
        self.wait(2)
        self.fade_all(res_title, main_stack, main_arrows,
                      skip_arrow, skip_lbl, res_note2)

        # ── Section: How many layers? — bar chart ─────────────────────────────
        layers_title = body_text("How Many Layers Do Real Models Have?", color=WHITE)
        layers_title.to_edge(UP, buff=0.6)
        self.play(Write(layers_title), run_time=0.6)

        model_data = [
            ("GPT-2",  12,  BLUE_LIGHT),
            ("GPT-3",  96,  BLUE_MED),
            ("LLaMA 3\n8B", 32, GREEN_MED),
            ("LLaMA 3\n70B", 80, GREEN_DARK),
            ("GPT-4\n(est.)", 120, PURPLE_MED),
        ]

        max_layers = 120
        bar_h_max  = 3.5
        bar_w      = 1.0
        bars = VGroup()

        for i, (name, n_layers, col) in enumerate(model_data):
            h = bar_h_max * (n_layers / max_layers)
            bar = Rectangle(width=bar_w, height=max(h, 0.05),
                            fill_color=col, fill_opacity=0.85,
                            stroke_color=WHITE, stroke_width=1)
            count_lbl = label_text(str(n_layers), color=WHITE)
            count_lbl.next_to(bar, UP, buff=0.1)
            name_lbl = label_text(name, color=GREY_LIGHT)
            name_lbl.next_to(bar, DOWN, buff=0.15)
            grp = VGroup(bar, count_lbl, name_lbl)
            grp.shift(RIGHT * i * (bar_w + 0.55))
            bars.add(grp)

        bars.move_to(ORIGIN + DOWN * 0.3)

        self.play(LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.18),
                  run_time=1.5)

        layers_note = label_text(
            "Each layer refines representations — depth ≈ reasoning power",
            color=GREY_LIGHT,
        )
        layers_note.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(layers_note), run_time=0.6)
        self.wait(2)
        self.fade_all(layers_title, bars, layers_note)

        # ── Section: What each layer learns — vertical stack ──────────────────
        learns_title = body_text("What Each Layer Learns", color=WHITE)
        learns_title.to_edge(UP, buff=0.6)
        self.play(Write(learns_title), run_time=0.6)

        layer_groups = [
            ("Lower Layers\n(1–30%)",  RED_MED,    "Spelling, part-of-speech, local syntax"),
            ("Middle Layers\n(30–70%)", ORANGE_MED, "Word sense, coreference, semantics"),
            ("Upper Layers\n(70–100%)", GREEN_MED,  "Reasoning, facts, task completion"),
        ]

        lg_stack = VGroup()
        lg_notes = VGroup()
        for label, col, note in layer_groups:
            b = rounded_box(3.5, 0.8,
                            fill_color=str(col) + "22",
                            stroke_color=col,
                            label=label, label_color=col)
            n = label_text(note, color=WHITE)
            lg_stack.add(b)
            lg_notes.add(n)

        lg_stack.arrange(DOWN, buff=0.35)
        lg_stack.shift(LEFT * 1.8)

        for b, n in zip(lg_stack, lg_notes):
            n.next_to(b, RIGHT, buff=0.5)
            n.align_to(b, UP)

        arrow_up = Arrow(lg_stack[-1].get_top(), lg_stack[0].get_bottom() + UP * 0.05,
                         color=GREY_MED, stroke_width=1.5,
                         max_tip_length_to_length_ratio=0.15)
        arrow_up.next_to(lg_stack, LEFT, buff=0.35)
        depth_lbl = label_text("depth", color=GREY_MED)
        depth_lbl.next_to(arrow_up, LEFT, buff=0.12)

        self.play(LaggedStart(*[FadeIn(b) for b in lg_stack], lag_ratio=0.2),
                  run_time=1.0)
        self.play(LaggedStart(*[FadeIn(n) for n in lg_notes], lag_ratio=0.2),
                  run_time=1.0)
        self.play(Create(arrow_up), FadeIn(depth_lbl), run_time=0.6)
        self.wait(2)
        self.fade_all(learns_title, lg_stack, lg_notes, arrow_up, depth_lbl)

        # ── Section: Pre-norm vs Post-norm ────────────────────────────────────
        norm_title = body_text("Pre-norm vs Post-norm", color=WHITE)
        norm_title.to_edge(UP, buff=0.6)
        self.play(Write(norm_title), run_time=0.6)

        # Post-norm diagram (left)
        post_label = body_text("Post-norm\n(Original 2017)", color=ORANGE_MED)
        post_label.shift(LEFT * 3.5 + UP * 2.0)

        post_steps = ["Attention", "Add  +", "LayerNorm"]
        post_stack = VGroup()
        for s in post_steps:
            col = BLUE_MED if s == "Attention" else (GREEN_MED if s == "Add  +" else GREY_LIGHT)
            b = rounded_box(2.2, 0.6, fill_color=str(col) + "22",
                            stroke_color=col, label=s, label_color=col)
            post_stack.add(b)
        post_stack.arrange(DOWN, buff=0.25)
        post_stack.shift(LEFT * 3.5 + DOWN * 0.3)

        # Pre-norm diagram (right)
        pre_label = body_text("Pre-norm\n(Modern LLMs)", color=GREEN_MED)
        pre_label.shift(RIGHT * 3.0 + UP * 2.0)

        pre_steps = ["LayerNorm", "Attention", "Add  +"]
        pre_stack = VGroup()
        for s in pre_steps:
            col = GREY_LIGHT if s == "LayerNorm" else (BLUE_MED if s == "Attention" else GREEN_MED)
            b = rounded_box(2.2, 0.6, fill_color=str(col) + "22",
                            stroke_color=col, label=s, label_color=col)
            pre_stack.add(b)
        pre_stack.arrange(DOWN, buff=0.25)
        pre_stack.shift(RIGHT * 3.0 + DOWN * 0.3)

        vs_txt = body_text("vs.", color=GREY_MED)
        vs_txt.move_to(ORIGIN + DOWN * 0.3)

        self.play(FadeIn(post_label), FadeIn(pre_label), FadeIn(vs_txt), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(b) for b in post_stack], lag_ratio=0.15),
                  LaggedStart(*[FadeIn(b) for b in pre_stack], lag_ratio=0.15),
                  run_time=1.0)

        stable_note = label_text(
            "Pre-norm: LayerNorm runs BEFORE the sublayer\n"
            "→ More stable training, used in GPT-3, LLaMA, Claude",
            color=YELLOW_MED,
        )
        stable_note.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(stable_note), run_time=0.7)
        self.wait(2)
        self.fade_all(norm_title, post_label, pre_label, vs_txt,
                      post_stack, pre_stack, stable_note)

        # ── Closing recap ─────────────────────────────────────────────────────
        recap_title = body_text("The Transformer — Key Ideas", color=WHITE)
        recap_title.to_edge(UP, buff=0.6)
        self.play(Write(recap_title), run_time=0.6)

        recap_items = [
            ("Pipeline",           BLUE_MED,   "Embed → N blocks → output head"),
            ("Block internals",    PURPLE_MED, "LayerNorm → Attention → + → LayerNorm → FFN → +"),
            ("Residual skip",      GREEN_MED,  "Add input back: x = x + Layer(x)"),
            ("Encoder vs Decoder", ORANGE_MED, "BERT reads all; GPT generates left-to-right"),
            ("Layer depth",        RED_MED,    "Lower = syntax; upper = reasoning"),
            ("Pre-norm",           GREY_LIGHT, "Normalize before sublayer for stability"),
        ]

        recap_rows = VGroup()
        for key, col, val in recap_items:
            k = body_text(key + ":", color=col)
            v = label_text(val, color=WHITE)
            v.next_to(k, RIGHT, buff=0.3)
            recap_rows.add(VGroup(k, v))

        recap_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        recap_rows.move_to(ORIGIN + DOWN * 0.2)

        self.play(LaggedStart(*[FadeIn(r) for r in recap_rows], lag_ratio=0.12),
                  run_time=1.8)
        self.wait(2.5)
        self.fade_all(recap_title, recap_rows)
