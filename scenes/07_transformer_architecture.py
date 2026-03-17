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
                              fill_color=col + "22",
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
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.1),
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
                            fill_color=col + "22",
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
        self.play(LaggedStart(*[GrowArrow(a) for a in v_arrows], lag_ratio=0.08),
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
