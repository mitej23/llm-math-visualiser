"""
Scene 17 — Training Loop & Loss Functions
Run: manim -pql 17_training_loop.py TrainingLoopScene
"""

from manim import *
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class TrainingLoopScene(LLMScene):
    def construct(self):
        title = self.show_title("Training Loop & Loss", "How a Model Actually Learns")
        self.wait(0.5)
        self.fade_all(title)

        # ── 1. The training objective: predict next token ─────────────────────
        obj_title = body_text("The task: predict the next token", color=WHITE)
        obj_title.to_edge(UP, buff=0.6)
        self.play(Write(obj_title), run_time=0.7)

        prompt_boxes = VGroup()
        prompt_tokens = ["The", "cat", "sat", "on", "the"]
        for tok in prompt_tokens:
            b = rounded_box(0.9, 0.55, fill_color=BLUE_DARK,
                            stroke_color=BLUE_MED, label=tok, label_color=BLUE_MED)
            prompt_boxes.add(b)

        prompt_boxes.arrange(RIGHT, buff=0.18)
        prompt_boxes.shift(UP * 0.5)

        blank_box = rounded_box(0.9, 0.55, fill_color=GREY_DARK,
                                stroke_color=YELLOW_MED, label="___", label_color=YELLOW_MED)
        blank_box.next_to(prompt_boxes, RIGHT, buff=0.18)

        self.play(LaggedStart(*[FadeIn(b) for b in prompt_boxes], lag_ratio=0.1),
                  run_time=0.8)
        self.play(FadeIn(blank_box), run_time=0.4)

        # Model's guesses
        guesses = [("mat", 42, GREEN_MED), ("floor", 18, BLUE_MED),
                   ("chair", 11, GREY_LIGHT), ("roof", 3, GREY_MED)]
        guess_group = VGroup()
        for word, pct, col in guesses:
            bar = Rectangle(width=pct * 0.04, height=0.4,
                            fill_color=col, fill_opacity=0.8,
                            stroke_color=col, stroke_width=1)
            lbl = label_text(f"{word} {pct}%", color=col)
            lbl.next_to(bar, RIGHT, buff=0.1)
            row = VGroup(bar, lbl)
            guess_group.add(row)

        guess_group.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        guess_group.next_to(blank_box, DOWN, buff=0.5)

        self.play(LaggedStart(*[FadeIn(g) for g in guess_group], lag_ratio=0.15),
                  run_time=1.0)
        self.wait(0.8)
        self.fade_all(obj_title, prompt_boxes, blank_box, guess_group)

        # ── 2. Cross-entropy loss ─────────────────────────────────────────────
        loss_title = body_text("Cross-Entropy Loss — penalty for wrong confidence",
                               color=WHITE)
        loss_title.to_edge(UP, buff=0.6)
        self.play(Write(loss_title), run_time=0.7)

        cases = [
            ("Model said 90%  →  correct!",  GREEN_MED,  "Very small loss 😊"),
            ("Model said 42%  →  correct",   YELLOW_MED, "Medium loss 😐"),
            ("Model said 10%  →  correct",   ORANGE_MED, "Large loss 😬"),
            ("Model said  1%  →  correct",   RED_MED,    "Huge loss 😱"),
        ]
        case_rows = VGroup()
        for pred, col, result in cases:
            p = body_text(pred, color=col)
            r = label_text(result, color=col)
            r.next_to(p, RIGHT, buff=0.5)
            case_rows.add(VGroup(p, r))

        case_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        case_rows.move_to(ORIGIN + DOWN * 0.2)
        box = SurroundingRectangle(case_rows, color=GREY_MED, buff=0.3, corner_radius=0.12)

        self.play(Create(box), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r) for r in case_rows], lag_ratio=0.2),
                  run_time=1.2)
        self.wait(1.2)
        self.fade_all(loss_title, case_rows, box)

        # ── 3. The training loop ──────────────────────────────────────────────
        loop_title = body_text("The Full Training Loop", color=WHITE)
        loop_title.to_edge(UP, buff=0.6)
        self.play(Write(loop_title), run_time=0.6)

        steps = [
            ("1. Sample\nmini-batch",   BLUE_MED),
            ("2. Forward\npass",        GREEN_MED),
            ("3. Compute\nloss",        ORANGE_MED),
            ("4. Backprop\ngradients",  RED_MED),
            ("5. Update\nweights",      PURPLE_MED),
        ]
        step_boxes = VGroup()
        for lbl, col in steps:
            b = rounded_box(1.7, 0.9, fill_color=col + "22",
                            stroke_color=col, label=lbl, label_color=col)
            step_boxes.add(b)

        step_boxes.arrange(RIGHT, buff=0.4)
        step_boxes.scale_to_fit_width(13)
        step_boxes.move_to(ORIGIN)

        step_arrows = VGroup(*[
            Arrow(step_boxes[i].get_right(), step_boxes[i + 1].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.18)
            for i in range(len(step_boxes) - 1)
        ])

        # Loop-back arrow
        loop_back = CurvedArrow(
            step_boxes[-1].get_bottom() + DOWN * 0.1,
            step_boxes[0].get_bottom() + DOWN * 0.1,
            angle=TAU / 5, color=GREY_MED, stroke_width=1.5,
        )
        loop_lbl = label_text("repeat millions of times", color=GREY_LIGHT)
        loop_lbl.next_to(loop_back, DOWN, buff=0.1)

        self.play(LaggedStart(*[FadeIn(b) for b in step_boxes], lag_ratio=0.15),
                  run_time=1.4)
        self.play(LaggedStart(*[GrowArrow(a) for a in step_arrows], lag_ratio=0.1),
                  run_time=0.8)
        self.play(Create(loop_back), FadeIn(loop_lbl), run_time=0.7)
        self.wait(1.2)
        self.fade_all(loop_title, step_boxes, step_arrows, loop_back, loop_lbl)

        # ── 4. Loss curve ──────────────────────────────────────────────────────
        curve_title = body_text("Training loss should fall over time:", color=WHITE)
        curve_title.to_edge(UP, buff=0.6)
        self.play(Write(curve_title), run_time=0.6)

        axes = Axes(
            x_range=[0, 10, 2], y_range=[0, 5, 1],
            x_length=8, y_length=4,
            axis_config={"color": GREY_MED, "include_numbers": False},
            tips=False,
        )
        axes.shift(DOWN * 0.3)
        xl = label_text("Training steps →", color=GREY_MED)
        xl.next_to(axes, DOWN, buff=0.2)
        yl = label_text("Loss ↑", color=GREY_MED)
        yl.next_to(axes, LEFT, buff=0.2)

        loss_curve = axes.plot(
            lambda x: 4.5 * np.exp(-0.5 * x) + 0.3,
            color=GREEN_MED, stroke_width=2.5,
        )
        self.play(Create(axes), FadeIn(xl), FadeIn(yl), run_time=0.6)
        self.play(Create(loss_curve), run_time=1.2)

        note = label_text(
            "High loss early (random guesses) → falls as weights improve",
            color=GREY_LIGHT,
        )
        note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(2)
