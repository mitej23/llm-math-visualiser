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
            ("Model said 90%  ->  correct!",  GREEN_MED,  "Very small loss"),
            ("Model said 42%  ->  correct",   YELLOW_MED, "Medium loss"),
            ("Model said 10%  ->  correct",   ORANGE_MED, "Large loss"),
            ("Model said  1%  ->  correct",   RED_MED,    "Huge loss"),
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
            b = rounded_box(1.7, 0.9, fill_color=str(col) + "22",
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
        self.play(LaggedStart(*[Create(a) for a in step_arrows], lag_ratio=0.1),
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
        xl = label_text("Training steps ->", color=GREY_MED)
        xl.next_to(axes, DOWN, buff=0.2)
        yl = label_text("Loss", color=GREY_MED)
        yl.next_to(axes, LEFT, buff=0.2)

        loss_curve = axes.plot(
            lambda x: 4.5 * np.exp(-0.5 * x) + 0.3,
            color=GREEN_MED, stroke_width=2.5,
        )
        self.play(Create(axes), FadeIn(xl), FadeIn(yl), run_time=0.6)
        self.play(Create(loss_curve), run_time=1.2)

        note = label_text(
            "High loss early (random guesses) -> falls as weights improve",
            color=GREY_LIGHT,
        )
        note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(2)
        self.fade_all(curve_title, axes, xl, yl, loss_curve, note)

        # ── 5. Perplexity ──────────────────────────────────────────────────────
        perp_title = body_text("Perplexity = e^(loss) — how confused is the model?",
                               color=WHITE)
        perp_title.to_edge(UP, buff=0.6)
        self.play(Write(perp_title), run_time=0.7)

        perp_data = [
            ("Loss = 0.0",  "Perplexity = 1",    GREEN_MED,   "Perfect — always right"),
            ("Loss = 2.3",  "Perplexity = 10",   BLUE_MED,    "Good — top 10 choices"),
            ("Loss = 4.6",  "Perplexity = 100",  YELLOW_MED,  "Confused — 100 options"),
            ("Loss = 6.9",  "Perplexity = 1000", ORANGE_MED,  "Very confused"),
            ("Loss = 10.8", "Perplexity = 50k",  RED_MED,     "Random model"),
        ]

        perp_rows = VGroup()
        for loss_s, perp_s, col, meaning in perp_data:
            loss_lbl = label_text(loss_s, color=col)
            arr = label_text("->", color=GREY_MED)
            perp_lbl = body_text(perp_s, color=col)
            mean_lbl = label_text(meaning, color=GREY_LIGHT)
            arr.next_to(loss_lbl, RIGHT, buff=0.2)
            perp_lbl.next_to(arr, RIGHT, buff=0.2)
            mean_lbl.next_to(perp_lbl, RIGHT, buff=0.4)
            perp_rows.add(VGroup(loss_lbl, arr, perp_lbl, mean_lbl))

        perp_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        perp_rows.move_to(ORIGIN + DOWN * 0.2)

        self.play(LaggedStart(*[FadeIn(r) for r in perp_rows], lag_ratio=0.2),
                  run_time=1.5)
        perp_note = label_text("Lower perplexity = better model", color=YELLOW_MED)
        perp_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(perp_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(perp_title, perp_rows, perp_note)

        # ── 6. Gradient descent visualisation ─────────────────────────────────
        gd_title = body_text("Gradient descent: rolling a ball to the lowest valley",
                             color=WHITE)
        gd_title.to_edge(UP, buff=0.6)
        self.play(Write(gd_title), run_time=0.7)

        gd_axes = Axes(
            x_range=[-3, 3, 1], y_range=[0, 5, 1],
            x_length=9, y_length=4.5,
            axis_config={"color": GREY_MED, "include_numbers": False},
            tips=False,
        )
        gd_axes.move_to(ORIGIN + DOWN * 0.2)
        xl2 = label_text("Weight value", color=GREY_MED)
        xl2.next_to(gd_axes, DOWN, buff=0.2)
        yl2 = label_text("Loss", color=GREY_MED)
        yl2.next_to(gd_axes, LEFT, buff=0.2)

        # Bowl-shaped loss landscape
        bowl = gd_axes.plot(
            lambda x: 0.5 * x ** 2 + 0.4,
            color=BLUE_MED, stroke_width=2.5,
        )

        # Ball starting high and rolling down
        ball_positions = [-2.5, -1.8, -1.2, -0.7, -0.3, 0.0]
        ball_dots = []
        for xp in ball_positions:
            yp = 0.5 * xp ** 2 + 0.4
            dot = Dot(gd_axes.c2p(xp, yp), color=YELLOW_MED, radius=0.12)
            ball_dots.append(dot)

        step_lbl = label_text("Each step = learning_rate x gradient", color=YELLOW_MED)
        step_lbl.to_edge(DOWN, buff=0.4)

        minimum_lbl = label_text("Minimum (loss = 0.4)", color=GREEN_MED)
        minimum_lbl.next_to(gd_axes.c2p(0, 0.4), UR, buff=0.2)

        self.play(Create(gd_axes), FadeIn(xl2), FadeIn(yl2), run_time=0.6)
        self.play(Create(bowl), FadeIn(minimum_lbl), run_time=0.8)
        self.play(FadeIn(ball_dots[0]), run_time=0.4)
        for i in range(1, len(ball_dots)):
            self.play(
                Transform(ball_dots[0], ball_dots[i]),
                run_time=0.4,
            )
        self.play(FadeIn(step_lbl), run_time=0.5)
        self.wait(1.5)
        self.fade_all(gd_title, gd_axes, xl2, yl2, bowl, ball_dots[0],
                      minimum_lbl, step_lbl)

        # ── 7. Learning rate schedule ──────────────────────────────────────────
        lr_title = body_text("Learning rate schedule: warmup then cosine decay",
                             color=WHITE)
        lr_title.to_edge(UP, buff=0.6)
        self.play(Write(lr_title), run_time=0.7)

        lr_axes = Axes(
            x_range=[0, 10, 2], y_range=[0, 1.2, 0.3],
            x_length=9, y_length=4,
            axis_config={"color": GREY_MED, "include_numbers": False},
            tips=False,
        )
        lr_axes.move_to(ORIGIN + DOWN * 0.2)
        xl3 = label_text("Training steps ->", color=GREY_MED)
        xl3.next_to(lr_axes, DOWN, buff=0.2)
        yl3 = label_text("LR", color=GREY_MED)
        yl3.next_to(lr_axes, LEFT, buff=0.2)

        def lr_schedule(x):
            warmup_end = 1.0
            if x < warmup_end:
                return x / warmup_end
            else:
                t = (x - warmup_end) / (10.0 - warmup_end)
                return 0.05 + 0.5 * (1.0 - 0.05) * (1 + np.cos(np.pi * t))

        lr_curve = lr_axes.plot(lr_schedule, color=ORANGE_MED, stroke_width=2.5,
                                x_range=[0, 10, 0.05])

        warmup_label = label_text("Warmup", color=BLUE_MED)
        warmup_label.next_to(lr_axes.c2p(0.5, 1.1), UP, buff=0.1)

        decay_label = label_text("Cosine decay", color=GREEN_MED)
        decay_label.next_to(lr_axes.c2p(5.5, 0.7), UP, buff=0.1)

        lr_note = label_text(
            "All frontier LLMs use warmup + cosine decay — never fixed learning rate",
            color=GREY_LIGHT,
        )
        lr_note.to_edge(DOWN, buff=0.4)

        self.play(Create(lr_axes), FadeIn(xl3), FadeIn(yl3), run_time=0.6)
        self.play(Create(lr_curve), run_time=1.2)
        self.play(FadeIn(warmup_label), FadeIn(decay_label), run_time=0.5)
        self.play(FadeIn(lr_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(lr_title, lr_axes, xl3, yl3, lr_curve,
                      warmup_label, decay_label, lr_note)

        # ── 8. Overfitting vs underfitting ─────────────────────────────────────
        ov_title = body_text("Overfitting vs Underfitting vs Good Fit", color=WHITE)
        ov_title.to_edge(UP, buff=0.6)
        self.play(Write(ov_title), run_time=0.7)

        ov_axes = Axes(
            x_range=[0, 10, 2], y_range=[0, 5, 1],
            x_length=9, y_length=4,
            axis_config={"color": GREY_MED, "include_numbers": False},
            tips=False,
        )
        ov_axes.move_to(ORIGIN + DOWN * 0.2)
        xl4 = label_text("Training steps ->", color=GREY_MED)
        xl4.next_to(ov_axes, DOWN, buff=0.2)
        yl4 = label_text("Loss", color=GREY_MED)
        yl4.next_to(ov_axes, LEFT, buff=0.2)

        # Train loss: always falling
        train_curve = ov_axes.plot(
            lambda x: 4.2 * np.exp(-0.45 * x) + 0.25,
            color=BLUE_MED, stroke_width=2.5,
        )
        # Val loss: falls then rises (overfitting)
        val_curve = ov_axes.plot(
            lambda x: 4.0 * np.exp(-0.4 * x) + 0.5 + (0.003 * (x - 5) ** 2 if x > 5 else 0),
            color=RED_MED, stroke_width=2.5,
        )

        train_lbl = label_text("Train loss", color=BLUE_MED)
        train_lbl.next_to(ov_axes.c2p(9, 0.3), RIGHT, buff=0.05)
        val_lbl = label_text("Val loss", color=RED_MED)
        val_lbl.next_to(ov_axes.c2p(9, 1.1), RIGHT, buff=0.05)

        good_zone = DashedLine(
            ov_axes.c2p(5, 0), ov_axes.c2p(5, 5),
            color=GREEN_MED, stroke_width=1.5,
        )
        good_zone_lbl = label_text("Overfitting starts here", color=GREEN_MED)
        good_zone_lbl.next_to(ov_axes.c2p(5, 4.5), RIGHT, buff=0.1)

        ov_note = label_text(
            "Good fit: both losses low | Overfit: train low but val rising",
            color=GREY_LIGHT,
        )
        ov_note.to_edge(DOWN, buff=0.4)

        self.play(Create(ov_axes), FadeIn(xl4), FadeIn(yl4), run_time=0.6)
        self.play(Create(train_curve), Create(val_curve), run_time=1.2)
        self.play(FadeIn(train_lbl), FadeIn(val_lbl), run_time=0.5)
        self.play(Create(good_zone), FadeIn(good_zone_lbl), run_time=0.5)
        self.play(FadeIn(ov_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(ov_title, ov_axes, xl4, yl4, train_curve, val_curve,
                      train_lbl, val_lbl, good_zone, good_zone_lbl, ov_note)

        # ── 9. The Adam optimizer ──────────────────────────────────────────────
        adam_title = body_text("Adam optimizer: smarter than plain gradient descent",
                               color=WHITE)
        adam_title.to_edge(UP, buff=0.6)
        self.play(Write(adam_title), run_time=0.7)

        sgd_lbl = body_text("Plain SGD", color=GREY_MED)
        sgd_lbl.move_to(LEFT * 3.5 + UP * 1.5)

        adam_lbl = body_text("Adam", color=GREEN_MED)
        adam_lbl.move_to(RIGHT * 2.5 + UP * 1.5)

        sgd_steps = [
            ("Uses only current gradient", GREY_MED),
            ("Same LR for all weights",    GREY_MED),
            ("Sensitive to gradient scale",GREY_MED),
        ]
        sgd_group = VGroup()
        for txt, col in sgd_steps:
            t = label_text(txt, color=col)
            sgd_group.add(t)
        sgd_group.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        sgd_group.next_to(sgd_lbl, DOWN, buff=0.3)

        adam_steps = [
            ("Tracks momentum (past direction)",       GREEN_MED),
            ("Adaptive LR per parameter",              GREEN_MED),
            ("Handles sparse/dense gradients well",    GREEN_MED),
        ]
        adam_group = VGroup()
        for txt, col in adam_steps:
            t = label_text(txt, color=col)
            adam_group.add(t)
        adam_group.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        adam_group.next_to(adam_lbl, DOWN, buff=0.3)

        divider3 = Line(UP * 2.0, DOWN * 1.8, color=GREY_MED, stroke_width=1)

        adam_note = label_text(
            "Adam = momentum + adaptive rates per weight — standard for all LLMs",
            color=YELLOW_MED,
        )
        adam_note.to_edge(DOWN, buff=0.4)

        self.play(FadeIn(sgd_lbl), FadeIn(adam_lbl), Create(divider3), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(s) for s in sgd_group], lag_ratio=0.2),
                  run_time=0.8)
        self.play(LaggedStart(*[FadeIn(a) for a in adam_group], lag_ratio=0.2),
                  run_time=0.8)
        self.play(FadeIn(adam_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(adam_title, sgd_lbl, adam_lbl, divider3,
                      sgd_group, adam_group, adam_note)

        # ── 10. Data preprocessing pipeline ───────────────────────────────────
        pipe_title = body_text("Data pipeline: raw text -> training batches", color=WHITE)
        pipe_title.to_edge(UP, buff=0.6)
        self.play(Write(pipe_title), run_time=0.7)

        pipeline_steps = [
            ("Raw text\n(web, books)", GREY_LIGHT),
            ("Tokenize\n(BPE)", BLUE_MED),
            ("Pack into\n2048-token\nchunks", GREEN_MED),
            ("Shuffle\nrandomly", ORANGE_MED),
            ("Mini-batches\n(512 chunks)", PURPLE_MED),
            ("Train\nstep", YELLOW_MED),
        ]

        pipe_boxes = VGroup()
        for lbl, col in pipeline_steps:
            b = rounded_box(1.55, 1.0,
                            fill_color=str(col) + "22",
                            stroke_color=col, label=lbl, label_color=col)
            pipe_boxes.add(b)

        pipe_boxes.arrange(RIGHT, buff=0.25)
        pipe_boxes.scale_to_fit_width(13.5)
        pipe_boxes.move_to(ORIGIN + DOWN * 0.1)

        pipe_arrows = VGroup(*[
            Arrow(pipe_boxes[i].get_right(), pipe_boxes[i + 1].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.2)
            for i in range(len(pipe_boxes) - 1)
        ])

        pipe_note = label_text(
            "Packing short docs end-to-end fills 2048-token windows efficiently",
            color=GREY_LIGHT,
        )
        pipe_note.to_edge(DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(b) for b in pipe_boxes], lag_ratio=0.12),
                  run_time=1.5)
        self.play(LaggedStart(*[Create(a) for a in pipe_arrows], lag_ratio=0.1),
                  run_time=0.8)
        self.play(FadeIn(pipe_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(pipe_title, pipe_boxes, pipe_arrows, pipe_note)
