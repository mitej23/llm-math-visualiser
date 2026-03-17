"""
Scene 11 — Layer Normalization
Run: manim -pql 11_layer_normalization.py LayerNormScene
"""

from manim import *
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class LayerNormScene(LLMScene):
    def construct(self):
        title = self.show_title("Layer Normalization", "Keeping Values Stable")
        self.wait(0.5)
        self.fade_all(title)

        # ── 1. The problem: exploding / vanishing values ───────────────────────
        problem_title = body_text("Without normalisation — values explode or collapse:",
                                  color=WHITE)
        problem_title.to_edge(UP, buff=0.6)
        self.play(Write(problem_title), run_time=0.7)

        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[-10, 10000, 2000],
            x_length=6, y_length=4,
            axis_config={"color": GREY_MED, "include_numbers": False},
            tips=False,
        )
        axes.shift(DOWN * 0.2)
        x_lbl = label_text("Layer depth ->", color=GREY_MED)
        x_lbl.next_to(axes, DOWN, buff=0.2)

        explode_curve = axes.plot(lambda x: np.exp(x * 0.8) - 1,
                                  color=RED_MED, stroke_width=2.5)
        vanish_curve  = axes.plot(lambda x: 9000 * np.exp(-x * 0.9),
                                  color=BLUE_MED, stroke_width=2.5)

        exp_lbl = label_text("Exploding activations", color=RED_MED)
        exp_lbl.next_to(axes, RIGHT, buff=0.3).shift(UP * 0.5)
        van_lbl = label_text("Vanishing activations", color=BLUE_MED)
        van_lbl.next_to(axes, RIGHT, buff=0.3).shift(DOWN * 0.3)

        self.play(Create(axes), FadeIn(x_lbl), run_time=0.6)
        self.play(Create(explode_curve), FadeIn(exp_lbl), run_time=0.7)
        self.play(Create(vanish_curve), FadeIn(van_lbl), run_time=0.7)
        self.wait(1)
        self.fade_all(problem_title, axes, x_lbl,
                      explode_curve, vanish_curve, exp_lbl, van_lbl)

        # ── 2. What normalisation does — visual ───────────────────────────────
        norm_title = body_text("LayerNorm: rescale each token's vector to mean=0, std=1",
                               color=WHITE)
        norm_title.to_edge(UP, buff=0.6)
        self.play(Write(norm_title), run_time=0.7)

        # Before
        before_vals = [8.5, -12.3, 45.1, 3.2, -0.8, 22.4]
        after_vals  = [(v - np.mean(before_vals)) / np.std(before_vals)
                       for v in before_vals]

        before_bars = VGroup()
        after_bars  = VGroup()
        bar_w = 0.5
        for i, (bv, av) in enumerate(zip(before_vals, after_vals)):
            x = i * (bar_w + 0.15)
            # Before bar
            b_height = min(abs(bv) * 0.05, 2.5) * np.sign(bv)
            b_col = RED_MED if bv > 0 else BLUE_MED
            b_bar = Rectangle(width=bar_w, height=abs(b_height),
                               fill_color=b_col, fill_opacity=0.7,
                               stroke_color=b_col, stroke_width=1)
            b_bar.shift(RIGHT * x + UP * (b_height / 2 if bv > 0 else b_height / 2))
            before_bars.add(b_bar)
            # After bar
            a_height = av * 0.5
            a_col = GREEN_MED if av > 0 else ORANGE_MED
            a_bar = Rectangle(width=bar_w, height=abs(a_height),
                               fill_color=a_col, fill_opacity=0.7,
                               stroke_color=a_col, stroke_width=1)
            a_bar.shift(RIGHT * x + UP * (a_height / 2 if av > 0 else a_height / 2))
            after_bars.add(a_bar)

        before_bars.move_to(LEFT * 3)
        after_bars.move_to(RIGHT * 3)

        b_label = label_text("Before LayerNorm\n(wild range)", color=RED_MED)
        b_label.next_to(before_bars, DOWN, buff=0.3)
        a_label = label_text("After LayerNorm\n(mean=0, std=1)", color=GREEN_MED)
        a_label.next_to(after_bars, DOWN, buff=0.3)

        zero_line_b = DashedLine(before_bars.get_left() + LEFT * 0.2,
                                 before_bars.get_right() + RIGHT * 0.2,
                                 color=GREY_MED, stroke_width=1)
        zero_line_b.move_to(before_bars.get_center())
        zero_line_a = DashedLine(after_bars.get_left() + LEFT * 0.2,
                                 after_bars.get_right() + RIGHT * 0.2,
                                 color=GREY_MED, stroke_width=1)
        zero_line_a.move_to(after_bars.get_center())

        arrow_transform = Arrow(LEFT * 0.8, RIGHT * 0.8, color=WHITE,
                                stroke_width=2, buff=0.1)
        norm_txt = label_text("LayerNorm", color=YELLOW_MED)
        norm_txt.next_to(arrow_transform, UP, buff=0.1)

        self.play(FadeIn(before_bars), FadeIn(zero_line_b), FadeIn(b_label),
                  run_time=0.8)
        self.play(Create(arrow_transform), Write(norm_txt), run_time=0.5)
        self.play(FadeIn(after_bars), FadeIn(zero_line_a), FadeIn(a_label),
                  run_time=0.8)
        self.wait(1.2)
        self.fade_all(norm_title, before_bars, zero_line_b, b_label,
                      arrow_transform, norm_txt, after_bars, zero_line_a, a_label)

        # ── 3. Pre-norm vs post-norm and RMSNorm ──────────────────────────────
        comp_title = body_text("Modern models use Pre-Norm with RMSNorm:", color=WHITE)
        comp_title.to_edge(UP, buff=0.6)
        self.play(Write(comp_title), run_time=0.7)

        rows_data = [
            ("Post-Norm (original)", GREY_LIGHT,
             "Norm AFTER sublayer + residual",    "Original Transformer"),
            ("Pre-Norm (modern)",    YELLOW_MED,
             "Norm BEFORE sublayer",              "GPT, LLaMA, most LLMs"),
            ("RMSNorm",              GREEN_MED,
             "Simplified: no mean subtraction",   "LLaMA, Mistral, Gemma"),
        ]

        rows = VGroup()
        for name, col, desc, used in rows_data:
            n = body_text(name, color=col)
            d = label_text(desc, color=WHITE)
            u = label_text(f"-> {used}", color=GREY_LIGHT)
            row = VGroup(n, d, u)
            row.arrange(RIGHT, buff=0.45)
            rows.add(row)

        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        rows.move_to(ORIGIN + DOWN * 0.2)
        box = SurroundingRectangle(rows, color=BLUE_MED, buff=0.35, corner_radius=0.15)

        self.play(Create(box), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.2),
                  run_time=1.2)

        rmsnorm_note = label_text(
            "RMSNorm skips mean subtraction — ~20% faster, same quality.",
            color=GREY_LIGHT,
        )
        rmsnorm_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(rmsnorm_note), run_time=0.6)
        self.wait(2)
        self.fade_all(comp_title, rows, box, rmsnorm_note)

        # ── 4. Why activations explode — the compounding effect ───────────────
        compound_title = body_text("Compounding: x1.1 per layer = disaster at 100 layers:",
                                   color=WHITE)
        compound_title.to_edge(UP, buff=0.6)
        self.play(Write(compound_title), run_time=0.7)

        layer_vals = [1.1 ** i for i in range(12)]
        layer_nums = list(range(12))

        compound_axes = Axes(
            x_range=[0, 11, 2],
            y_range=[0, 4, 1],
            x_length=8, y_length=3.5,
            axis_config={"color": GREY_MED, "include_numbers": False},
            tips=False,
        )
        compound_axes.shift(DOWN * 0.5)
        x_label = label_text("Layer number", color=GREY_MED)
        x_label.next_to(compound_axes, DOWN, buff=0.2)
        y_label = label_text("Activation scale", color=GREY_MED)
        y_label.next_to(compound_axes, LEFT, buff=0.2)

        compound_curve = compound_axes.plot(lambda x: 1.1 ** x,
                                            color=RED_MED, stroke_width=2.5)

        danger_lbl = label_text("x1.1 per layer -> 3.1x after just 12 layers!", color=RED_MED)
        danger_lbl.next_to(compound_axes, RIGHT, buff=0.2).shift(UP * 0.5)

        self.play(Create(compound_axes), FadeIn(x_label), FadeIn(y_label), run_time=0.7)
        self.play(Create(compound_curve), run_time=1.0)
        self.play(FadeIn(danger_lbl), run_time=0.5)

        solution_lbl = label_text(
            "LayerNorm resets scale to ~1.0 after each layer",
            color=GREEN_MED,
        )
        solution_lbl.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(solution_lbl), run_time=0.6)
        self.wait(1.5)
        self.fade_all(compound_title, compound_axes, x_label, y_label,
                      compound_curve, danger_lbl, solution_lbl)

        # ── 5. Normalisation formula step by step ─────────────────────────────
        formula_title = body_text("The LayerNorm formula — step by step:", color=WHITE)
        formula_title.to_edge(UP, buff=0.6)
        self.play(Write(formula_title), run_time=0.7)

        raw_vals = [3, 7, 2, 9, 4]
        mean_val = 5.0
        centered = [-2, 2, -3, 4, -1]
        std_val = 2.61
        normed = [-0.77, 0.77, -1.15, 1.53, -0.38]

        step_groups = []

        # Step 1: raw values
        raw_boxes = VGroup()
        for v in raw_vals:
            col = GREEN_MED if v > mean_val else ORANGE_MED
            b = rounded_box(0.7, 0.55,
                            fill_color=str(col) + "33",
                            stroke_color=col,
                            label=str(v), label_color=col)
            raw_boxes.add(b)
        raw_boxes.arrange(RIGHT, buff=0.15)
        raw_label = label_text("Input: [3, 7, 2, 9, 4]  mean = 5.0", color=WHITE)
        raw_label.next_to(raw_boxes, DOWN, buff=0.2)
        raw_group = VGroup(raw_boxes, raw_label)
        raw_group.shift(UP * 1.5)
        step_groups.append(raw_group)

        # Step 2: centered
        cent_boxes = VGroup()
        for v in centered:
            col = GREEN_MED if v > 0 else RED_MED
            b = rounded_box(0.7, 0.55,
                            fill_color=str(col) + "33",
                            stroke_color=col,
                            label=str(v), label_color=col)
            cent_boxes.add(b)
        cent_boxes.arrange(RIGHT, buff=0.15)
        cent_label = label_text("Subtract mean: [-2, 2, -3, 4, -1]", color=WHITE)
        cent_label.next_to(cent_boxes, DOWN, buff=0.2)
        cent_group = VGroup(cent_boxes, cent_label)
        cent_group.shift(DOWN * 0.0)
        step_groups.append(cent_group)

        # Step 3: normalized
        norm_boxes = VGroup()
        for v in normed:
            col = BLUE_LIGHT if v > 0 else PURPLE_MED
            b = rounded_box(0.85, 0.55,
                            fill_color=str(col) + "33",
                            stroke_color=col,
                            label=f"{v:.2f}", label_color=col)
            norm_boxes.add(b)
        norm_boxes.arrange(RIGHT, buff=0.15)
        norm_label = label_text("Divide by std (2.61): values near +/-1", color=GREEN_MED)
        norm_label.next_to(norm_boxes, DOWN, buff=0.2)
        norm_group = VGroup(norm_boxes, norm_label)
        norm_group.shift(DOWN * 1.6)
        step_groups.append(norm_group)

        step_arr1 = Arrow(raw_group.get_bottom(), cent_group.get_top(),
                          color=YELLOW_MED, buff=0.05, stroke_width=1.5,
                          max_tip_length_to_length_ratio=0.2)
        step_arr2 = Arrow(cent_group.get_bottom(), norm_group.get_top(),
                          color=YELLOW_MED, buff=0.05, stroke_width=1.5,
                          max_tip_length_to_length_ratio=0.2)

        self.play(FadeIn(raw_group), run_time=0.6)
        self.play(Create(step_arr1), run_time=0.3)
        self.play(FadeIn(cent_group), run_time=0.6)
        self.play(Create(step_arr2), run_time=0.3)
        self.play(FadeIn(norm_group), run_time=0.6)
        self.wait(1.5)
        self.fade_all(formula_title, raw_group, cent_group, norm_group,
                      step_arr1, step_arr2)

        # ── 6. Batch Norm vs Layer Norm diagram ───────────────────────────────
        bn_ln_title = body_text("Batch Norm vs Layer Norm — what gets normalised:", color=WHITE)
        bn_ln_title.to_edge(UP, buff=0.6)
        self.play(Write(bn_ln_title), run_time=0.7)

        rows_bn = 4
        cols_bn = 5
        cell_size = 0.55

        def make_grid(highlight_rows=None, highlight_cols=None,
                      fill_col=BLUE_MED, highlight_col=GREEN_MED):
            grid = VGroup()
            for r in range(rows_bn):
                for c in range(cols_bn):
                    is_hl = False
                    if highlight_rows is not None and r in highlight_rows:
                        is_hl = True
                    if highlight_cols is not None and c in highlight_cols:
                        is_hl = True
                    color = highlight_col if is_hl else fill_col
                    opacity = 0.7 if is_hl else 0.2
                    cell = Square(side_length=cell_size,
                                  fill_color=color, fill_opacity=opacity,
                                  stroke_color=GREY_MED, stroke_width=1)
                    cell.move_to([c * cell_size, -r * cell_size, 0])
                    grid.add(cell)
            return grid

        bn_grid = make_grid(highlight_cols=[2], fill_col=BLUE_MED, highlight_col=GREEN_MED)
        ln_grid = make_grid(highlight_rows=[1], fill_col=BLUE_MED, highlight_col=ORANGE_MED)

        bn_grid.shift(LEFT * 3.5 + UP * 0.5)
        ln_grid.shift(RIGHT * 1.5 + UP * 0.5)

        bn_heading = body_text("Batch Norm", color=GREEN_MED)
        bn_heading.next_to(bn_grid, UP, buff=0.2)
        bn_note = label_text("Normalise down\nthe BATCH dimension\n(across examples)", color=GREEN_MED)
        bn_note.next_to(bn_grid, DOWN, buff=0.2)

        ln_heading = body_text("Layer Norm", color=ORANGE_MED)
        ln_heading.next_to(ln_grid, UP, buff=0.2)
        ln_note = label_text("Normalise across\nFEATURES dimension\n(within one example)", color=ORANGE_MED)
        ln_note.next_to(ln_grid, DOWN, buff=0.2)

        row_lbl = label_text("Batch examples ->", color=GREY_MED)
        row_lbl.next_to(bn_grid, LEFT, buff=0.2)

        self.play(FadeIn(bn_grid), FadeIn(ln_grid), run_time=0.6)
        self.play(FadeIn(bn_heading), FadeIn(ln_heading), run_time=0.5)
        self.play(FadeIn(bn_note), FadeIn(ln_note), FadeIn(row_lbl), run_time=0.7)
        self.wait(2)
        self.fade_all(bn_ln_title, bn_grid, ln_grid, bn_heading, ln_heading,
                      bn_note, ln_note, row_lbl)

        # ── 7. Gamma and beta — learnable scale and shift ─────────────────────
        gamma_title = body_text("Learnable gamma and beta restore useful scale:", color=WHITE)
        gamma_title.to_edge(UP, buff=0.6)
        self.play(Write(gamma_title), run_time=0.7)

        pipeline = [
            ("Raw input", GREY_LIGHT),
            ("Subtract mean,\ndivide by std", YELLOW_MED),
            ("Multiply by\ngamma", GREEN_MED),
            ("Add beta", BLUE_LIGHT),
            ("Output", WHITE),
        ]

        pipe_boxes = VGroup()
        for lbl, col in pipeline:
            b = rounded_box(2.0, 0.9,
                            fill_color=str(col) + "22",
                            stroke_color=col,
                            label=lbl, label_color=col)
            pipe_boxes.add(b)

        pipe_boxes.arrange(RIGHT, buff=0.5)
        pipe_boxes.scale_to_fit_width(13)
        pipe_boxes.move_to(ORIGIN + UP * 0.3)

        pipe_arrows = VGroup()
        for i in range(len(pipe_boxes) - 1):
            arr = Arrow(pipe_boxes[i].get_right(), pipe_boxes[i + 1].get_left(),
                        color=GREY_MED, buff=0.05, stroke_width=1.5,
                        max_tip_length_to_length_ratio=0.2)
            pipe_arrows.add(arr)

        gamma_note = label_text(
            "gamma (scale) and beta (shift) are learned during training.\n"
            "Initially gamma=1, beta=0 — then the model adjusts them freely.",
            color=GREY_LIGHT,
        )
        gamma_note.to_edge(DOWN, buff=0.35)

        self.play(LaggedStart(*[FadeIn(b) for b in pipe_boxes], lag_ratio=0.15),
                  run_time=1.2)
        self.play(LaggedStart(*[Create(a) for a in pipe_arrows], lag_ratio=0.15),
                  run_time=0.8)
        self.play(FadeIn(gamma_note), run_time=0.6)
        self.wait(2)
        self.fade_all(gamma_title, pipe_boxes, pipe_arrows, gamma_note)

        # ── 8. RMSNorm — the simplified version ───────────────────────────────
        rms_title = body_text("RMSNorm — faster, simpler, equally good:", color=WHITE)
        rms_title.to_edge(UP, buff=0.6)
        self.play(Write(rms_title), run_time=0.7)

        ln_steps = VGroup(
            label_text("LayerNorm steps:", color=YELLOW_MED),
            label_text("1. Compute mean", color=WHITE),
            label_text("2. Subtract mean", color=WHITE),
            label_text("3. Compute variance", color=WHITE),
            label_text("4. Divide by std", color=WHITE),
            label_text("5. Apply gamma + beta", color=WHITE),
        )
        ln_steps.arrange(DOWN, aligned_edge=LEFT, buff=0.18)

        rms_steps = VGroup(
            label_text("RMSNorm steps:", color=GREEN_MED),
            label_text("1. Compute RMS (root mean square)", color=WHITE),
            label_text("2. Divide by RMS", color=WHITE),
            label_text("3. Apply gamma only", color=WHITE),
            label_text("(no mean subtraction, no beta)", color=GREY_MED),
        )
        rms_steps.arrange(DOWN, aligned_edge=LEFT, buff=0.18)

        ln_box = SurroundingRectangle(ln_steps, color=YELLOW_MED,
                                      buff=0.25, corner_radius=0.12)
        rms_box = SurroundingRectangle(rms_steps, color=GREEN_MED,
                                       buff=0.25, corner_radius=0.12)

        ln_group = VGroup(ln_box, ln_steps)
        rms_group = VGroup(rms_box, rms_steps)

        ln_group.shift(LEFT * 3.2)
        rms_group.shift(RIGHT * 3.0)

        speedup_lbl = label_text(
            "RMSNorm is ~20% faster and achieves equal quality in practice",
            color=GREEN_MED,
        )
        speedup_lbl.to_edge(DOWN, buff=0.4)

        used_lbl = label_text(
            "Used in: LLaMA, LLaMA 2/3, Mistral, Gemma",
            color=GREY_LIGHT,
        )
        used_lbl.next_to(speedup_lbl, UP, buff=0.1)

        self.play(FadeIn(ln_group), FadeIn(rms_group), run_time=0.8)
        self.play(FadeIn(speedup_lbl), FadeIn(used_lbl), run_time=0.6)
        self.wait(2)
        self.fade_all(rms_title, ln_group, rms_group, speedup_lbl, used_lbl)
