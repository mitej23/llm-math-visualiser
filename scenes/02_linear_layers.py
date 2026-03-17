"""
Scene 02 — Linear Layers & Matrix Multiplication
Run: manim -pql 02_linear_layers.py LinearLayersScene
"""

from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class LinearLayersScene(LLMScene):
    def construct(self):
        # ── Title ─────────────────────────────────────────────────────────────
        title = self.show_title("Linear Layers", "Matrix Multiplication — the Core Operation")
        self.wait(0.5)

        # ── 1. Vector × Matrix = Vector ───────────────────────────────────────
        input_vec_vals = [2.0, -1.0, 3.0]
        weight_rows = [
            [0.5, -0.3,  0.8],
            [0.1,  0.9, -0.2],
            [0.4, -0.1,  0.6],
        ]
        output_vec_vals = [
            sum(input_vec_vals[j] * weight_rows[i][j] for j in range(3))
            for i in range(3)
        ]

        # Input vector
        in_label = label_text("Input\nvector x", color=GREEN_LIGHT)
        in_vec = make_vector_display(input_vec_vals, color=GREEN_LIGHT)
        in_group = VGroup(in_label, in_vec)
        in_label.next_to(in_vec, UP, buff=0.2)
        in_group.to_edge(LEFT, buff=0.8)

        # Weight matrix
        def make_matrix(rows, color=BLUE_LIGHT):
            entries = VGroup()
            for r in rows:
                row_grp = VGroup(*[
                    Text(f"{v:.1f}", color=color, font_size=18)
                    for v in r
                ])
                row_grp.arrange(RIGHT, buff=0.35)
                entries.add(row_grp)
            entries.arrange(DOWN, buff=0.25)
            bl = Text("[", color=WHITE, font_size=44)
            br = Text("]", color=WHITE, font_size=44)
            bl.next_to(entries, LEFT, buff=0.08)
            br.next_to(entries, RIGHT, buff=0.08)
            return VGroup(bl, entries, br)

        W = make_matrix(weight_rows)
        w_label = label_text("Weight\nmatrix W", color=BLUE_LIGHT)
        w_label.next_to(W, UP, buff=0.2)
        w_group = VGroup(w_label, W)
        w_group.move_to(ORIGIN)

        # Multiply sign
        mult_sign = Text("×", color=WHITE, font_size=36)
        mult_sign.between_mobjects = True  # position manually
        mult_sign.move_to(
            (in_group.get_right() + w_group.get_left()) / 2
        )

        # Output vector
        out_vec = make_vector_display(output_vec_vals, color=ORANGE_MED)
        out_label = label_text("Output\nvector y", color=ORANGE_MED)
        out_label.next_to(out_vec, UP, buff=0.2)
        out_group = VGroup(out_label, out_vec)
        out_group.to_edge(RIGHT, buff=0.8)

        equals_sign = Text("=", color=WHITE, font_size=36)
        equals_sign.move_to(
            (w_group.get_right() + out_group.get_left()) / 2
        )

        self.play(FadeIn(in_group), run_time=0.7)
        self.play(Write(mult_sign), FadeIn(w_group), run_time=0.8)
        self.wait(0.5)
        self.play(Write(equals_sign), run_time=0.3)

        # Animate each output value computing from a row
        out_vec_entries = out_vec[1]  # the VGroup of DecimalNumbers
        for i, row in enumerate(weight_rows):
            # Highlight row i in W
            row_rect = SurroundingRectangle(W[1][i], color=YELLOW_MED,
                                            buff=0.05, stroke_width=2)
            self.play(Create(row_rect), run_time=0.3)
            self.play(FadeIn(out_vec_entries[i]), run_time=0.5)
            self.play(FadeOut(row_rect), run_time=0.2)

        # Show brackets after all entries
        self.play(FadeIn(out_vec[0]), FadeIn(out_vec[2]),
                  FadeIn(out_label), run_time=0.5)
        self.wait(1)

        # ── 2. Analogy label ──────────────────────────────────────────────────
        analogy = body_text(
            "Like a currency exchange: input euros → exchange rates → output dollars",
            color=GREY_LIGHT
        )
        analogy.to_edge(DOWN, buff=0.6)
        self.play(Write(analogy), run_time=1)
        self.wait(1.2)

        # ── 3. Bias addition ──────────────────────────────────────────────────
        self.fade_all(analogy)

        bias_label = body_text("+ Bias  (a constant offset added to every output)",
                               color=ORANGE_DARK)
        bias_label.to_edge(DOWN, buff=0.6)
        plus_b = Text("+ b", color=ORANGE_DARK, font_size=36)
        plus_b.next_to(out_group, RIGHT, buff=0.3)

        self.play(Write(plus_b), Write(bias_label), run_time=0.9)
        self.wait(1.2)

        # ── 4. Dimension transformation diagram ───────────────────────────────
        self.fade_all(in_group, mult_sign, w_group, equals_sign,
                      out_group, plus_b, bias_label, title)

        dim_title = body_text("Linear layers change vector dimensions", color=WHITE)
        dim_title.to_edge(UP, buff=0.7)
        self.play(Write(dim_title), run_time=0.7)

        examples = [
            ("512", "2048", "Expand  (think wide)"),
            ("2048", "512",  "Compress  (distil)"),
            ("512", "32000", "Project to vocabulary"),
        ]
        arrows_group = VGroup()
        for idx, (d_in, d_out, lbl) in enumerate(examples):
            box_in  = rounded_box(1.2, 0.55, fill_color=BLUE_DARK,
                                  stroke_color=BLUE_MED, label=d_in)
            box_out = rounded_box(1.2, 0.55, fill_color=GREEN_DARK,
                                  stroke_color=GREEN_MED, label=d_out)
            arr = Arrow(box_in.get_right(), box_out.get_left(),
                        color=WHITE, buff=0.1, stroke_width=2)
            lbl_txt = label_text(lbl, color=GREY_LIGHT)
            lbl_txt.next_to(arr, UP, buff=0.1)
            row = VGroup(box_in, arr, lbl_txt, box_out)
            row.arrange(RIGHT, buff=0.25)
            arrows_group.add(row)

        arrows_group.arrange(DOWN, buff=0.5)
        arrows_group.move_to(ORIGIN)

        self.play(LaggedStart(*[FadeIn(r) for r in arrows_group], lag_ratio=0.3),
                  run_time=1.5)
        self.wait(2)

        # ── 5. Why matrices? — processing a batch ─────────────────────────────
        self.fade_all(dim_title, arrows_group)

        batch_title = body_text("Why matrices? Process many inputs at once!", color=WHITE)
        batch_title.to_edge(UP, buff=0.7)
        self.play(Write(batch_title), run_time=0.8)

        # Show 3 input rows (batch of 3)
        batch_rows = [
            [1.0, 0.5, -0.3],
            [0.2, 1.1,  0.8],
            [-0.4, 0.3, 0.9],
        ]
        row_colors = [GREEN_LIGHT, BLUE_LIGHT, PURPLE_MED]
        batch_display = VGroup()
        row_labels_group = VGroup()
        for i, (row, col) in enumerate(zip(batch_rows, row_colors)):
            row_grp = VGroup(*[
                Text(f"{v:.1f}", color=col, font_size=20) for v in row
            ])
            row_grp.arrange(RIGHT, buff=0.4)
            lbl = label_text(f"token {i+1}", color=col)
            lbl.next_to(row_grp, LEFT, buff=0.3)
            batch_display.add(row_grp)
            row_labels_group.add(lbl)

        batch_display.arrange(DOWN, buff=0.3)
        batch_display.to_edge(LEFT, buff=1.5)
        for i, lbl in enumerate(row_labels_group):
            lbl.next_to(batch_display[i], LEFT, buff=0.3)

        bl = Text("[", color=WHITE, font_size=52)
        br = Text("]", color=WHITE, font_size=52)
        bl.next_to(batch_display, LEFT, buff=0.05)
        br.next_to(batch_display, RIGHT, buff=0.05)
        batch_bracket = VGroup(bl, batch_display, br)

        batch_size_lbl = label_text("Batch: 3 tokens processed\nsimultaneously", color=YELLOW_MED)
        batch_size_lbl.next_to(batch_bracket, DOWN, buff=0.3)

        arr_to_W = Arrow(batch_bracket.get_right(), batch_bracket.get_right() + RIGHT * 1.0,
                         color=WHITE, buff=0.1, stroke_width=2)
        W_box = rounded_box(1.2, 1.0, fill_color=BLUE_DARK,
                            stroke_color=BLUE_MED, label="W\n3×4")
        W_box.next_to(arr_to_W, RIGHT, buff=0.1)
        arr_out = Arrow(W_box.get_right(), W_box.get_right() + RIGHT * 0.8,
                        color=ORANGE_MED, buff=0.05, stroke_width=2)
        out_lbl = label_text("3 outputs\n(in parallel)", color=ORANGE_MED)
        out_lbl.next_to(arr_out, RIGHT, buff=0.1)

        self.play(FadeIn(batch_bracket), FadeIn(row_labels_group), run_time=0.8)
        self.play(Write(batch_size_lbl), run_time=0.5)
        self.play(Create(arr_to_W), FadeIn(W_box), run_time=0.6)
        self.play(Create(arr_out), Write(out_lbl), run_time=0.6)
        self.wait(1.5)

        # ── 6. What happens to dimensions ─────────────────────────────────────
        self.fade_all(batch_title, batch_bracket, row_labels_group,
                      batch_size_lbl, arr_to_W, W_box, arr_out, out_lbl)

        dim2_title = body_text("How shapes change step by step", color=WHITE)
        dim2_title.to_edge(UP, buff=0.7)
        self.play(Write(dim2_title), run_time=0.7)

        # Labeled shape boxes
        shape_steps = [
            ("[batch=1,\nd_in=3]",  BLUE_DARK,  BLUE_MED),
            ("[3 × 4\nweight]",     GREEN_DARK, GREEN_MED),
            ("[batch=1,\nd_out=4]", ORANGE_DARK, ORANGE_MED),
        ]
        shape_boxes = VGroup()
        for lbl, fill, stroke in shape_steps:
            box = rounded_box(1.5, 0.9, fill_color=fill,
                              stroke_color=stroke, label=lbl)
            shape_boxes.add(box)

        shape_boxes.arrange(RIGHT, buff=0.8)
        shape_boxes.move_to(ORIGIN)

        arrows_between = VGroup()
        for i in range(len(shape_boxes) - 1):
            arr = Arrow(shape_boxes[i].get_right(), shape_boxes[i+1].get_left(),
                        color=WHITE, buff=0.1, stroke_width=2)
            arrows_between.add(arr)

        op_labels = ["×", "="]
        op_txts = VGroup()
        for i, op in enumerate(op_labels):
            t = Text(op, color=WHITE, font_size=28)
            t.move_to(arrows_between[i].get_center() + UP * 0.3)
            op_txts.add(t)

        self.play(LaggedStart(*[FadeIn(b) for b in shape_boxes], lag_ratio=0.3),
                  run_time=1.0)
        self.play(LaggedStart(*[Create(a) for a in arrows_between], lag_ratio=0.3),
                  run_time=0.6)
        self.play(FadeIn(op_txts), run_time=0.4)

        dim_note = label_text("Input shape × Weight shape = Output shape", color=GREY_LIGHT)
        dim_note.next_to(shape_boxes, DOWN, buff=0.4)
        self.play(Write(dim_note), run_time=0.7)
        self.wait(1.5)

        # ── 7. Bias shifts the output distribution ─────────────────────────────
        self.fade_all(dim2_title, shape_boxes, arrows_between, op_txts, dim_note)

        bias2_title = body_text("Bias — shift the output up or down", color=WHITE)
        bias2_title.to_edge(UP, buff=0.7)
        self.play(Write(bias2_title), run_time=0.7)

        # Side-by-side: no bias vs with bias
        no_bias_box = rounded_box(2.0, 0.55, fill_color=BLUE_DARK,
                                  stroke_color=BLUE_MED, label="output = 0.61")
        no_bias_lbl = label_text("No bias", color=GREY_LIGHT)
        no_bias_lbl.next_to(no_bias_box, DOWN, buff=0.2)
        no_bias_group = VGroup(no_bias_box, no_bias_lbl)
        no_bias_group.to_edge(LEFT, buff=1.5)

        with_bias_box = rounded_box(2.0, 0.55, fill_color=ORANGE_DARK,
                                    stroke_color=ORANGE_MED, label="output = 0.71")
        with_bias_lbl = label_text("With bias +0.1", color=ORANGE_MED)
        with_bias_lbl.next_to(with_bias_box, DOWN, buff=0.2)
        with_bias_group = VGroup(with_bias_box, with_bias_lbl)
        with_bias_group.to_edge(RIGHT, buff=1.5)

        vs_txt = body_text("vs", color=GREY_MED)
        vs_txt.move_to(ORIGIN)

        bias2_note = label_text(
            "Bias gives the network a tunable starting point — like a default preference",
            color=GREY_LIGHT
        )
        bias2_note.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(no_bias_group), FadeIn(with_bias_group),
                  Write(vs_txt), run_time=0.8)
        self.play(Write(bias2_note), run_time=0.7)
        self.wait(1.5)

        # ── 8. Stacking linear layers ──────────────────────────────────────────
        self.fade_all(bias2_title, no_bias_group, with_bias_group,
                      vs_txt, bias2_note)

        stack_title = body_text("Stacking linear layers — building a pipeline", color=WHITE)
        stack_title.to_edge(UP, buff=0.7)
        self.play(Write(stack_title), run_time=0.8)

        stack_boxes = [
            ("Input\n[512]",   BLUE_DARK,   BLUE_MED),
            ("Linear 1\n512→2048", GREEN_DARK,  GREEN_MED),
            ("Linear 2\n2048→512", PURPLE_MED,  PURPLE_MED),
            ("Linear 3\n512→vocab", ORANGE_DARK, ORANGE_MED),
            ("Output\n[vocab]", BLUE_DARK,   BLUE_LIGHT),
        ]
        box_grp = VGroup()
        for lbl, fill, stroke in stack_boxes:
            b = rounded_box(1.6, 0.7, fill_color=fill, stroke_color=stroke, label=lbl)
            box_grp.add(b)

        box_grp.arrange(RIGHT, buff=0.5)
        box_grp.scale_to_fit_width(11.5)
        box_grp.move_to(ORIGIN)

        stack_arrows = VGroup()
        for i in range(len(box_grp) - 1):
            arr = Arrow(box_grp[i].get_right(), box_grp[i+1].get_left(),
                        color=WHITE, buff=0.05, stroke_width=1.8)
            stack_arrows.add(arr)

        role_labels = ["raw token", "expand &\nthink wide", "compress &\ndistil", "project to\nvocab", "next word"]
        role_grp = VGroup()
        for i, (lbl, box) in enumerate(zip(role_labels, box_grp)):
            t = label_text(lbl, color=GREY_LIGHT)
            t.next_to(box, DOWN, buff=0.25)
            role_grp.add(t)

        self.play(LaggedStart(*[FadeIn(b) for b in box_grp], lag_ratio=0.2),
                  run_time=1.2)
        self.play(LaggedStart(*[Create(a) for a in stack_arrows], lag_ratio=0.2),
                  run_time=0.8)
        self.play(LaggedStart(*[FadeIn(r) for r in role_grp], lag_ratio=0.15),
                  run_time=0.8)
        self.wait(2)

        # ── 9. Summary box ─────────────────────────────────────────────────────
        self.fade_all(stack_title, box_grp, stack_arrows, role_grp)

        summary_lines = [
            "📋  Vector          — a list of numbers",
            "🔢  Matrix          — a 2D grid of numbers",
            "⚙️   Dot product     — multiply pairs, then sum",
            "⚖️   Bias            — constant offset added after multiply",
            "📐  Dimensions      — layers reshape vectors (e.g. 512→2048)",
        ]
        summary = VGroup(*[body_text(line, color=WHITE) for line in summary_lines])
        summary.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        summary.move_to(ORIGIN)

        box = SurroundingRectangle(summary, color=BLUE_MED,
                                   buff=0.35, corner_radius=0.15)
        self.play(Create(box), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(l) for l in summary], lag_ratio=0.15),
                  run_time=1.5)
        self.wait(2)
