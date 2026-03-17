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
                    DecimalNumber(v, num_decimal_places=1, color=color, font_size=18)
                    for v in r
                ])
                row_grp.arrange(RIGHT, buff=0.35)
                entries.add(row_grp)
            entries.arrange(DOWN, buff=0.25)
            bl = MathTex("[", color=WHITE, font_size=44)
            br = MathTex("]", color=WHITE, font_size=44)
            bl.next_to(entries, LEFT, buff=0.08)
            br.next_to(entries, RIGHT, buff=0.08)
            return VGroup(bl, entries, br)

        W = make_matrix(weight_rows)
        w_label = label_text("Weight\nmatrix W", color=BLUE_LIGHT)
        w_label.next_to(W, UP, buff=0.2)
        w_group = VGroup(w_label, W)
        w_group.move_to(ORIGIN)

        # Multiply sign
        mult_sign = MathTex("\\times", color=WHITE, font_size=36)
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

        equals_sign = MathTex("=", color=WHITE, font_size=36)
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
        plus_b = MathTex("+ b", color=ORANGE_DARK, font_size=36)
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
