"""
Scene 03 — Activation Functions
Run: manim -pql 03_activation_functions.py ActivationFunctionsScene
"""

from manim import *
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


def relu(x):
    return np.maximum(0, x)

def silu(x):
    return x / (1 + np.exp(-x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class ActivationFunctionsScene(LLMScene):
    def construct(self):
        title = self.show_title("Activation Functions", "Breaking Linearity")
        self.wait(0.5)

        # ── 1. Why we need non-linearity ──────────────────────────────────────
        explanation = body_text(
            "Without activations, stacking linear layers = still one linear layer",
            color=GREY_LIGHT
        )
        explanation.next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(explanation), run_time=0.8)
        self.wait(1)

        # ── 2. Plot all four activation functions ──────────────────────────────
        self.fade_all(explanation)

        axes_config = dict(
            x_range=[-4, 4, 1],
            y_range=[-1.5, 3, 0.5],
            axis_config={"color": GREY_MED, "include_numbers": False},
            tips=False,
        )

        functions = [
            ("ReLU",    relu,    RED_MED,    "max(0, x)"),
            ("SiLU",    silu,    BLUE_MED,   "x · σ(x)"),
            ("Sigmoid", sigmoid, GREEN_MED,  "1/(1+e^-x)"),
            ("GELU",    gelu,    ORANGE_MED, "smooth gating"),
        ]

        # Arrange in a 2×2 grid
        plots = VGroup()
        for fn_name, fn, col, desc in functions:
            ax = Axes(**axes_config, x_length=3.0, y_length=2.0)
            curve = ax.plot(fn, color=col, stroke_width=2.5)
            fn_label = label_text(fn_name, color=col)
            fn_label.next_to(ax, UP, buff=0.1)
            desc_label = label_text(desc, color=GREY_LIGHT)
            desc_label.next_to(ax, DOWN, buff=0.1)
            # zero reference lines
            h_line = DashedLine(ax.c2p(-4, 0), ax.c2p(4, 0),
                                color=GREY_MED, stroke_width=1)
            v_line = DashedLine(ax.c2p(0, -1.5), ax.c2p(0, 3),
                                color=GREY_MED, stroke_width=1)
            plots.add(VGroup(ax, h_line, v_line, curve, fn_label, desc_label))

        plots.arrange_in_grid(2, 2, buff=0.6)
        plots.scale_to_fit_height(5.5)
        plots.shift(DOWN * 0.3)

        self.play(LaggedStart(*[Create(p) for p in plots], lag_ratio=0.2),
                  run_time=2.5)
        self.wait(1.5)

        # ── 3. Highlight ReLU — the one-way valve ─────────────────────────────
        relu_plot = plots[0]
        highlight = SurroundingRectangle(relu_plot, color=YELLOW_MED,
                                         buff=0.1, corner_radius=0.1)
        valve_text = body_text("One-way valve: blocks negatives, passes positives",
                               color=YELLOW_MED)
        valve_text.to_edge(DOWN, buff=0.35)

        self.play(Create(highlight), Write(valve_text), run_time=0.8)
        self.wait(1.5)

        # ── 4. Without activation functions — collapsing layers ────────────────
        self.fade_all(highlight, valve_text, plots, title)

        no_act_title = body_text("Without activations — layers collapse into one",
                                 color=WHITE)
        no_act_title.to_edge(UP, buff=0.7)
        self.play(Write(no_act_title), run_time=0.8)

        # Two linear boxes → collapse into one
        box_a = rounded_box(1.6, 0.7, fill_color=BLUE_DARK,
                            stroke_color=BLUE_MED, label="Linear 1")
        box_b = rounded_box(1.6, 0.7, fill_color=GREEN_DARK,
                            stroke_color=GREEN_MED, label="Linear 2")
        two_boxes = VGroup(box_a, box_b)
        two_boxes.arrange(RIGHT, buff=0.8)
        two_boxes.move_to(ORIGIN + UP * 0.5)
        arr_ab = Arrow(box_a.get_right(), box_b.get_left(),
                       color=WHITE, buff=0.05, stroke_width=2)
        note_a = label_text("no activation\nbetween them", color=GREY_LIGHT)
        note_a.next_to(arr_ab, DOWN, buff=0.15)

        collapsed = rounded_box(2.0, 0.7, fill_color=GREY_DARK,
                                stroke_color=GREY_MED, label="One linear layer")
        collapsed.move_to(ORIGIN + DOWN * 1.2)
        collapse_lbl = label_text("Mathematically equivalent!", color=YELLOW_MED)
        collapse_lbl.next_to(collapsed, DOWN, buff=0.25)
        collapse_arr = Arrow(two_boxes.get_bottom(), collapsed.get_top(),
                             color=YELLOW_MED, buff=0.1, stroke_width=2)

        self.play(FadeIn(two_boxes), Create(arr_ab), run_time=0.7)
        self.play(Write(note_a), run_time=0.5)
        self.play(Create(collapse_arr), FadeIn(collapsed), run_time=0.7)
        self.play(Write(collapse_lbl), run_time=0.5)
        self.wait(1.5)

        # ── 5. ReLU deep dive — plot with annotated regions ────────────────────
        self.fade_all(no_act_title, two_boxes, arr_ab, note_a,
                      collapse_arr, collapsed, collapse_lbl)

        relu_title = body_text("ReLU — the one-way valve", color=RED_MED)
        relu_title.to_edge(UP, buff=0.7)
        self.play(Write(relu_title), run_time=0.7)

        relu_ax = Axes(
            x_range=[-3, 3, 1],
            y_range=[-0.5, 3.5, 1],
            x_length=7.0,
            y_length=4.0,
            axis_config={"color": GREY_MED, "include_numbers": False},
            tips=False,
        )
        relu_ax.move_to(ORIGIN + DOWN * 0.2)
        relu_curve = relu_ax.plot(relu, color=RED_MED, stroke_width=3)
        h_ref = DashedLine(relu_ax.c2p(-3, 0), relu_ax.c2p(3, 0),
                           color=GREY_MED, stroke_width=1)
        v_ref = DashedLine(relu_ax.c2p(0, -0.5), relu_ax.c2p(0, 3.5),
                           color=GREY_MED, stroke_width=1)

        neg_region = relu_ax.get_area(
            relu_ax.plot(relu, x_range=[-3, 0]),
            x_range=[-3, 0], color=RED_MED, opacity=0.15
        )
        pos_region = relu_ax.get_area(
            relu_ax.plot(relu, x_range=[0, 3]),
            x_range=[0, 3], color=GREEN_MED, opacity=0.15
        )

        neg_lbl = label_text("negative → output 0\n(blocked)", color=RED_MED)
        neg_lbl.next_to(relu_ax.c2p(-1.5, 0.5), UP, buff=0.1)
        pos_lbl = label_text("positive → unchanged\n(passed through)", color=GREEN_MED)
        pos_lbl.next_to(relu_ax.c2p(1.5, 1.5), UP, buff=0.1)

        self.play(Create(relu_ax), Create(h_ref), Create(v_ref), run_time=0.7)
        self.play(Create(relu_curve), run_time=0.8)
        self.play(FadeIn(neg_region), FadeIn(pos_region), run_time=0.5)
        self.play(Write(neg_lbl), Write(pos_lbl), run_time=0.7)
        self.wait(1.5)

        # ── 6. Sigmoid — the probability squisher ──────────────────────────────
        self.fade_all(relu_title, relu_ax, relu_curve, h_ref, v_ref,
                      neg_region, pos_region, neg_lbl, pos_lbl)

        sig_title = body_text("Sigmoid — squishes everything to 0-1", color=GREEN_MED)
        sig_title.to_edge(UP, buff=0.7)
        self.play(Write(sig_title), run_time=0.7)

        sig_ax = Axes(
            x_range=[-5, 5, 1],
            y_range=[-0.1, 1.2, 0.5],
            x_length=7.0,
            y_length=4.0,
            axis_config={"color": GREY_MED, "include_numbers": False},
            tips=False,
        )
        sig_ax.move_to(ORIGIN + DOWN * 0.2)
        sig_curve = sig_ax.plot(sigmoid, color=GREEN_MED, stroke_width=3)
        h_zero = DashedLine(sig_ax.c2p(-5, 0), sig_ax.c2p(5, 0),
                            color=GREY_MED, stroke_width=1)
        h_one  = DashedLine(sig_ax.c2p(-5, 1), sig_ax.c2p(5, 1),
                            color=GREY_LIGHT, stroke_width=1)

        lbl_zero = label_text("0", color=GREY_LIGHT)
        lbl_zero.next_to(sig_ax.c2p(-5, 0), LEFT, buff=0.2)
        lbl_one  = label_text("1", color=GREY_LIGHT)
        lbl_one.next_to(sig_ax.c2p(-5, 1), LEFT, buff=0.2)

        sig_note = label_text("Use case: probability output (0 = impossible, 1 = certain)",
                              color=GREY_LIGHT)
        sig_note.to_edge(DOWN, buff=0.5)

        self.play(Create(sig_ax), Create(h_zero), Create(h_one), run_time=0.7)
        self.play(Write(lbl_zero), Write(lbl_one), run_time=0.4)
        self.play(Create(sig_curve), run_time=0.8)
        self.play(Write(sig_note), run_time=0.6)
        self.wait(1.5)

        # ── 7. SiLU vs ReLU side by side ──────────────────────────────────────
        self.fade_all(sig_title, sig_ax, sig_curve, h_zero, h_one,
                      lbl_zero, lbl_one, sig_note)

        silu_title = body_text("SiLU vs ReLU — smooth wins for modern LLMs",
                               color=WHITE)
        silu_title.to_edge(UP, buff=0.7)
        self.play(Write(silu_title), run_time=0.8)

        compare_config = dict(
            x_range=[-3, 3, 1],
            y_range=[-0.5, 3.5, 1],
            x_length=4.5,
            y_length=3.5,
            axis_config={"color": GREY_MED, "include_numbers": False},
            tips=False,
        )

        relu_ax2 = Axes(**compare_config)
        relu_ax2.to_edge(LEFT, buff=1.0)
        relu_ax2.shift(DOWN * 0.3)
        relu_curve2 = relu_ax2.plot(relu, color=RED_MED, stroke_width=3)
        relu_lbl2 = label_text("ReLU", color=RED_MED)
        relu_lbl2.next_to(relu_ax2, UP, buff=0.2)
        relu_note2 = label_text("Sharp corner\nDead neuron risk", color=GREY_LIGHT)
        relu_note2.next_to(relu_ax2, DOWN, buff=0.2)

        silu_ax = Axes(**compare_config)
        silu_ax.to_edge(RIGHT, buff=1.0)
        silu_ax.shift(DOWN * 0.3)
        silu_curve = silu_ax.plot(silu, color=BLUE_MED, stroke_width=3)
        silu_lbl = label_text("SiLU", color=BLUE_MED)
        silu_lbl.next_to(silu_ax, UP, buff=0.2)
        silu_note = label_text("Smooth gradient\nUsed in LLaMA", color=BLUE_LIGHT)
        silu_note.next_to(silu_ax, DOWN, buff=0.2)

        self.play(Create(relu_ax2), Create(silu_ax), run_time=0.7)
        self.play(Create(relu_curve2), Create(silu_curve), run_time=0.8)
        self.play(Write(relu_lbl2), Write(silu_lbl), run_time=0.5)
        self.play(Write(relu_note2), Write(silu_note), run_time=0.6)
        self.wait(1.5)

        # ── 8. Dead ReLU problem ───────────────────────────────────────────────
        self.fade_all(silu_title, relu_ax2, relu_curve2, relu_lbl2, relu_note2,
                      silu_ax, silu_curve, silu_lbl, silu_note)

        dead_title = body_text("Dead ReLU — a neuron stuck at zero forever",
                               color=RED_MED)
        dead_title.to_edge(UP, buff=0.7)
        self.play(Write(dead_title), run_time=0.8)

        # Show a neuron that is "dead"
        dead_node = Circle(radius=0.55, color=GREY_MED,
                           fill_color=GREY_DARK, fill_opacity=0.9)
        dead_node.move_to(ORIGIN)
        dead_node_lbl = label_text("output = 0", color=GREY_MED)
        dead_node_lbl.move_to(dead_node.get_center())

        input_arrows_dead = VGroup()
        for offset in [-0.5, 0, 0.5]:
            arr = Arrow(dead_node.get_left() + LEFT * 1.5 + UP * offset,
                        dead_node.get_left(),
                        color=GREY_MED, buff=0.05, stroke_width=1.5)
            val_lbl = label_text("negative", color=RED_MED)
            val_lbl.next_to(arr, UP, buff=0.05)
            input_arrows_dead.add(VGroup(arr, val_lbl))

        out_arr_dead = Arrow(dead_node.get_right(), dead_node.get_right() + RIGHT * 1.5,
                             color=GREY_MED, buff=0.05, stroke_width=1.5)
        out_dead_lbl = label_text("always 0\n(no update)", color=GREY_MED)
        out_dead_lbl.next_to(out_arr_dead, RIGHT, buff=0.1)

        warning_lbl = body_text("Gradient = 0  →  weight never updates  →  neuron is dead",
                                color=RED_MED)
        warning_lbl.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(dead_node), Write(dead_node_lbl), run_time=0.6)
        self.play(FadeIn(input_arrows_dead), run_time=0.7)
        self.play(Create(out_arr_dead), Write(out_dead_lbl), run_time=0.6)
        self.play(Write(warning_lbl), run_time=0.7)
        self.wait(1.5)

        # ── 9. Which LLMs use which activation? ───────────────────────────────
        self.fade_all(dead_title, dead_node, dead_node_lbl, input_arrows_dead,
                      out_arr_dead, out_dead_lbl, warning_lbl)

        llm_title = body_text("Which LLMs use which activation?", color=WHITE)
        llm_title.to_edge(UP, buff=0.7)
        self.play(Write(llm_title), run_time=0.7)

        llm_rows = [
            ("GPT-2 / GPT-3",  BLUE_LIGHT,  "GELU",           ORANGE_MED),
            ("LLaMA 1/2/3",    GREEN_LIGHT, "SwiGLU (SiLU)",  BLUE_MED),
            ("BERT",           PURPLE_MED,  "GELU",           ORANGE_MED),
            ("Older CNNs",     GREY_LIGHT,  "ReLU",           RED_MED),
            ("LSTM / GRU",     YELLOW_MED,  "Sigmoid + Tanh", GREEN_MED),
        ]

        table_grp = VGroup()
        for model_name, model_col, act_name, act_col in llm_rows:
            model_txt = body_text(model_name, color=model_col)
            act_txt   = label_text(act_name, color=act_col)
            act_txt.next_to(model_txt, RIGHT, buff=0.6)
            row = VGroup(model_txt, act_txt)
            table_grp.add(row)

        table_grp.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        table_grp.move_to(ORIGIN)

        box = SurroundingRectangle(table_grp, color=BLUE_MED,
                                   buff=0.35, corner_radius=0.15)
        self.play(Create(box), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(r) for r in table_grp], lag_ratio=0.2),
                  run_time=1.5)
        self.wait(2)

        # ── 10. Summary card ───────────────────────────────────────────────────
        self.fade_all(llm_title, box, table_grp)

        rows = [
            ("ReLU",    RED_MED,    "Sharp corner — fast & simple"),
            ("SiLU",    BLUE_MED,   "Smooth ReLU — used in LLaMA"),
            ("Sigmoid", GREEN_MED,  "Squish to 0–1 — good for gates"),
            ("GELU",    ORANGE_MED, "Probabilistic gate — used in GPT/BERT"),
        ]

        table_items = VGroup()
        for name, col, desc in rows:
            name_txt = body_text(name, color=col)
            desc_txt = label_text(desc, color=WHITE)
            desc_txt.next_to(name_txt, RIGHT, buff=0.4)
            row_grp = VGroup(name_txt, desc_txt)
            table_items.add(row_grp)

        table_items.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        table_items.move_to(ORIGIN)
        box2 = SurroundingRectangle(table_items, color=BLUE_MED,
                                    buff=0.35, corner_radius=0.15)

        heading = body_text("Quick Reference", color=WHITE)
        heading.next_to(box2, UP, buff=0.2)

        self.play(Write(heading), Create(box2), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(r) for r in table_items], lag_ratio=0.2),
                  run_time=1.2)
        self.wait(2)
