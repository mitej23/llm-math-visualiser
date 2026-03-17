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
            ("Sigmoid", sigmoid, GREEN_MED,  "1/(1+e⁻ˣ)"),
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

        # ── 4. Summary card ───────────────────────────────────────────────────
        self.fade_all(highlight, valve_text, plots, title)

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
        box = SurroundingRectangle(table_items, color=BLUE_MED,
                                   buff=0.35, corner_radius=0.15)

        heading = body_text("Quick Reference", color=WHITE)
        heading.next_to(box, UP, buff=0.2)

        self.play(Write(heading), Create(box), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(r) for r in table_items], lag_ratio=0.2),
                  run_time=1.2)
        self.wait(2)
