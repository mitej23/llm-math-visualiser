"""
Scene 01 — Neural Networks
Run: manim -pql 01_neural_networks.py NeuralNetworksScene
"""

from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class NeuralNetworksScene(LLMScene):
    def construct(self):
        # ── Title ─────────────────────────────────────────────────────────────
        title = self.show_title("Neural Networks", "Nodes · Layers · Hidden States")
        self.wait(0.5)

        # ── 1. Build a 3-layer network diagram ────────────────────────────────
        layer_sizes = [3, 4, 4, 2]
        colors = [GREEN_MED, BLUE_MED, PURPLE_MED, ORANGE_MED]
        layer_labels = ["Input\nLayer", "Hidden\nLayer 1", "Hidden\nLayer 2", "Output\nLayer"]

        layers = []
        for i, (n, col, lbl) in enumerate(zip(layer_sizes, colors, layer_labels)):
            layer = make_layer(n, color=col, label=lbl)
            layers.append(layer)

        network = VGroup(*layers)
        network.arrange(RIGHT, buff=1.8)
        network.shift(DOWN * 0.3)

        # Draw edges first (behind nodes)
        all_edges = VGroup()
        for i in range(len(layers) - 1):
            edges = connect_layers(layers[i], layers[i + 1])
            all_edges.add(edges)

        self.play(FadeIn(all_edges), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(l) for l in layers], lag_ratio=0.25),
                  run_time=1.5)
        self.wait(0.8)

        # ── 2. Highlight the hidden layers ────────────────────────────────────
        hidden_highlight = SurroundingRectangle(
            VGroup(layers[1], layers[2]),
            color=YELLOW_MED, buff=0.25, corner_radius=0.15
        )
        hidden_label = body_text("Hidden Layers", color=YELLOW_MED)
        hidden_label.next_to(hidden_highlight, UP, buff=0.2)

        self.play(Create(hidden_highlight), Write(hidden_label), run_time=0.8)
        self.wait(1)

        # ── 3. Zoom in on one node — show weighted sum ────────────────────────
        self.fade_all(hidden_highlight, hidden_label)

        node_zoom_label = body_text("Each node: multiply inputs, add bias, activate",
                                    color=WHITE)
        node_zoom_label.to_edge(DOWN, buff=0.4)
        self.play(Write(node_zoom_label), run_time=0.9)
        self.wait(1)

        # Highlight one hidden node
        target_node = layers[1][0][1]   # second node of hidden layer 1
        highlight_circle = Circle(radius=0.45, color=YELLOW_MED, stroke_width=3)
        highlight_circle.move_to(target_node.get_center())
        self.play(Create(highlight_circle), run_time=0.6)
        self.wait(0.8)

        # ── 4. Show data flowing forward ──────────────────────────────────────
        self.fade_all(node_zoom_label, highlight_circle)

        flow_label = body_text("Data flows left → right through every layer",
                               color=GREEN_LIGHT)
        flow_label.to_edge(DOWN, buff=0.4)
        self.play(Write(flow_label), run_time=0.8)

        # Animate dots flowing through edges left to right
        for i in range(len(layers) - 1):
            dots = []
            # pick a couple of representative edges per transition
            src_nodes = layers[i][0] if isinstance(layers[i][0], VGroup) else layers[i]
            dst_nodes = layers[i + 1][0] if isinstance(layers[i + 1][0], VGroup) else layers[i + 1]
            for src in list(src_nodes)[:2]:
                for dst in list(dst_nodes)[:2]:
                    dot = Dot(color=YELLOW_MED, radius=0.08)
                    dot.move_to(src.get_center())
                    dots.append(dot)
                    self.add(dot)
            self.play(
                *[dot.animate.move_to(dst.get_center())
                  for dot, dst in zip(dots, list(dst_nodes) * 2)],
                run_time=0.6
            )
            self.remove(*dots)
        self.wait(1)

        # ── 5. Summary box ────────────────────────────────────────────────────
        self.fade_all(flow_label, all_edges, *layers, title)

        summary_lines = [
            "🔵  Input Layer  — raw data enters here",
            "🟡  Hidden Layers — where 'thinking' happens",
            "🟢  Output Layer — the final prediction",
            "🔗  Weights       — learned connection strengths",
            "📝  Hidden State  — internal working memory",
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
