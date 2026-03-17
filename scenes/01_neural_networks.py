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

        # ── 5. Biases — each node has a default position ──────────────────────
        self.fade_all(flow_label, all_edges, *layers, title)

        bias_title = body_text("Biases — each neuron has a built-in offset", color=WHITE)
        bias_title.to_edge(UP, buff=0.7)
        self.play(Write(bias_title), run_time=0.8)

        # Draw a single neuron diagram: inputs + weights + bias → output
        inp1 = label_text("x1 = 0.5", color=GREEN_LIGHT)
        inp2 = label_text("x2 = 0.2", color=GREEN_LIGHT)
        inp3 = label_text("x3 = 0.8", color=GREEN_LIGHT)
        inputs_col = VGroup(inp1, inp2, inp3)
        inputs_col.arrange(DOWN, buff=0.35)
        inputs_col.to_edge(LEFT, buff=1.2)

        node_circle = Circle(radius=0.55, color=BLUE_MED,
                             fill_color=BLUE_DARK, fill_opacity=0.9)
        node_circle.move_to(ORIGIN)
        node_sum_label = label_text("Σ", color=WHITE)
        node_sum_label.move_to(node_circle.get_center())
        neuron = VGroup(node_circle, node_sum_label)

        w_labels = VGroup(
            label_text("w=0.3", color=BLUE_LIGHT),
            label_text("w=-0.5", color=BLUE_LIGHT),
            label_text("w=0.7", color=BLUE_LIGHT),
        )

        arrows_in = VGroup()
        for i, (inp, w_lbl) in enumerate(zip(inputs_col, w_labels)):
            arr = Arrow(inp.get_right(), node_circle.get_left(),
                        color=GREY_MED, buff=0.1, stroke_width=1.5)
            w_lbl.next_to(arr, UP, buff=0.05)
            arrows_in.add(VGroup(arr, w_lbl))

        bias_box = rounded_box(1.0, 0.45, fill_color=ORANGE_DARK,
                               stroke_color=ORANGE_MED, label="b=0.1")
        bias_box.next_to(node_circle, DOWN, buff=0.5)
        bias_arrow = Arrow(bias_box.get_top(), node_circle.get_bottom(),
                           color=ORANGE_MED, buff=0.05, stroke_width=1.5)

        out_arrow = Arrow(node_circle.get_right(), node_circle.get_right() + RIGHT * 1.2,
                          color=GREEN_MED, buff=0.0, stroke_width=2)
        out_label = label_text("output = 0.71", color=GREEN_MED)
        out_label.next_to(out_arrow, RIGHT, buff=0.1)

        self.play(FadeIn(inputs_col), run_time=0.6)
        self.play(Create(arrows_in), run_time=0.8)
        self.play(FadeIn(neuron), run_time=0.5)
        self.play(FadeIn(bias_box), Create(bias_arrow), run_time=0.6)
        self.play(Create(out_arrow), Write(out_label), run_time=0.6)
        self.wait(1.5)

        # ── 6. The forward pass step by step ──────────────────────────────────
        self.fade_all(bias_title, inputs_col, arrows_in, neuron,
                      bias_box, bias_arrow, out_arrow, out_label)

        fwd_title = body_text("The Forward Pass — numbers flowing through the network",
                              color=WHITE)
        fwd_title.to_edge(UP, buff=0.7)
        self.play(Write(fwd_title), run_time=0.8)

        step_labels = [
            "Input:  [0.5,  0.2,  0.8]",
            "× weights  →  weighted sum = 0.61",
            "+ bias (0.1)  →  pre-activation = 0.71",
            "ReLU(0.71)  →  output = 0.71",
            "Pass to next layer →",
        ]
        step_colors = [GREEN_LIGHT, BLUE_LIGHT, ORANGE_MED, YELLOW_MED, WHITE]
        steps_group = VGroup()
        for txt, col in zip(step_labels, step_colors):
            s = body_text(txt, color=col)
            steps_group.add(s)
        steps_group.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        steps_group.move_to(ORIGIN)

        for step in steps_group:
            self.play(FadeIn(step), run_time=0.5)
            self.wait(0.6)
        self.wait(1)

        # ── 7. What makes a neuron "fire"? ─────────────────────────────────────
        self.fade_all(fwd_title, steps_group)

        fire_title = body_text("What makes a neuron fire?", color=WHITE)
        fire_title.to_edge(UP, buff=0.7)
        self.play(Write(fire_title), run_time=0.7)

        # Bar chart showing low vs high input sum
        bar_labels = ["Low sum\n(-1.2)", "Medium\n(0.3)", "High sum\n(2.1)"]
        bar_probs   = [0.0, 0.3, 1.0]
        bar_colors  = [RED_MED, YELLOW_MED, GREEN_MED]
        bars = make_prob_bars(bar_labels, bar_probs,
                              max_height=2.5, bar_width=0.7,
                              colors=bar_colors)
        bars.move_to(ORIGIN + DOWN * 0.2)

        bar_note = label_text("Output after ReLU — low inputs are silenced", color=GREY_LIGHT)
        bar_note.next_to(bars, DOWN, buff=0.4)

        self.play(FadeIn(bars), run_time=1.0)
        self.play(Write(bar_note), run_time=0.7)
        self.wait(1.5)

        # ── 8. Deep vs Shallow networks ────────────────────────────────────────
        self.fade_all(fire_title, bars, bar_note)

        depth_title = body_text("Deep vs Shallow — more layers = more abstraction",
                                color=WHITE)
        depth_title.to_edge(UP, buff=0.7)
        self.play(Write(depth_title), run_time=0.8)

        # Shallow: 2 hidden layers
        shallow_sizes = [2, 3, 2]
        shallow_cols  = [GREEN_MED, BLUE_MED, ORANGE_MED]
        shallow_layers = [make_layer(n, color=c) for n, c in zip(shallow_sizes, shallow_cols)]
        shallow_net = VGroup(*shallow_layers)
        shallow_net.arrange(RIGHT, buff=1.2)
        shallow_net.scale(0.85)
        shallow_net.to_edge(LEFT, buff=0.8)
        shallow_net.shift(DOWN * 0.3)
        shallow_edges = VGroup()
        for i in range(len(shallow_layers) - 1):
            shallow_edges.add(connect_layers(shallow_layers[i], shallow_layers[i+1]))

        shallow_lbl = label_text("Shallow (2 hidden layers)", color=GREY_LIGHT)
        shallow_lbl.next_to(shallow_net, DOWN, buff=0.3)

        # Deep: 5 hidden layers
        deep_sizes = [2, 3, 3, 3, 3, 3, 2]
        deep_cols  = [GREEN_MED, BLUE_DARK, BLUE_MED, PURPLE_MED,
                      BLUE_MED, BLUE_DARK, ORANGE_MED]
        deep_layers = [make_layer(n, color=c) for n, c in zip(deep_sizes, deep_cols)]
        deep_net = VGroup(*deep_layers)
        deep_net.arrange(RIGHT, buff=0.7)
        deep_net.scale(0.7)
        deep_net.to_edge(RIGHT, buff=0.5)
        deep_net.shift(DOWN * 0.3)
        deep_edges = VGroup()
        for i in range(len(deep_layers) - 1):
            deep_edges.add(connect_layers(deep_layers[i], deep_layers[i+1]))

        deep_lbl = label_text("Deep (5 hidden layers)\nMore levels of abstraction",
                              color=GREEN_LIGHT)
        deep_lbl.next_to(deep_net, DOWN, buff=0.3)

        self.play(FadeIn(shallow_edges), FadeIn(deep_edges), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(l) for l in shallow_layers], lag_ratio=0.2),
                  LaggedStart(*[FadeIn(l) for l in deep_layers], lag_ratio=0.15),
                  run_time=1.2)
        self.play(Write(shallow_lbl), Write(deep_lbl), run_time=0.7)
        self.wait(1.5)

        # ── 9. Backpropagation intuition ───────────────────────────────────────
        self.fade_all(depth_title, shallow_edges, deep_edges, shallow_lbl, deep_lbl,
                      *shallow_layers, *deep_layers)

        back_title = body_text("Backpropagation — learn by fixing mistakes", color=WHITE)
        back_title.to_edge(UP, buff=0.7)
        self.play(Write(back_title), run_time=0.8)

        # Rebuild a small 3-layer network for the backprop demo
        bp_sizes = [2, 3, 2]
        bp_cols  = [GREEN_MED, BLUE_MED, ORANGE_MED]
        bp_layers = [make_layer(n, color=c) for n, c in zip(bp_sizes, bp_cols)]
        bp_net = VGroup(*bp_layers)
        bp_net.arrange(RIGHT, buff=1.6)
        bp_net.move_to(ORIGIN + DOWN * 0.2)

        bp_edges = VGroup()
        for i in range(len(bp_layers) - 1):
            bp_edges.add(connect_layers(bp_layers[i], bp_layers[i+1]))

        self.play(FadeIn(bp_edges), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(l) for l in bp_layers], lag_ratio=0.2),
                  run_time=0.8)

        # Forward pass: green arrows left to right
        fwd_arrows = VGroup()
        for i in range(len(bp_layers) - 1):
            src_nodes = bp_layers[i][0] if isinstance(bp_layers[i][0], VGroup) else bp_layers[i]
            dst_nodes = bp_layers[i+1][0] if isinstance(bp_layers[i+1][0], VGroup) else bp_layers[i+1]
            arr = Arrow(src_nodes.get_right() + UP * 0.5,
                        dst_nodes.get_left() + UP * 0.5,
                        color=GREEN_MED, buff=0.1, stroke_width=2.5)
            fwd_arrows.add(arr)

        fwd_lbl = label_text("Forward: make prediction", color=GREEN_MED)
        fwd_lbl.to_edge(DOWN, buff=0.7)

        self.play(LaggedStart(*[Create(a) for a in fwd_arrows], lag_ratio=0.3),
                  Write(fwd_lbl), run_time=1.0)
        self.wait(1)

        # Backward pass: red arrows right to left
        back_arrows = VGroup()
        for i in range(len(bp_layers) - 1, 0, -1):
            src_nodes = bp_layers[i][0] if isinstance(bp_layers[i][0], VGroup) else bp_layers[i]
            dst_nodes = bp_layers[i-1][0] if isinstance(bp_layers[i-1][0], VGroup) else bp_layers[i-1]
            arr = Arrow(src_nodes.get_left() + DOWN * 0.5,
                        dst_nodes.get_right() + DOWN * 0.5,
                        color=RED_MED, buff=0.1, stroke_width=2.5)
            back_arrows.add(arr)

        back_lbl = label_text("Backward: fix mistakes (adjust weights)", color=RED_MED)
        back_lbl.next_to(fwd_lbl, DOWN, buff=0.15)

        self.play(LaggedStart(*[Create(a) for a in back_arrows], lag_ratio=0.3),
                  Write(back_lbl), run_time=1.0)
        self.wait(1.5)

        # ── 10. Summary box ────────────────────────────────────────────────────
        self.fade_all(back_title, bp_edges, fwd_arrows, back_arrows,
                      fwd_lbl, back_lbl, *bp_layers)

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
