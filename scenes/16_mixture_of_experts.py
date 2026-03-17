"""
Scene 16 — Mixture of Experts (MoE)
Run: manim -pql 16_mixture_of_experts.py MixtureOfExpertsScene
"""

from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class MixtureOfExpertsScene(LLMScene):
    def construct(self):
        title = self.show_title("Mixture of Experts", "Specialization at Scale")
        self.wait(0.5)
        self.fade_all(title)

        # ── 1. Dense vs sparse ────────────────────────────────────────────────
        compare_title = body_text("Dense model: every token uses every parameter",
                                  color=WHITE)
        compare_title.to_edge(UP, buff=0.6)
        self.play(Write(compare_title), run_time=0.7)

        # Dense — all neurons lit
        dense_nodes = VGroup(*[
            Circle(radius=0.22, color=BLUE_MED, fill_color=BLUE_MED, fill_opacity=0.8)
            for _ in range(12)
        ])
        dense_nodes.arrange_in_grid(3, 4, buff=0.3)
        dense_nodes.shift(LEFT * 3.5)
        dense_lbl = label_text("Dense FFN\n(all active)", color=BLUE_MED)
        dense_lbl.next_to(dense_nodes, DOWN, buff=0.2)

        # MoE — only 2 of 8 experts lit
        expert_colors = [GREY_DARK] * 8
        expert_colors[2] = GREEN_MED
        expert_colors[5] = ORANGE_MED
        moe_nodes = VGroup(*[
            Circle(radius=0.22,
                   color=col if col != GREY_DARK else GREY_MED,
                   fill_color=col if col != GREY_DARK else GREY_DARK,
                   fill_opacity=0.8 if col != GREY_DARK else 0.3)
            for col in expert_colors
        ])
        moe_nodes.arrange_in_grid(2, 4, buff=0.3)
        moe_nodes.shift(RIGHT * 3.0)
        moe_lbl = label_text("MoE  (8 experts, 2 active)", color=GREEN_MED)
        moe_lbl.next_to(moe_nodes, DOWN, buff=0.2)

        active_lbl = label_text("Active", color=GREEN_MED)
        active_lbl.next_to(moe_nodes[2], UL, buff=0.08)
        active_lbl2 = label_text("Active", color=ORANGE_MED)
        active_lbl2.next_to(moe_nodes[5], UL, buff=0.08)

        vs = body_text("vs.", color=GREY_LIGHT).move_to(ORIGIN + LEFT * 0.5)

        self.play(FadeIn(dense_nodes), FadeIn(dense_lbl), run_time=0.7)
        self.play(Write(vs), FadeIn(moe_nodes), FadeIn(moe_lbl),
                  FadeIn(active_lbl), FadeIn(active_lbl2), run_time=0.9)
        self.wait(1)
        self.fade_all(compare_title, dense_nodes, dense_lbl,
                      vs, moe_nodes, moe_lbl, active_lbl, active_lbl2)

        # ── 2. Router mechanism ────────────────────────────────────────────────
        router_title = body_text("The Router decides which experts handle each token:",
                                  color=WHITE)
        router_title.to_edge(UP, buff=0.6)
        self.play(Write(router_title), run_time=0.7)

        token_box = rounded_box(1.4, 0.65,
                                fill_color=BLUE_DARK, stroke_color=BLUE_MED,
                                label='"chemistry"', label_color=BLUE_LIGHT)
        token_box.shift(LEFT * 5)

        router_box = rounded_box(1.6, 0.65,
                                 fill_color=PURPLE_MED + "33", stroke_color=PURPLE_MED,
                                 label="Router\n(small linear)", label_color=PURPLE_MED)
        router_box.shift(LEFT * 2)

        n_experts = 8
        expert_boxes = VGroup()
        colors_e = [GREY_MED, GREY_MED, GREEN_MED, GREY_MED,
                    GREY_MED, ORANGE_MED, GREY_MED, GREY_MED]
        opacities = [0.2, 0.2, 0.9, 0.2, 0.2, 0.9, 0.2, 0.2]
        for i, (col, op) in enumerate(zip(colors_e, opacities)):
            e = rounded_box(0.85, 0.55,
                            fill_color=col + "33" if op > 0.5 else GREY_DARK,
                            stroke_color=col,
                            label=f"E{i+1}", label_color=col)
            e.set_opacity(op)
            expert_boxes.add(e)

        expert_boxes.arrange(DOWN, buff=0.15)
        expert_boxes.shift(RIGHT * 1.5)

        combine_box = rounded_box(1.6, 0.65,
                                  fill_color=GREEN_DARK, stroke_color=GREEN_MED,
                                  label="Weighted\nCombine", label_color=GREEN_MED)
        combine_box.shift(RIGHT * 4.0)

        out_box = rounded_box(1.4, 0.65,
                              fill_color=BLUE_DARK, stroke_color=BLUE_MED,
                              label="Output", label_color=BLUE_MED)
        out_box.shift(RIGHT * 6.2)

        # Arrows
        a0 = Arrow(token_box.get_right(), router_box.get_left(),
                   color=GREY_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.18)
        active_experts = [expert_boxes[2], expert_boxes[5]]
        a1 = Arrow(router_box.get_right(),
                   active_experts[0].get_left() + LEFT * 0.1,
                   color=GREEN_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        a2 = Arrow(router_box.get_right(),
                   active_experts[1].get_left() + LEFT * 0.1,
                   color=ORANGE_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        a3 = Arrow(active_experts[0].get_right(),
                   combine_box.get_left() + UP * 0.15,
                   color=GREEN_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        a4 = Arrow(active_experts[1].get_right(),
                   combine_box.get_left() + DOWN * 0.15,
                   color=ORANGE_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        a5 = Arrow(combine_box.get_right(), out_box.get_left(),
                   color=GREY_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.18)

        diagram = VGroup(token_box, a0, router_box, expert_boxes,
                         a1, a2, a3, a4, combine_box, a5, out_box)
        diagram.scale_to_fit_width(13.5)
        diagram.move_to(ORIGIN + DOWN * 0.2)

        self.play(LaggedStart(*[FadeIn(m) for m in [
            token_box, a0, router_box, expert_boxes,
            a1, a2, a3, a4, combine_box, a5, out_box
        ]], lag_ratio=0.08), run_time=2.2)
        self.wait(1)
        self.fade_all(router_title, diagram)

        # ── 3. Real-world models ───────────────────────────────────────────────
        models_title = body_text("MoE in the wild:", color=WHITE)
        models_title.to_edge(UP, buff=0.6)
        self.play(Write(models_title), run_time=0.5)

        model_data = [
            ("Mixtral 8×7B",  BLUE_MED,    "8 experts, 2 active · Total 47B, Active 13B"),
            ("Grok-1",        ORANGE_MED,  "64 experts, 8 active · 314B total, 86B active"),
            ("DeepSeek-V3",   GREEN_MED,   "256 experts, 8 active · frontier quality"),
            ("GPT-4 (est.)",  PURPLE_MED,  "~8 experts · ~1.8T total parameters"),
        ]
        rows = VGroup()
        for name, col, desc in model_data:
            n = body_text(name, color=col)
            d = label_text(desc, color=WHITE)
            d.next_to(n, RIGHT, buff=0.4)
            rows.add(VGroup(n, d))

        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        rows.move_to(ORIGIN + DOWN * 0.3)
        box = SurroundingRectangle(rows, color=GREY_MED, buff=0.35, corner_radius=0.15)

        self.play(Create(box), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.2),
                  run_time=1.3)

        summary = label_text(
            "Key insight: more total knowledge, same compute per token!",
            color=YELLOW_MED,
        )
        summary.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(summary), run_time=0.7)
        self.wait(2)
