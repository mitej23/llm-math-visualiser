"""
Scene 21 — RLHF Overview
Run: manim -pql 21_rlhf_overview.py RLHFOverviewScene
"""

from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class RLHFOverviewScene(LLMScene):
    def construct(self):
        title = self.show_title("RLHF Overview", "The Full Alignment Pipeline")
        self.wait(0.5)
        self.fade_all(title)

        # ── 1. The three phases ────────────────────────────────────────────────
        phases_title = body_text("Three phases: SFT → Reward Model → RL Fine-Tuning",
                                 color=WHITE)
        phases_title.to_edge(UP, buff=0.6)
        self.play(Write(phases_title), run_time=0.7)

        phase_data = [
            ("Phase 1\nSFT",
             GREEN_MED,
             "Collect human demos\nFine-tune base model\nResult: SFT model"),
            ("Phase 2\nReward Model",
             ORANGE_MED,
             "Generate responses\nHumans rank them\nTrain reward network"),
            ("Phase 3\nRL + PPO",
             PURPLE_MED,
             "Sample prompts\nScore with RM\nUpdate via PPO"),
        ]

        phase_boxes = VGroup()
        for lbl, col, details in phase_data:
            header = body_text(lbl, color=col)
            detail_txt = label_text(details, color=WHITE)
            detail_txt.next_to(header, DOWN, buff=0.2)
            content = VGroup(header, detail_txt)
            bg = SurroundingRectangle(content, color=col, fill_color=col + "11",
                                      fill_opacity=1, buff=0.3, corner_radius=0.12)
            phase_boxes.add(VGroup(bg, content))

        phase_boxes.arrange(RIGHT, buff=0.6)
        phase_boxes.scale_to_fit_width(12.5)
        phase_boxes.move_to(ORIGIN + DOWN * 0.2)

        phase_arrows = VGroup(*[
            Arrow(phase_boxes[i][0].get_right(), phase_boxes[i + 1][0].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.18)
            for i in range(2)
        ])

        self.play(LaggedStart(*[FadeIn(p) for p in phase_boxes], lag_ratio=0.3),
                  run_time=1.5)
        self.play(LaggedStart(*[GrowArrow(a) for a in phase_arrows], lag_ratio=0.3),
                  run_time=0.6)
        self.wait(1.2)
        self.fade_all(phases_title, phase_boxes, phase_arrows)

        # ── 2. The RL loop diagram ─────────────────────────────────────────────
        loop_title = body_text("The Phase 3 RL Training Loop:", color=WHITE)
        loop_title.to_edge(UP, buff=0.6)
        self.play(Write(loop_title), run_time=0.6)

        nodes = [
            ("Prompt\nbatch",         GREY_LIGHT,  [-5.0,  0.5, 0]),
            ("RL Policy\ngenerates",  BLUE_MED,    [-2.0,  0.5, 0]),
            ("Reward\nModel scores",  ORANGE_MED,  [ 1.0,  0.5, 0]),
            ("KL\nPenalty",           RED_MED,     [ 1.0, -1.2, 0]),
            ("Total\nReward",         YELLOW_MED,  [ 4.0,  0.5, 0]),
            ("PPO\nUpdate",           PURPLE_MED,  [ 4.0, -1.2, 0]),
        ]

        node_boxes = {}
        node_group = VGroup()
        for name, col, pos in nodes:
            b = rounded_box(1.6, 0.75, fill_color=col + "22",
                            stroke_color=col, label=name, label_color=col)
            b.move_to(pos)
            node_boxes[name] = b
            node_group.add(b)

        # Reference policy box (frozen)
        ref_box = rounded_box(1.8, 0.65, fill_color=GREY_DARK,
                              stroke_color=GREY_MED,
                              label="Reference\nPolicy (frozen)", label_color=GREY_MED)
        ref_box.move_to([-2.0, -1.2, 0])
        node_group.add(ref_box)

        node_group.scale_to_fit_width(13.5)
        node_group.move_to(ORIGIN + DOWN * 0.2)

        self.play(LaggedStart(*[FadeIn(b) for b in node_group], lag_ratio=0.1),
                  run_time=1.5)

        # Draw edges
        def quick_arrow(a, b, col=GREY_MED):
            return Arrow(a.get_right(), b.get_left(), color=col, buff=0.05,
                         stroke_width=1.5, max_tip_length_to_length_ratio=0.15)

        e1 = quick_arrow(node_boxes["Prompt\nbatch"],        node_boxes["RL Policy\ngenerates"])
        e2 = quick_arrow(node_boxes["RL Policy\ngenerates"], node_boxes["Reward\nModel scores"])
        e3 = quick_arrow(node_boxes["Reward\nModel scores"], node_boxes["Total\nReward"])
        e4 = Arrow(node_boxes["KL\nPenalty"].get_right(),
                   node_boxes["Total\nReward"].get_bottom(),
                   color=RED_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        e5 = Arrow(node_boxes["Total\nReward"].get_bottom(),
                   node_boxes["PPO\nUpdate"].get_top(),
                   color=YELLOW_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        # PPO updates RL policy
        e6 = CurvedArrow(
            node_boxes["PPO\nUpdate"].get_left(),
            node_boxes["RL Policy\ngenerates"].get_bottom(),
            angle=TAU / 5, color=PURPLE_MED, stroke_width=1.5,
        )
        # Ref policy → KL
        e7 = Arrow(ref_box.get_right(),
                   node_boxes["KL\nPenalty"].get_left(),
                   color=GREY_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        # RL policy → KL
        e8 = Arrow(node_boxes["RL Policy\ngenerates"].get_bottom(),
                   node_boxes["KL\nPenalty"].get_top(),
                   color=GREY_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)

        edges = VGroup(e1, e2, e3, e4, e5, e6, e7, e8)
        self.play(LaggedStart(*[GrowArrow(e) for e in edges], lag_ratio=0.08),
                  run_time=1.5)
        self.wait(1.2)
        self.fade_all(loop_title, node_group, edges)

        # ── 3. Key InstructGPT result ─────────────────────────────────────────
        result_title = body_text("Key Result from InstructGPT (Ouyang et al., 2022):",
                                 color=WHITE)
        result_title.to_edge(UP, buff=0.6)
        self.play(Write(result_title), run_time=0.7)

        comparison = VGroup()
        models = [
            ("GPT-3 175B\n(base model)",   GREY_MED,  1.3, "100× bigger"),
            ("InstructGPT 1.3B\n(RLHF)",   GREEN_MED, 3.2, "Human preferred ✅"),
        ]
        for name, col, preference_scale, note in models:
            bar = Rectangle(width=preference_scale * 1.5, height=0.7,
                            fill_color=col, fill_opacity=0.8,
                            stroke_color=col, stroke_width=1)
            lbl = label_text(name, color=col)
            lbl.next_to(bar, LEFT, buff=0.2)
            n_lbl = label_text(note, color=col)
            n_lbl.next_to(bar, RIGHT, buff=0.2)
            comparison.add(VGroup(bar, lbl, n_lbl))

        comparison.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        comparison.move_to(ORIGIN)

        headline = body_text(
            "Alignment > Scale: 1.3B aligned beats 175B base",
            color=YELLOW_MED,
        )
        headline.next_to(comparison, DOWN, buff=0.5)

        source = label_text(
            "Source: Ouyang et al., arXiv 2203.02155 (2022)",
            color=GREY_MED,
        )
        source.to_edge(DOWN, buff=0.35)

        self.play(LaggedStart(*[FadeIn(c) for c in comparison], lag_ratio=0.4),
                  run_time=1.0)
        self.play(Write(headline), run_time=0.8)
        self.play(FadeIn(source), run_time=0.5)
        self.wait(2)
