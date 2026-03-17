"""
Scene 18 — Supervised Fine-Tuning (SFT)
Run: manim -pql 18_supervised_fine_tuning.py SFTScene
"""

from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class SFTScene(LLMScene):
    def construct(self):
        title = self.show_title("Supervised Fine-Tuning", "From Base Model to Assistant")
        self.wait(0.5)
        self.fade_all(title)

        # ── 1. Base model vs SFT model comparison ─────────────────────────────
        compare_title = body_text("Same knowledge — very different behaviour:", color=WHITE)
        compare_title.to_edge(UP, buff=0.6)
        self.play(Write(compare_title), run_time=0.6)

        prompt = rounded_box(5.5, 0.65, stroke_color=GREY_MED,
                             label='"Write a poem about autumn."',
                             label_color=GREY_LIGHT)
        prompt.shift(UP * 1.5)
        self.play(FadeIn(prompt), run_time=0.5)

        base_box = rounded_box(4.5, 1.5,
                               fill_color=GREY_DARK, stroke_color=GREY_MED,
                               label="Base model:\n\"Autumn, Autumn — See also: Fall,\nSeason, Temperature...\"",
                               label_color=GREY_MED)
        base_box.shift(LEFT * 3 + DOWN * 0.3)
        base_lbl = label_text("Continues text like a web page 😕", color=GREY_MED)
        base_lbl.next_to(base_box, DOWN, buff=0.2)

        sft_box = rounded_box(4.5, 1.5,
                              fill_color=GREEN_DARK, stroke_color=GREEN_MED,
                              label="SFT model:\n\"Golden leaves drift down,\nWhispering of summer gone...\"",
                              label_color=GREEN_LIGHT)
        sft_box.shift(RIGHT * 3 + DOWN * 0.3)
        sft_lbl = label_text("Responds as a helpful assistant ✅", color=GREEN_MED)
        sft_lbl.next_to(sft_box, DOWN, buff=0.2)

        arr_base = Arrow(prompt.get_bottom(), base_box.get_top(),
                         color=GREY_MED, buff=0.05, stroke_width=1.5,
                         max_tip_length_to_length_ratio=0.18)
        arr_sft  = Arrow(prompt.get_bottom(), sft_box.get_top(),
                         color=GREEN_MED, buff=0.05, stroke_width=1.5,
                         max_tip_length_to_length_ratio=0.18)

        self.play(GrowArrow(arr_base), FadeIn(base_box), FadeIn(base_lbl), run_time=0.8)
        self.play(GrowArrow(arr_sft),  FadeIn(sft_box),  FadeIn(sft_lbl),  run_time=0.8)
        self.wait(1.2)
        self.fade_all(compare_title, prompt, arr_base, base_box, base_lbl,
                      arr_sft, sft_box, sft_lbl)

        # ── 2. The SFT dataset — demo pairs ───────────────────────────────────
        dataset_title = body_text("SFT Dataset: (prompt, ideal response) pairs",
                                  color=WHITE)
        dataset_title.to_edge(UP, buff=0.6)
        self.play(Write(dataset_title), run_time=0.6)

        examples = [
            ('"Explain black holes simply."',
             '"A black hole is a region where gravity is so strong\nthat nothing — not even light — can escape."'),
            ('"Summarise this article in 3 bullets."',
             '"• Main finding\\n• Supporting evidence\\n• Conclusion"'),
            ('"Is this email spam?"',
             '"Yes — it contains common spam indicators..."'),
        ]

        ex_group = VGroup()
        for prompt_t, resp_t in examples:
            p_box = rounded_box(5.5, 0.65, fill_color=BLUE_DARK,
                                stroke_color=BLUE_MED, label=prompt_t, label_color=BLUE_LIGHT)
            r_box = rounded_box(5.5, 0.85, fill_color=GREEN_DARK,
                                stroke_color=GREEN_MED, label=resp_t, label_color=GREEN_LIGHT)
            r_box.next_to(p_box, DOWN, buff=0.1)
            ex_group.add(VGroup(p_box, r_box))

        ex_group.arrange(DOWN, buff=0.35)
        ex_group.scale_to_fit_height(5.0)
        ex_group.move_to(ORIGIN + DOWN * 0.2)

        self.play(LaggedStart(*[FadeIn(e) for e in ex_group], lag_ratio=0.3),
                  run_time=1.5)
        self.wait(1)
        self.fade_all(dataset_title, ex_group)

        # ── 3. Three phases of RLHF — position SFT in context ─────────────────
        phase_title = body_text("SFT is Phase 1 of 3 in RLHF:", color=WHITE)
        phase_title.to_edge(UP, buff=0.6)
        self.play(Write(phase_title), run_time=0.6)

        phases = [
            ("Phase 1\nSFT",          GREEN_MED,   "Imitate demos"),
            ("Phase 2\nReward Model", ORANGE_MED,  "Learn to judge"),
            ("Phase 3\nRL + PPO",     PURPLE_MED,  "Exceed demos"),
        ]
        phase_boxes = VGroup()
        for lbl, col, sub in phases:
            b = rounded_box(2.8, 1.0, fill_color=col + "22",
                            stroke_color=col, label=lbl, label_color=col)
            s = label_text(sub, color=GREY_LIGHT)
            s.next_to(b, DOWN, buff=0.2)
            phase_boxes.add(VGroup(b, s))

        phase_boxes.arrange(RIGHT, buff=0.8)
        phase_boxes.move_to(ORIGIN)

        phase_arrows = VGroup(*[
            Arrow(phase_boxes[i][0].get_right(), phase_boxes[i + 1][0].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.2)
            for i in range(2)
        ])

        # Highlight phase 1
        hl = SurroundingRectangle(phase_boxes[0], color=YELLOW_MED,
                                  buff=0.12, corner_radius=0.1, stroke_width=2)

        self.play(LaggedStart(*[FadeIn(b) for b in phase_boxes], lag_ratio=0.3),
                  run_time=1.2)
        self.play(LaggedStart(*[GrowArrow(a) for a in phase_arrows], lag_ratio=0.3),
                  run_time=0.7)
        self.play(Create(hl), run_time=0.5)

        key_note = label_text(
            "Source: Ouyang et al., InstructGPT (2022)  —  1.3B SFT model beats 175B base",
            color=GREY_MED,
        )
        key_note.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(key_note), run_time=0.5)
        self.wait(2)
