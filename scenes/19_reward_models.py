"""
Scene 19 — Reward Models
Run: manim -pql 19_reward_models.py RewardModelsScene
"""

from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class RewardModelsScene(LLMScene):
    def construct(self):
        title = self.show_title("Reward Models", "Teaching a Machine to Judge")
        self.wait(0.5)
        self.fade_all(title)

        # ── 1. Why rankings, not scores ────────────────────────────────────────
        rank_title = body_text("Human rankings are more reliable than scores:", color=WHITE)
        rank_title.to_edge(UP, buff=0.6)
        self.play(Write(rank_title), run_time=0.7)

        prompt_box = rounded_box(6.0, 0.65, stroke_color=GREY_MED,
                                 label='"Explain black holes to a child."',
                                 label_color=GREY_LIGHT)
        prompt_box.shift(UP * 1.5)
        self.play(FadeIn(prompt_box), run_time=0.5)

        resp_a = rounded_box(4.0, 1.1, fill_color=GREEN_DARK, stroke_color=GREEN_MED,
                             label="Response A:\n\"Imagine a cosmic vacuum cleaner...\"",
                             label_color=GREEN_LIGHT)
        resp_a.shift(LEFT * 3 + DOWN * 0.2)

        resp_b = rounded_box(4.0, 1.1, fill_color=BLUE_DARK, stroke_color=BLUE_MED,
                             label="Response B:\n\"A singularity of infinite density...\"",
                             label_color=BLUE_LIGHT)
        resp_b.shift(RIGHT * 3 + DOWN * 0.2)

        human_pref = rounded_box(3.0, 0.65, fill_color=YELLOW_MED + "33",
                                 stroke_color=YELLOW_MED,
                                 label="Human: A > B  ✅", label_color=YELLOW_MED)
        human_pref.shift(DOWN * 1.8)

        self.play(FadeIn(resp_a), FadeIn(resp_b), run_time=0.7)
        self.play(FadeIn(human_pref), run_time=0.5)
        self.wait(1)
        self.fade_all(rank_title, prompt_box, resp_a, resp_b, human_pref)

        # ── 2. Reward model architecture ──────────────────────────────────────
        arch_title = body_text("Architecture: LLM + scalar output head", color=WHITE)
        arch_title.to_edge(UP, buff=0.6)
        self.play(Write(arch_title), run_time=0.7)

        input_box  = rounded_box(3.0, 0.65, stroke_color=BLUE_MED,
                                 label="[prompt + response]", label_color=BLUE_LIGHT)
        trans_box  = rounded_box(3.0, 0.9,  fill_color=PURPLE_MED + "22",
                                 stroke_color=PURPLE_MED,
                                 label="Transformer layers\n(same as base LLM)", label_color=PURPLE_MED)
        last_tok   = rounded_box(2.5, 0.65, fill_color=GREY_DARK,
                                 stroke_color=GREY_LIGHT,
                                 label="Last token hidden state", label_color=GREY_LIGHT)
        linear     = rounded_box(2.2, 0.65, fill_color=ORANGE_MED + "33",
                                 stroke_color=ORANGE_MED,
                                 label="Linear → scalar", label_color=ORANGE_MED)
        score_box  = rounded_box(1.6, 0.65, fill_color=GREEN_DARK,
                                 stroke_color=GREEN_MED,
                                 label="Score: 7.3", label_color=GREEN_MED)

        arch_chain = VGroup(input_box, trans_box, last_tok, linear, score_box)
        arch_chain.arrange(RIGHT, buff=0.5)
        arch_chain.scale_to_fit_width(13)
        arch_chain.move_to(ORIGIN)

        arch_arrows = VGroup(*[
            Arrow(arch_chain[i].get_right(), arch_chain[i + 1].get_left(),
                  color=GREY_MED, buff=0.04, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.18)
            for i in range(len(arch_chain) - 1)
        ])

        self.play(LaggedStart(*[FadeIn(b) for b in arch_chain], lag_ratio=0.15),
                  run_time=1.3)
        self.play(LaggedStart(*[GrowArrow(a) for a in arch_arrows], lag_ratio=0.1),
                  run_time=0.8)

        last_tok_note = label_text(
            "Last token has attended to everything — it summarises the full response",
            color=GREY_LIGHT,
        )
        last_tok_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(last_tok_note), run_time=0.5)
        self.wait(1.2)
        self.fade_all(arch_title, arch_chain, arch_arrows, last_tok_note)

        # ── 3. Reward hacking warning ─────────────────────────────────────────
        hack_title = body_text("Danger: Reward Hacking", color=RED_MED)
        hack_title.to_edge(UP, buff=0.6)
        self.play(Write(hack_title), run_time=0.6)

        hack_flow = [
            ("RL Policy",        BLUE_MED,   "tries to get high scores"),
            ("Learns reward\nmodel quirks", ORANGE_MED, "not genuine quality"),
            ("Gibberish that\nscores high!", RED_MED,   "reward hacking 💥"),
        ]
        hack_boxes = VGroup()
        for lbl, col, sub in hack_flow:
            b = rounded_box(2.8, 1.0, fill_color=col + "22",
                            stroke_color=col, label=lbl, label_color=col)
            s = label_text(sub, color=GREY_LIGHT)
            s.next_to(b, DOWN, buff=0.2)
            hack_boxes.add(VGroup(b, s))

        hack_boxes.arrange(RIGHT, buff=1.0)
        hack_boxes.move_to(ORIGIN + DOWN * 0.1)
        hack_arrows = VGroup(*[
            Arrow(hack_boxes[i][0].get_right(), hack_boxes[i + 1][0].get_left(),
                  color=RED_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.2)
            for i in range(2)
        ])

        solution_box = rounded_box(5.5, 0.65, fill_color=GREEN_DARK,
                                   stroke_color=GREEN_MED,
                                   label="Solution: KL penalty — don't drift too far from original model",
                                   label_color=GREEN_MED)
        solution_box.to_edge(DOWN, buff=0.5)

        self.play(LaggedStart(*[FadeIn(b) for b in hack_boxes], lag_ratio=0.3),
                  run_time=1.1)
        self.play(LaggedStart(*[GrowArrow(a) for a in hack_arrows], lag_ratio=0.3),
                  run_time=0.6)
        self.play(FadeIn(solution_box), run_time=0.6)

        source_lbl = label_text(
            "Source: Lambert et al., HuggingFace RLHF Blog (2022)",
            color=GREY_MED,
        )
        source_lbl.next_to(solution_box, DOWN, buff=0.15)
        self.play(FadeIn(source_lbl), run_time=0.4)
        self.wait(2)
