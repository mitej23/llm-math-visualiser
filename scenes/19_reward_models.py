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

        human_pref = rounded_box(3.0, 0.65, fill_color=str(YELLOW_MED) + "33",
                                 stroke_color=YELLOW_MED,
                                 label="Human: A > B  ✅", label_color=WHITE)
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
        trans_box  = rounded_box(3.0, 0.9,  fill_color=str(PURPLE_MED) + "22",
                                 stroke_color=PURPLE_MED,
                                 label="Transformer layers\n(same as base LLM)", label_color=WHITE)
        last_tok   = rounded_box(2.5, 0.65, fill_color=GREY_DARK,
                                 stroke_color=GREY_LIGHT,
                                 label="Last token hidden state", label_color=GREY_LIGHT)
        linear     = rounded_box(2.2, 0.65, fill_color=str(ORANGE_MED) + "33",
                                 stroke_color=ORANGE_MED,
                                 label="Linear → scalar", label_color=WHITE)
        score_box  = rounded_box(1.6, 0.65, fill_color=GREEN_DARK,
                                 stroke_color=GREEN_MED,
                                 label="Score: 7.3", label_color=WHITE)

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
        self.play(LaggedStart(*[Create(a) for a in arch_arrows], lag_ratio=0.1),
                  run_time=0.8)

        last_tok_note = label_text(
            "Last token has attended to everything — it summarises the full response",
            color=GREY_LIGHT,
        )
        last_tok_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(last_tok_note), run_time=0.5)
        self.wait(1.2)
        self.fade_all(arch_title, arch_chain, arch_arrows, last_tok_note)

        # ── 3. What a reward model is trained on — the data flow ──────────────
        data_title = body_text("Training data: human rater sees two responses, picks one",
                               color=WHITE)
        data_title.to_edge(UP, buff=0.6)
        self.play(Write(data_title), run_time=0.7)

        # Show the rater flow: prompt → two responses → label → training pair
        rater_prompt = rounded_box(5.5, 0.6, stroke_color=GREY_LIGHT,
                                   label="Prompt: \"Write a cover letter for a junior engineer role.\"",
                                   label_color=GREY_LIGHT)
        rater_prompt.shift(UP * 2.0)
        self.play(FadeIn(rater_prompt), run_time=0.5)

        rater_a = rounded_box(4.2, 1.0, fill_color=GREEN_DARK, stroke_color=GREEN_MED,
                              label="Response A:\nFormal, concise, specific examples",
                              label_color=GREEN_LIGHT)
        rater_a.shift(LEFT * 3.2 + UP * 0.5)

        rater_b = rounded_box(4.2, 1.0, fill_color=str(RED_MED) + "22",
                              stroke_color=RED_MED,
                              label="Response B:\nGeneric, padded with clichés",
                              label_color=WHITE)
        rater_b.shift(RIGHT * 3.2 + UP * 0.5)

        self.play(FadeIn(rater_a), FadeIn(rater_b), run_time=0.7)

        rater_box = rounded_box(2.5, 0.65, fill_color=str(YELLOW_MED) + "22",
                                stroke_color=YELLOW_MED,
                                label="Human Rater\nA is better ✅", label_color=WHITE)
        rater_box.shift(DOWN * 0.4)
        self.play(FadeIn(rater_box), run_time=0.5)

        pair_note = label_text(
            "Training pair: (prompt + A, prompt + B, label = A preferred)",
            color=GREY_LIGHT,
        )
        pair_note.shift(DOWN * 1.5)
        pair_arrow = Arrow(rater_box.get_bottom(), pair_note.get_top(),
                           color=GREY_MED, buff=0.05, stroke_width=1.5,
                           max_tip_length_to_length_ratio=0.2)
        self.play(Create(pair_arrow), run_time=0.4)
        self.play(FadeIn(pair_note), run_time=0.5)

        scale_note = label_text(
            "InstructGPT: 33,000 comparison pairs   |   LLaMA 2: ~1 million pairs",
            color=GREY_MED,
        )
        scale_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(scale_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(data_title, rater_prompt, rater_a, rater_b, rater_box,
                      pair_arrow, pair_note, scale_note)

        # ── 4. Bradley-Terry model ────────────────────────────────────────────
        bt_title = body_text("Bradley-Terry: preferences become probabilities", color=WHITE)
        bt_title.to_edge(UP, buff=0.6)
        self.play(Write(bt_title), run_time=0.7)

        # Show two response score boxes
        score_a = rounded_box(2.4, 0.85, fill_color=GREEN_DARK, stroke_color=GREEN_MED,
                              label="Response A\nscore = s_A", label_color=GREEN_LIGHT)
        score_a.shift(LEFT * 3.5 + UP * 0.8)

        score_b = rounded_box(2.4, 0.85, fill_color=str(RED_MED) + "22",
                              stroke_color=RED_MED,
                              label="Response B\nscore = s_B", label_color=WHITE)
        score_b.shift(RIGHT * 3.5 + UP * 0.8)

        self.play(FadeIn(score_a), FadeIn(score_b), run_time=0.7)

        # Show the sigmoid formula as labeled boxes
        formula_parts = [
            ("P( A preferred )", GREEN_MED),
            ("  =  ", GREY_LIGHT),
            ("sigmoid(", WHITE),
            ("s_A", GREEN_MED),
            ("  -  ", WHITE),
            ("s_B", RED_MED),
            (")", WHITE),
        ]
        formula_row = VGroup(*[body_text(t, color=c) for t, c in formula_parts])
        formula_row.arrange(RIGHT, buff=0.06)
        formula_row.move_to(ORIGIN + UP * 0.1)
        self.play(LaggedStart(*[FadeIn(p) for p in formula_row], lag_ratio=0.1),
                  run_time=1.0)

        # Show three example cases
        cases = [
            ("s_A = 4.0, s_B = 1.0", GREEN_MED, "sigmoid(3.0) = 95%  →  A almost certain"),
            ("s_A = 2.0, s_B = 2.0", YELLOW_MED, "sigmoid(0.0) = 50%  →  coin flip"),
            ("s_A = 1.0, s_B = 4.0", RED_MED,   "sigmoid(-3.0) = 5%  →  B almost certain"),
        ]
        case_rows = VGroup()
        for scores, col, result in cases:
            s_txt = label_text(scores, color=col)
            r_txt = label_text(result, color=WHITE)
            r_txt.next_to(s_txt, RIGHT, buff=0.5)
            case_rows.add(VGroup(s_txt, r_txt))

        case_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        case_rows.move_to(ORIGIN + DOWN * 1.4)
        self.play(LaggedStart(*[FadeIn(r) for r in case_rows], lag_ratio=0.3),
                  run_time=1.0)

        bt_note = label_text(
            "Loss = -log(sigmoid(r_winner - r_loser))  — push winner score above loser",
            color=ORANGE_MED,
        )
        bt_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(bt_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(bt_title, score_a, score_b, formula_row, case_rows, bt_note)

        # ── 5. Training the reward model ──────────────────────────────────────
        train_title = body_text("Training: base LLM + linear head, fine-tuned on comparisons",
                                color=WHITE)
        train_title.to_edge(UP, buff=0.6)
        self.play(Write(train_title), run_time=0.7)

        # Show the pipeline: base LLM → add head → fine-tune → outputs scalar
        step_data = [
            ("Base LLM\n(SFT model)", BLUE_MED,    "Start from SFT weights"),
            ("Add linear\nhead", ORANGE_MED,        "hidden_dim → 1 scalar"),
            ("Fine-tune on\ncomparisons", PURPLE_MED, "want r(A) > r(B) when A preferred"),
            ("Reward Model\n✅", GREEN_MED,         "scores any (prompt, response)"),
        ]
        step_boxes = VGroup()
        for lbl, col, note in step_data:
            b = rounded_box(2.6, 0.85, fill_color=str(col) + "22",
                            stroke_color=col, label=lbl, label_color=WHITE)
            n = label_text(note, color=GREY_LIGHT)
            n.next_to(b, DOWN, buff=0.18)
            step_boxes.add(VGroup(b, n))

        step_boxes.arrange(RIGHT, buff=0.55)
        step_boxes.scale_to_fit_width(13)
        step_boxes.move_to(ORIGIN + DOWN * 0.1)

        step_arrows = VGroup(*[
            Arrow(step_boxes[i][0].get_right(), step_boxes[i + 1][0].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.18)
            for i in range(3)
        ])

        self.play(LaggedStart(*[FadeIn(b) for b in step_boxes], lag_ratio=0.2),
                  run_time=1.3)
        self.play(LaggedStart(*[Create(a) for a in step_arrows], lag_ratio=0.15),
                  run_time=0.6)

        loss_note = label_text(
            "Training loss: -log(sigmoid(r_winner - r_loser))  updates weights each batch",
            color=ORANGE_MED,
        )
        loss_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(loss_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(train_title, step_boxes, step_arrows, loss_note)

        # ── 6. Reward model generalisation ───────────────────────────────────
        gen_title = body_text("Generalisation: trained on 100k pairs, scores millions of responses",
                              color=WHITE)
        gen_title.to_edge(UP, buff=0.6)
        self.play(Write(gen_title), run_time=0.7)

        # Show training data size vs deployment scale
        train_box = rounded_box(3.8, 0.85, fill_color=GREEN_DARK, stroke_color=GREEN_MED,
                                label="Training data\n~33k–100k comparisons", label_color=GREEN_LIGHT)
        train_box.shift(LEFT * 3.5 + UP * 1.0)

        deploy_box = rounded_box(3.8, 0.85, fill_color=str(ORANGE_MED) + "22",
                                 stroke_color=ORANGE_MED,
                                 label="RL training\nmillions of novel responses", label_color=WHITE)
        deploy_box.shift(RIGHT * 3.5 + UP * 1.0)

        gap_arrow = Arrow(train_box.get_right(), deploy_box.get_left(),
                          color=YELLOW_MED, buff=0.05, stroke_width=2,
                          max_tip_length_to_length_ratio=0.18)
        gap_label = label_text("must generalise!", color=YELLOW_MED)
        gap_label.next_to(gap_arrow, UP, buff=0.15)

        self.play(FadeIn(train_box), FadeIn(deploy_box), run_time=0.7)
        self.play(Create(gap_arrow), FadeIn(gap_label), run_time=0.5)

        # Failure modes
        fail_data = [
            ("Length bias", ORANGE_MED, "Longer responses score higher regardless of quality"),
            ("Style bias",  BLUE_MED,   "Bullet points score higher even when prose is better"),
            ("Out-of-dist", RED_MED,    "Reward model unreliable on novel topics/styles"),
        ]
        fail_rows = VGroup()
        for lbl, col, desc in fail_data:
            l_txt = label_text(lbl, color=col)
            d_txt = label_text(desc, color=GREY_LIGHT)
            d_txt.next_to(l_txt, RIGHT, buff=0.4)
            fail_rows.add(VGroup(l_txt, d_txt))

        fail_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        fail_rows.move_to(ORIGIN + DOWN * 1.3)
        fail_header = label_text("Generalisation failure modes:", color=WHITE)
        fail_header.next_to(fail_rows, UP, buff=0.3)

        self.play(FadeIn(fail_header), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r) for r in fail_rows], lag_ratio=0.25),
                  run_time=0.9)
        self.wait(1.5)
        self.fade_all(gen_title, train_box, deploy_box, gap_arrow, gap_label,
                      fail_header, fail_rows)

        # ── 7. Reward model failures — real examples ──────────────────────────
        fail_title = body_text("Reward model failures: what reward hacking looks like",
                               color=RED_MED)
        fail_title.to_edge(UP, buff=0.6)
        self.play(Write(fail_title), run_time=0.7)

        examples = [
            ("Length Hacking", ORANGE_MED,
             "Model pads responses with unnecessary caveats to increase length → higher score"),
            ("Sycophancy", YELLOW_MED,
             "User: \"Einstein failed maths, right?\"  Model: \"Absolutely correct!\" (wrong)"),
            ("Format Gaming", BLUE_MED,
             "Simple question → unnecessary headers + bullet points → scores well"),
            ("Confident Errors", RED_MED,
             "States false facts confidently — rated higher than honest uncertainty"),
        ]

        ex_rows = VGroup()
        for name, col, desc in examples:
            name_txt = label_text(name, color=col)
            desc_txt = label_text(desc, color=GREY_LIGHT)
            desc_txt.next_to(name_txt, RIGHT, buff=0.35)
            ex_rows.add(VGroup(name_txt, desc_txt))

        ex_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        ex_rows.scale_to_fit_width(12.5)
        ex_rows.move_to(ORIGIN + DOWN * 0.2)

        self.play(LaggedStart(*[FadeIn(r) for r in ex_rows], lag_ratio=0.25),
                  run_time=1.2)

        rm_note = label_text(
            "The reward model is a proxy — it captures annotator preferences, not ground truth quality",
            color=GREY_MED,
        )
        rm_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(rm_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(fail_title, ex_rows, rm_note)

        # ── 8. Reward hacking warning (original section, kept as finale) ──────
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
            b = rounded_box(2.8, 1.0, fill_color=str(col) + "22",
                            stroke_color=col, label=lbl, label_color=WHITE)
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
                                   label_color=WHITE)
        solution_box.to_edge(DOWN, buff=0.5)

        self.play(LaggedStart(*[FadeIn(b) for b in hack_boxes], lag_ratio=0.3),
                  run_time=1.1)
        self.play(LaggedStart(*[Create(a) for a in hack_arrows], lag_ratio=0.3),
                  run_time=0.6)
        self.play(FadeIn(solution_box), run_time=0.6)

        source_lbl = label_text(
            "Source: Lambert et al., HuggingFace RLHF Blog (2022)",
            color=GREY_MED,
        )
        source_lbl.next_to(solution_box, DOWN, buff=0.15)
        self.play(FadeIn(source_lbl), run_time=0.4)
        self.wait(2)
        self.fade_all(hack_title, hack_boxes, hack_arrows, solution_box, source_lbl)
