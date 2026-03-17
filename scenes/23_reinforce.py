"""
Scene 23 — REINFORCE Algorithm
Run: manim -pql 23_reinforce.py REINFORCEScene
"""

from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class REINFORCEScene(LLMScene):
    def construct(self):

        # ── 1. Title card ──────────────────────────────────────────────────────
        title = self.show_title("REINFORCE", "Policy Gradient from Scratch")
        self.wait(0.5)

        tagline = label_text(
            "The simplest RL algorithm — and the foundation of everything else",
            color=GREY_LIGHT,
        )
        tagline.next_to(title, DOWN, buff=0.45)
        self.play(FadeIn(tagline), run_time=0.6)
        self.wait(1.5)
        self.fade_all(title, tagline)
        self.wait(1.0)

        # ── 2. Core idea: good actions → up, bad → down ───────────────────────
        core_title = body_text("Core Idea: nudge action probabilities using reward", color=WHITE)
        core_title.to_edge(UP, buff=0.6)
        self.play(Write(core_title), run_time=0.7)

        good_box = rounded_box(
            4.8, 1.5,
            fill_color=str(GREEN_MED) + "22",
            stroke_color=GREEN_MED,
            label="High reward episode\nActions were good\nIncrease their probability",
            label_color=GREEN_LIGHT,
        )
        good_box.move_to(LEFT * 3.3 + DOWN * 0.3)

        bad_box = rounded_box(
            4.8, 1.5,
            fill_color=str(RED_MED) + "22",
            stroke_color=RED_MED,
            label="Low reward episode\nActions were poor\nDecrease their probability",
            label_color=WHITE,
        )
        bad_box.move_to(RIGHT * 3.3 + DOWN * 0.3)

        up_arrow = Arrow(
            good_box.get_top() + LEFT * 0.6,
            good_box.get_top() + LEFT * 0.6 + UP * 0.9,
            color=GREEN_MED, buff=0.0, stroke_width=3,
            max_tip_length_to_length_ratio=0.25,
        )
        down_arrow = Arrow(
            bad_box.get_top() + RIGHT * 0.6,
            bad_box.get_top() + RIGHT * 0.6 + UP * 0.9,
            color=RED_MED, buff=0.0, stroke_width=3,
            max_tip_length_to_length_ratio=0.25,
        )
        # flip the bad arrow to point down
        down_arrow = Arrow(
            bad_box.get_bottom() + RIGHT * 0.6,
            bad_box.get_bottom() + RIGHT * 0.6 + DOWN * 0.9,
            color=RED_MED, buff=0.0, stroke_width=3,
            max_tip_length_to_length_ratio=0.25,
        )

        rule_lbl = label_text(
            "Every token in the episode gets nudged — up or down — by the same reward signal",
            color=GREY_LIGHT,
        )
        rule_lbl.to_edge(DOWN, buff=0.45)

        self.play(FadeIn(good_box), run_time=0.6)
        self.play(FadeIn(bad_box), run_time=0.6)
        self.play(Create(up_arrow), Create(down_arrow), run_time=0.7)
        self.play(FadeIn(rule_lbl), run_time=0.5)
        self.wait(1.8)
        self.fade_all(core_title, good_box, bad_box, up_arrow, down_arrow, rule_lbl)
        self.wait(1.0)

        # ── 3. Gradient estimator: action → reward → gradient → weights ────────
        grad_title = body_text(
            "The Update: log prob × reward → gradient → weight change",
            color=WHITE,
        )
        grad_title.to_edge(UP, buff=0.6)
        self.play(Write(grad_title), run_time=0.7)

        step_data = [
            ("Take\nAction",      BLUE_MED,    "Token chosen\nfrom vocab"),
            ("Observe\nReward",   ORANGE_MED,  "Reward model\nscores response"),
            ("Compute\nGradient", YELLOW_MED,  "log P(action)\n× reward"),
            ("Update\nWeights",   GREEN_MED,   "Gradient ascent\non policy"),
        ]

        step_boxes = VGroup()
        for lbl, col, note in step_data:
            b = rounded_box(
                2.4, 0.85,
                fill_color=str(col) + "22",
                stroke_color=col,
                label=lbl,
                label_color=WHITE,
            )
            n = label_text(note, color=GREY_LIGHT)
            n.next_to(b, DOWN, buff=0.2)
            step_boxes.add(VGroup(b, n))

        step_boxes.arrange(RIGHT, buff=0.55)
        step_boxes.scale_to_fit_width(13.0)
        step_boxes.move_to(ORIGIN + UP * 0.35)

        step_arrows = VGroup(*[
            Arrow(
                step_boxes[i][0].get_right(),
                step_boxes[i + 1][0].get_left(),
                color=GREY_MED, buff=0.05, stroke_width=1.5,
                max_tip_length_to_length_ratio=0.18,
            )
            for i in range(3)
        ])

        loop_back = CurvedArrow(
            step_boxes[3][0].get_bottom(),
            step_boxes[0][0].get_bottom(),
            angle=-TAU / 6,
            color=PURPLE_MED,
            stroke_width=1.5,
        )
        loop_lbl = label_text("next episode", color=PURPLE_MED)
        loop_lbl.next_to(loop_back, DOWN, buff=0.1)

        self.play(
            LaggedStart(*[FadeIn(b) for b in step_boxes], lag_ratio=0.2),
            run_time=1.2,
        )
        self.play(
            LaggedStart(*[Create(a) for a in step_arrows], lag_ratio=0.15),
            run_time=0.6,
        )
        self.play(Create(loop_back), FadeIn(loop_lbl), run_time=0.6)
        self.wait(1.8)
        self.fade_all(grad_title, step_boxes, step_arrows, loop_back, loop_lbl)
        self.wait(1.0)

        # ── 4. High variance: same action, many different rewards ──────────────
        var_title = body_text(
            "High Variance: same prompt, wildly different rewards",
            color=WHITE,
        )
        var_title.to_edge(UP, buff=0.6)
        self.play(Write(var_title), run_time=0.7)

        prompt_box = rounded_box(
            3.2, 0.7,
            fill_color=str(BLUE_MED) + "22",
            stroke_color=BLUE_MED,
            label='Prompt: "Explain gravity"',
            label_color=BLUE_LIGHT,
        )
        prompt_box.move_to(LEFT * 4.2 + UP * 0.5)

        rewards = [0.9, 0.2, 0.7, 0.1, 0.8, 0.4]
        reward_colors = [GREEN_MED, RED_MED, GREEN_MED, RED_MED, GREEN_MED, ORANGE_MED]
        reward_labels = ["0.9", "0.2", "0.7", "0.1", "0.8", "0.4"]

        reward_boxes = VGroup()
        for i, (val, col, lbl) in enumerate(zip(rewards, reward_colors, reward_labels)):
            b = rounded_box(
                1.5, 0.55,
                fill_color=str(col) + "22",
                stroke_color=col,
                label=f"Run {i+1}  r={lbl}",
                label_color=WHITE,
            )
            reward_boxes.add(b)

        reward_boxes.arrange(DOWN, buff=0.18)
        reward_boxes.move_to(RIGHT * 2.8 + UP * 0.0)

        spread_arrows = VGroup(*[
            Arrow(
                prompt_box.get_right(),
                b.get_left(),
                color=GREY_MED, buff=0.05, stroke_width=1.2,
                max_tip_length_to_length_ratio=0.15,
            )
            for b in reward_boxes
        ])

        noise_lbl = label_text(
            "Same prompt → 6 different trajectories → 6 different rewards → gradient is noisy",
            color=ORANGE_MED,
        )
        noise_lbl.to_edge(DOWN, buff=0.45)

        self.play(FadeIn(prompt_box), run_time=0.5)
        self.play(
            LaggedStart(*[Create(a) for a in spread_arrows], lag_ratio=0.1),
            LaggedStart(*[FadeIn(b) for b in reward_boxes], lag_ratio=0.1),
            run_time=1.4,
        )
        self.play(FadeIn(noise_lbl), run_time=0.5)
        self.wait(1.8)
        self.fade_all(var_title, prompt_box, spread_arrows, reward_boxes, noise_lbl)
        self.wait(1.0)

        # ── 5. Baseline subtraction ────────────────────────────────────────────
        base_title = body_text(
            'Baseline: "was this better than average?"',
            color=WHITE,
        )
        base_title.to_edge(UP, buff=0.6)
        self.play(Write(base_title), run_time=0.7)

        # Three columns: raw reward | baseline | adjusted
        col_headers = VGroup(
            body_text("Raw Reward", color=GREY_LIGHT),
            body_text("Baseline (avg)", color=YELLOW_MED),
            body_text("Adjusted Signal", color=WHITE),
        )
        col_headers.arrange(RIGHT, buff=2.2)
        col_headers.move_to(UP * 2.5)

        rows_data = [
            (0.9, 0.55, +0.35, GREEN_MED),
            (0.2, 0.55, -0.35, RED_MED),
            (0.7, 0.55, +0.15, GREEN_MED),
            (0.1, 0.55, -0.45, RED_MED),
            (0.8, 0.55, +0.25, GREEN_MED),
            (0.4, 0.55, -0.15, RED_MED),
        ]

        row_groups = VGroup()
        for i, (raw, avg, adj, col) in enumerate(rows_data):
            sign = "+" if adj >= 0 else ""
            raw_t  = label_text(f"{raw:.1f}", color=GREY_LIGHT)
            base_t = label_text(f"{avg:.2f}", color=YELLOW_MED)
            adj_t  = label_text(f"{sign}{adj:.2f}", color=col)

            row = VGroup(raw_t, base_t, adj_t)
            # align with col_headers
            raw_t.move_to(col_headers[0].get_center())
            base_t.move_to(col_headers[1].get_center())
            adj_t.move_to(col_headers[2].get_center())
            row.shift(DOWN * (1.0 + i * 0.42))
            row_groups.add(row)

        baseline_note = label_text(
            "Subtracting the baseline does NOT bias the gradient — it only reduces noise",
            color=GREEN_LIGHT,
        )
        baseline_note.to_edge(DOWN, buff=0.45)

        self.play(FadeIn(col_headers), run_time=0.5)
        self.play(
            LaggedStart(*[FadeIn(r) for r in row_groups], lag_ratio=0.12),
            run_time=1.2,
        )
        self.play(FadeIn(baseline_note), run_time=0.5)
        self.wait(1.8)
        self.fade_all(base_title, col_headers, row_groups, baseline_note)
        self.wait(1.0)

        # ── 6. LLM walkthrough: token-by-token, reward at end ─────────────────
        llm_title = body_text(
            "LLM Episode: every token is an action, reward arrives at the end",
            color=WHITE,
        )
        llm_title.to_edge(UP, buff=0.6)
        self.play(Write(llm_title), run_time=0.7)

        prompt_lbl = label_text('Prompt: "What is 3 + 5?"', color=BLUE_LIGHT)
        prompt_lbl.move_to(LEFT * 4.8 + UP * 1.5)

        token_data = [
            ("The",    GREY_MED,   "filler"),
            ("answer", GREY_MED,   "filler"),
            ("is",     GREY_MED,   "filler"),
            ("8",      GREEN_MED,  "KEY"),
            (".",      GREY_MED,   "filler"),
        ]

        token_boxes = VGroup()
        for tok, col, role in token_data:
            b = rounded_box(
                1.3, 0.65,
                fill_color=str(col) + "22",
                stroke_color=col,
                label=tok,
                label_color=WHITE,
            )
            token_boxes.add(b)

        token_boxes.arrange(RIGHT, buff=0.3)
        token_boxes.move_to(UP * 0.6)

        token_arrows = VGroup(*[
            Arrow(
                token_boxes[i].get_right(),
                token_boxes[i + 1].get_left(),
                color=GREY_MED, buff=0.05, stroke_width=1.5,
                max_tip_length_to_length_ratio=0.2,
            )
            for i in range(len(token_boxes) - 1)
        ])

        reward_arrive = rounded_box(
            2.8, 0.7,
            fill_color=str(ORANGE_MED) + "22",
            stroke_color=ORANGE_MED,
            label="Reward  r = +1.0\n(verifier: correct!)",
            label_color=WHITE,
        )
        reward_arrive.move_to(RIGHT * 4.2 + DOWN * 0.8)

        reward_arrow = Arrow(
            token_boxes[-1].get_bottom(),
            reward_arrive.get_top(),
            color=ORANGE_MED, buff=0.05, stroke_width=2.0,
            max_tip_length_to_length_ratio=0.18,
        )

        credit_note = label_text(
            "All 5 tokens receive the same signal — even 'The', 'is', and '.'",
            color=YELLOW_MED,
        )
        credit_note.move_to(DOWN * 2.4)

        key_note = label_text(
            "The critical token '8' is not treated differently — this is the credit assignment problem",
            color=RED_MED,
        )
        key_note.to_edge(DOWN, buff=0.35)

        self.play(FadeIn(prompt_lbl), run_time=0.4)
        self.play(
            LaggedStart(*[FadeIn(b) for b in token_boxes], lag_ratio=0.15),
            run_time=0.9,
        )
        self.play(
            LaggedStart(*[Create(a) for a in token_arrows], lag_ratio=0.1),
            run_time=0.5,
        )
        self.play(Create(reward_arrow), FadeIn(reward_arrive), run_time=0.6)
        self.play(FadeIn(credit_note), run_time=0.5)
        self.play(FadeIn(key_note), run_time=0.5)
        self.wait(1.8)
        self.fade_all(
            llm_title, prompt_lbl, token_boxes, token_arrows,
            reward_arrow, reward_arrive, credit_note, key_note,
        )
        self.wait(1.0)

        # ── 7. Problems: sparse reward & credit assignment ─────────────────────
        prob_title = body_text(
            "Two Hard Problems in REINFORCE",
            color=WHITE,
        )
        prob_title.to_edge(UP, buff=0.6)
        self.play(Write(prob_title), run_time=0.7)

        sparse_box = rounded_box(
            5.2, 2.0,
            fill_color=str(RED_MED) + "22",
            stroke_color=RED_MED,
            label="Sparse Reward\n\nReward only arrives at the end\nof the full response.\n50 tokens taken before any feedback.\nEarly tokens shaped with no real signal.",
            label_color=WHITE,
        )
        sparse_box.move_to(LEFT * 3.3 + DOWN * 0.3)

        credit_box = rounded_box(
            5.2, 2.0,
            fill_color=str(ORANGE_MED) + "22",
            stroke_color=ORANGE_MED,
            label="Credit Assignment\n\nEvery token blames/credits equally.\nThe critical word 'because' and the\nfiller word 'the' are treated the same.\nSignal is diluted across all tokens.",
            label_color=WHITE,
        )
        credit_box.move_to(RIGHT * 3.3 + DOWN * 0.3)

        fix_note = label_text(
            "PPO's value function and GRPO's group baseline both attack these two problems",
            color=GREEN_LIGHT,
        )
        fix_note.to_edge(DOWN, buff=0.45)

        self.play(FadeIn(sparse_box), run_time=0.6)
        self.play(FadeIn(credit_box), run_time=0.6)
        self.play(FadeIn(fix_note), run_time=0.5)
        self.wait(2.0)
        self.fade_all(prob_title, sparse_box, credit_box, fix_note)
        self.wait(1.0)

        # ── 8. What REINFORCE gets right vs fails at ───────────────────────────
        eval_title = body_text(
            "What REINFORCE Gets Right — and Where It Fails",
            color=WHITE,
        )
        eval_title.to_edge(UP, buff=0.6)
        self.play(Write(eval_title), run_time=0.7)

        pros = [
            ("Simple to implement", GREEN_MED),
            ("Unbiased gradient", GREEN_MED),
            ("Works with any reward", GREEN_MED),
            ("Foundation for PPO & GRPO", GREEN_MED),
        ]
        cons = [
            ("Very high variance", RED_MED),
            ("Slow to converge", RED_MED),
            ("No credit assignment", RED_MED),
            ("Poor sample efficiency", RED_MED),
        ]

        pro_boxes = VGroup()
        for txt, col in pros:
            b = rounded_box(
                5.0, 0.6,
                fill_color=str(col) + "22",
                stroke_color=col,
                label="+ " + txt,
                label_color=WHITE,
            )
            pro_boxes.add(b)

        pro_boxes.arrange(DOWN, buff=0.2)
        pro_boxes.move_to(LEFT * 3.3 + DOWN * 0.3)

        con_boxes = VGroup()
        for txt, col in cons:
            b = rounded_box(
                5.0, 0.6,
                fill_color=str(col) + "22",
                stroke_color=col,
                label="- " + txt,
                label_color=WHITE,
            )
            con_boxes.add(b)

        con_boxes.arrange(DOWN, buff=0.2)
        con_boxes.move_to(RIGHT * 3.3 + DOWN * 0.3)

        verdict = label_text(
            "Elegant and correct — but high variance makes it too slow for large LLM training runs",
            color=GREY_LIGHT,
        )
        verdict.to_edge(DOWN, buff=0.45)

        self.play(
            LaggedStart(*[FadeIn(b) for b in pro_boxes], lag_ratio=0.12),
            run_time=0.9,
        )
        self.play(
            LaggedStart(*[FadeIn(b) for b in con_boxes], lag_ratio=0.12),
            run_time=0.9,
        )
        self.play(FadeIn(verdict), run_time=0.5)
        self.wait(2.0)
        self.fade_all(eval_title, pro_boxes, con_boxes, verdict)
        self.wait(1.0)

        # ── 9. Real numbers: batch sizes, iterations ───────────────────────────
        num_title = body_text(
            "REINFORCE in Practice — Real Numbers",
            color=WHITE,
        )
        num_title.to_edge(UP, buff=0.6)
        self.play(Write(num_title), run_time=0.7)

        stat_data = [
            ("Rollout batch size",    "64 – 256 prompts per gradient step",   BLUE_LIGHT),
            ("Tokens per batch",      "64 prompts × ~100 tokens = ~6,400",     BLUE_LIGHT),
            ("Gradient steps",        "500 – 5,000 total for a typical run",   YELLOW_MED),
            ("Learning rate",         "1e-6 to 1e-5  (very small — unstable)", ORANGE_MED),
            ("Reward normalisation",  "Zero-mean, unit variance per batch",    GREEN_LIGHT),
            ("KL penalty",            "Added to reward: r_total = r - λ × KL", RED_MED),
            ("Wall-clock time (1B)",  "~17 hours on 8×A100 for 2,000 steps",  GREY_LIGHT),
            ("vs PPO efficiency",     "PPO reaches same result in ~6 hours",   GREY_LIGHT),
        ]

        stat_rows = VGroup()
        for label_str, value_str, col in stat_data:
            lbl = label_text(label_str, color=col)
            val = label_text(value_str, color=GREY_LIGHT)
            row = VGroup(lbl, val)
            lbl.move_to(LEFT * 3.5)
            val.move_to(RIGHT * 1.5)
            stat_rows.add(row)

        stat_rows.arrange(DOWN, buff=0.28)
        stat_rows.move_to(ORIGIN + DOWN * 0.1)
        stat_rows.scale_to_fit_height(5.5)

        self.play(
            LaggedStart(*[FadeIn(r) for r in stat_rows], lag_ratio=0.1),
            run_time=1.6,
        )
        self.wait(2.0)
        self.fade_all(num_title, stat_rows)
        self.wait(1.0)

        # ── 10. Summary: why we need PPO / GRPO next ───────────────────────────
        sum_title = body_text(
            "Summary — REINFORCE and Why We Need What Comes Next",
            color=WHITE,
        )
        sum_title.to_edge(UP, buff=0.6)
        self.play(Write(sum_title), run_time=0.7)

        reinforce_box = rounded_box(
            4.6, 2.2,
            fill_color=str(BLUE_MED) + "22",
            stroke_color=BLUE_MED,
            label="REINFORCE\n\nSimple. Unbiased.\nHigh variance.\nCredit blind.\nThrows away rollouts.",
            label_color=BLUE_LIGHT,
        )
        reinforce_box.move_to(LEFT * 4.0 + DOWN * 0.3)

        ppo_box = rounded_box(
            4.6, 2.2,
            fill_color=str(PURPLE_MED) + "22",
            stroke_color=PURPLE_MED,
            label="PPO\n\n+ Advantage estimation\n+ Clipped updates\n+ Reuse rollouts\n= Much lower variance",
            label_color=WHITE,
        )
        ppo_box.move_to(RIGHT * 0.5 + DOWN * 0.3)

        grpo_box = rounded_box(
            3.8, 2.2,
            fill_color=str(GREEN_MED) + "22",
            stroke_color=GREEN_MED,
            label="GRPO\n\n+ Group baseline\nNo value network\nSimpler than PPO\nDeepSeek-R1",
            label_color=GREEN_LIGHT,
        )
        grpo_box.move_to(RIGHT * 4.8 + DOWN * 0.3)

        arrow_rp = Arrow(
            reinforce_box.get_right(),
            ppo_box.get_left(),
            color=GREY_MED, buff=0.05, stroke_width=2.0,
            max_tip_length_to_length_ratio=0.18,
        )
        arrow_pg = Arrow(
            ppo_box.get_right(),
            grpo_box.get_left(),
            color=GREY_MED, buff=0.05, stroke_width=2.0,
            max_tip_length_to_length_ratio=0.18,
        )

        foundation_lbl = label_text(
            "REINFORCE is the root — every modern LLM RL algorithm is a branch of this tree",
            color=YELLOW_MED,
        )
        foundation_lbl.to_edge(DOWN, buff=0.45)

        self.play(FadeIn(reinforce_box), run_time=0.5)
        self.play(Create(arrow_rp), FadeIn(ppo_box), run_time=0.6)
        self.play(Create(arrow_pg), FadeIn(grpo_box), run_time=0.6)
        self.play(FadeIn(foundation_lbl), run_time=0.5)
        self.wait(2.5)
        self.fade_all(
            sum_title, reinforce_box, ppo_box, grpo_box,
            arrow_rp, arrow_pg, foundation_lbl,
        )
        self.wait(1.0)
