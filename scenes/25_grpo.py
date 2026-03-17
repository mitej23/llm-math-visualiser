"""
Scene 25 — GRPO: Group Relative Policy Optimization
Run: manim -pql 25_grpo.py GRPOScene
"""
from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class GRPOScene(LLMScene):
    def construct(self):

        # ── 1. Title ──────────────────────────────────────────────────────────
        title = self.show_title("GRPO", "Group Relative Policy Optimization")
        self.wait(0.5)
        self.fade_all(title)
        self.wait(1.0)

        # ── 2. Problem: PPO needs a critic — expensive and unstable ───────────
        prob_title = body_text("Problem: PPO requires a critic network the same size as the policy",
                               color=WHITE)
        prob_title.to_edge(UP, buff=0.6)
        self.play(Write(prob_title), run_time=0.7)

        ppo_models = [
            ("RL Policy\n(trained)",        BLUE_MED,    "~7B params\nfull gradients"),
            ("Reference\nPolicy (frozen)",  GREY_LIGHT,  "~7B params\nno gradients"),
            ("Reward\nModel",               ORANGE_MED,  "~3B params\nscores responses"),
            ("Critic /\nValue Model",       RED_MED,     "~7B params\nestimate V(state)"),
        ]
        model_boxes = VGroup()
        for lbl, col, note in ppo_models:
            b = rounded_box(2.3, 0.85, fill_color=str(col) + "22",
                            stroke_color=col, label=lbl, label_color=col)
            n = label_text(note, color=GREY_LIGHT)
            n.next_to(b, DOWN, buff=0.18)
            model_boxes.add(VGroup(b, n))

        model_boxes.arrange(RIGHT, buff=0.55)
        model_boxes.scale_to_fit_width(13)
        model_boxes.move_to(ORIGIN + UP * 0.3)

        self.play(LaggedStart(*[FadeIn(b) for b in model_boxes], lag_ratio=0.2),
                  run_time=1.2)

        critic_highlight = SurroundingRectangle(model_boxes[3], color=RED_MED,
                                                buff=0.12, corner_radius=0.1)
        critic_lbl = label_text("This is the problem — same size as policy!", color=RED_MED)
        critic_lbl.to_edge(DOWN, buff=0.55)

        self.play(Create(critic_highlight), run_time=0.5)
        self.play(FadeIn(critic_lbl), run_time=0.4)

        memory_note = label_text(
            "4 models in memory simultaneously — 100+ GB for a single 7B training run",
            color=GREY_MED,
        )
        memory_note.to_edge(DOWN, buff=0.25)
        self.play(FadeIn(memory_note), run_time=0.4)
        self.wait(1.5)
        self.fade_all(prob_title, model_boxes, critic_highlight, critic_lbl, memory_note)
        self.wait(1.0)

        # ── 3. Key insight: compare responses to each other ───────────────────
        insight_title = body_text("Key Insight: compare responses to each other — no critic needed",
                                  color=WHITE)
        insight_title.to_edge(UP, buff=0.6)
        self.play(Write(insight_title), run_time=0.7)

        ppo_box = rounded_box(4.8, 1.6, fill_color=str(RED_MED) + "22",
                              stroke_color=RED_MED,
                              label="PPO approach\n\nCritic estimates V(state)\nAdvantage = reward - V(state)\nRequires a learned network",
                              label_color=RED_MED)
        ppo_box.shift(LEFT * 3.2 + DOWN * 0.1)

        grpo_box = rounded_box(4.8, 1.6, fill_color=GREEN_DARK,
                               stroke_color=GREEN_MED,
                               label="GRPO approach\n\nGenerate G responses per prompt\nAdvantage = reward - group mean\nNo learned network needed",
                               label_color=GREEN_LIGHT)
        grpo_box.shift(RIGHT * 3.2 + DOWN * 0.1)

        vs_text = body_text("vs", color=GREY_MED)
        vs_text.move_to(ORIGIN + DOWN * 0.1)

        self.play(FadeIn(ppo_box), run_time=0.6)
        self.play(FadeIn(vs_text), run_time=0.3)
        self.play(FadeIn(grpo_box), run_time=0.6)

        curve_note = label_text(
            "Grading on a curve: each response is compared to its classmates, not an absolute standard",
            color=YELLOW_MED,
        )
        curve_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(curve_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(insight_title, ppo_box, vs_text, grpo_box, curve_note)
        self.wait(1.0)

        # ── 4. Group sampling: one prompt → G different responses ─────────────
        sample_title = body_text("Group Sampling: one prompt generates G=8 responses",
                                 color=WHITE)
        sample_title.to_edge(UP, buff=0.6)
        self.play(Write(sample_title), run_time=0.7)

        prompt_box = rounded_box(5.5, 0.7, stroke_color=BLUE_MED,
                                 label='Prompt: "What is the derivative of x squared?"',
                                 label_color=BLUE_LIGHT)
        prompt_box.move_to(ORIGIN + UP * 2.4)
        self.play(FadeIn(prompt_box), run_time=0.5)

        response_labels = [
            ("R1", GREEN_MED,   "2x  [correct]"),
            ("R2", GREEN_MED,   "2x  [correct]"),
            ("R3", ORANGE_MED,  "x^2 [wrong]"),
            ("R4", GREEN_MED,   "2x  [correct]"),
            ("R5", RED_MED,     "2   [wrong]"),
            ("R6", GREEN_MED,   "2x  [correct]"),
            ("R7", ORANGE_MED,  "x   [wrong]"),
            ("R8", GREEN_MED,   "2x  [correct]"),
        ]

        response_boxes = VGroup()
        for lbl, col, content in response_labels:
            b = rounded_box(1.35, 0.72, fill_color=str(col) + "22",
                            stroke_color=col, label=lbl + "\n" + content,
                            label_color=col)
            response_boxes.add(b)

        response_boxes.arrange(RIGHT, buff=0.22)
        response_boxes.scale_to_fit_width(13)
        response_boxes.move_to(ORIGIN + DOWN * 0.3)

        sample_arrows = VGroup(*[
            Arrow(prompt_box.get_bottom(), rb.get_top(), buff=0.05,
                  color=GREY_MED, stroke_width=1.2,
                  max_tip_length_to_length_ratio=0.18)
            for rb in response_boxes
        ])

        self.play(LaggedStart(*[Create(a) for a in sample_arrows], lag_ratio=0.05),
                  run_time=0.9)
        self.play(LaggedStart(*[FadeIn(b) for b in response_boxes], lag_ratio=0.08),
                  run_time=0.9)

        g_note = label_text(
            "All 8 responses generated from the same prompt — temperature sampling ensures diversity",
            color=GREY_MED,
        )
        g_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(g_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(sample_title, prompt_box, sample_arrows, response_boxes, g_note)
        self.wait(1.0)

        # ── 5. Reward scores for each response (bar chart using rectangles) ────
        reward_title = body_text("Each response gets a reward score from the reward signal",
                                 color=WHITE)
        reward_title.to_edge(UP, buff=0.6)
        self.play(Write(reward_title), run_time=0.7)

        reward_values = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        reward_colors = [GREEN_MED, GREEN_MED, RED_MED, GREEN_MED,
                         RED_MED, GREEN_MED, RED_MED, GREEN_MED]
        reward_names  = ["R1","R2","R3","R4","R5","R6","R7","R8"]

        bar_max_h = 2.4
        bar_w     = 0.9
        bar_gap   = 0.38

        bars_group = VGroup()
        for i, (val, col, name) in enumerate(zip(reward_values, reward_colors, reward_names)):
            h = max(bar_max_h * val, 0.08)
            bar = Rectangle(width=bar_w, height=h,
                            fill_color=col, fill_opacity=0.85,
                            stroke_color=col, stroke_width=1.2)
            # anchor bar bottom at y = -1.4
            bar.move_to([0, -1.4 + h / 2, 0])

            val_lbl = label_text(f"{val:.1f}", color=col)
            val_lbl.next_to(bar, UP, buff=0.1)

            name_lbl = label_text(name, color=GREY_LIGHT)
            name_lbl.next_to(bar, DOWN, buff=0.12)

            grp = VGroup(bar, val_lbl, name_lbl)
            grp.shift(RIGHT * (i * (bar_w + bar_gap)))
            bars_group.add(grp)

        bars_group.move_to(ORIGIN + DOWN * 0.15)

        self.play(LaggedStart(*[FadeIn(b) for b in bars_group], lag_ratio=0.08),
                  run_time=1.2)

        mean_line_y = -1.4 + bar_max_h * 0.625   # mean = 5/8 = 0.625
        mean_line = DashedLine(
            bars_group.get_left() + UP * (mean_line_y - bars_group.get_bottom()[1]),
            bars_group.get_right() + UP * (mean_line_y - bars_group.get_bottom()[1]),
            color=YELLOW_MED, stroke_width=2, dash_length=0.18,
        )
        # Simpler: draw mean line across the bars group at the correct height
        mean_line = DashedLine(
            LEFT * 6.0 + UP * 0.38,
            RIGHT * 6.0 + UP * 0.38,
            color=YELLOW_MED, stroke_width=2, dash_length=0.2,
        )
        mean_lbl = label_text("group mean = 0.625", color=YELLOW_MED)
        mean_lbl.move_to(RIGHT * 5.2 + UP * 0.65)

        self.play(Create(mean_line), FadeIn(mean_lbl), run_time=0.6)

        score_note = label_text(
            "Binary reward: 1.0 = correct answer, 0.0 = wrong  |  5 of 8 correct",
            color=GREY_MED,
        )
        score_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(score_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(reward_title, bars_group, mean_line, mean_lbl, score_note)
        self.wait(1.0)

        # ── 6. Advantage = (reward - group_mean) / group_std ─────────────────
        adv_title = body_text("Advantage = (reward - group mean) / group std",
                              color=WHITE)
        adv_title.to_edge(UP, buff=0.6)
        self.play(Write(adv_title), run_time=0.7)

        # Show the formula components as colored text blocks
        formula_parts = [
            ("Advantage", YELLOW_MED),
            ("  =  ", GREY_LIGHT),
            ("( reward", GREEN_MED),
            ("  -  ", WHITE),
            ("group mean )", ORANGE_MED),
            ("  /  ", WHITE),
            ("group std", BLUE_LIGHT),
        ]
        formula_row = VGroup(*[body_text(t, color=c) for t, c in formula_parts])
        formula_row.arrange(RIGHT, buff=0.05)
        formula_row.move_to(ORIGIN + UP * 1.0)
        self.play(LaggedStart(*[FadeIn(p) for p in formula_row], lag_ratio=0.1),
                  run_time=1.0)

        # Show the example calculation for a correct and incorrect response
        calc_rows = VGroup()
        calc_data = [
            ("Correct response (R1):", "reward = 1.0", "(1.0 - 0.625) / 0.48",
             "advantage = +0.78", GREEN_MED),
            ("Wrong response (R3):", "reward = 0.0", "(0.0 - 0.625) / 0.48",
             "advantage = -1.30", RED_MED),
        ]
        for label, rew, calc, result, col in calc_data:
            lbl_t   = label_text(label,  color=col)
            rew_t   = label_text(rew,    color=GREY_LIGHT)
            calc_t  = label_text(calc,   color=GREY_LIGHT)
            res_t   = label_text(result, color=col)
            rew_t.next_to(lbl_t,  RIGHT, buff=0.35)
            calc_t.next_to(rew_t, RIGHT, buff=0.35)
            res_t.next_to(calc_t, RIGHT, buff=0.35)
            calc_rows.add(VGroup(lbl_t, rew_t, calc_t, res_t))

        calc_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        calc_rows.scale_to_fit_width(12.8)
        calc_rows.move_to(ORIGIN + DOWN * 0.3)

        self.play(LaggedStart(*[FadeIn(r) for r in calc_rows], lag_ratio=0.4),
                  run_time=1.0)

        interp_note = label_text(
            "Positive advantage → reinforce this response  |  Negative advantage → discourage it",
            color=GREY_MED,
        )
        interp_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(interp_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(adv_title, formula_row, calc_rows, interp_note)
        self.wait(1.0)

        # ── 7. The group relative comparison visualization ─────────────────────
        compare_title = body_text("Group-relative: the baseline adapts to every prompt automatically",
                                  color=WHITE)
        compare_title.to_edge(UP, buff=0.6)
        self.play(Write(compare_title), run_time=0.7)

        # Show two prompts, each with their own group mean
        prompt_data = [
            ("Easy prompt\n\"What is 2+2?\"",
             BLUE_MED,
             ["R1: 4\n+0.9", "R2: 4\n+0.9", "R3: 5\n-1.1"],
             [GREEN_MED, GREEN_MED, RED_MED],
             "group mean = 0.9 (easy)"),
            ("Hard prompt\n\"Prove Fermat's Last Theorem\"",
             PURPLE_MED,
             ["R1: ...\n+1.2", "R2: ...\n-0.6", "R3: ...\n-0.6"],
             [GREEN_MED, RED_MED, RED_MED],
             "group mean = 0.2 (hard)"),
        ]

        prompt_groups = VGroup()
        for (p_lbl, p_col, resp_lbls, resp_cols, mean_lbl) in prompt_data:
            p_box = rounded_box(4.0, 0.75, stroke_color=p_col,
                                label=p_lbl, label_color=p_col)

            resp_boxes_sub = VGroup()
            for r_lbl, r_col in zip(resp_lbls, resp_cols):
                rb = rounded_box(1.1, 0.75, fill_color=str(r_col) + "22",
                                 stroke_color=r_col,
                                 label=r_lbl, label_color=r_col)
                resp_boxes_sub.add(rb)
            resp_boxes_sub.arrange(RIGHT, buff=0.2)

            m_lbl = label_text(mean_lbl, color=YELLOW_MED)

            col_group = VGroup(p_box, resp_boxes_sub, m_lbl)
            col_group.arrange(DOWN, buff=0.28)
            prompt_groups.add(col_group)

        prompt_groups.arrange(RIGHT, buff=1.2)
        prompt_groups.move_to(ORIGIN + DOWN * 0.1)

        self.play(LaggedStart(*[FadeIn(g) for g in prompt_groups], lag_ratio=0.4),
                  run_time=1.2)

        adapt_note = label_text(
            "Each group forms its own baseline — hard prompts and easy prompts are judged separately",
            color=GREY_MED,
        )
        adapt_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(adapt_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(compare_title, prompt_groups, adapt_note)
        self.wait(1.0)

        # ── 8. Clipping — same PPO mechanism, applied to group advantages ──────
        clip_title = body_text("Clipping: same as PPO — bound policy updates to prevent instability",
                               color=WHITE)
        clip_title.to_edge(UP, buff=0.6)
        self.play(Write(clip_title), run_time=0.7)

        clip_steps = [
            ("Old policy\ngenerates R1-R8",   GREY_LIGHT,  "temperature sampling\nduring rollout phase"),
            ("Compute\nadvantages",            YELLOW_MED,  "(reward - mean) / std\nfor each response"),
            ("Policy ratio\nr = new / old",    BLUE_MED,    "how much did the\nprobability change?"),
            ("Clip to\n[0.8, 1.2]",           ORANGE_MED,  "epsilon = 0.2\nbounded step size"),
            ("Gradient\nupdate",               GREEN_MED,   "min(r * A, clip(r) * A)\nconservative step"),
        ]

        clip_boxes = VGroup()
        for lbl, col, note in clip_steps:
            b = rounded_box(2.1, 0.85, fill_color=str(col) + "22",
                            stroke_color=col, label=lbl, label_color=col)
            n = label_text(note, color=GREY_LIGHT)
            n.next_to(b, DOWN, buff=0.18)
            clip_boxes.add(VGroup(b, n))

        clip_boxes.arrange(RIGHT, buff=0.4)
        clip_boxes.scale_to_fit_width(13)
        clip_boxes.move_to(ORIGIN + UP * 0.3)

        clip_arrows = VGroup(*[
            Arrow(clip_boxes[i][0].get_right(), clip_boxes[i + 1][0].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.18)
            for i in range(4)
        ])

        self.play(LaggedStart(*[FadeIn(b) for b in clip_boxes], lag_ratio=0.18),
                  run_time=1.3)
        self.play(LaggedStart(*[Create(a) for a in clip_arrows], lag_ratio=0.12),
                  run_time=0.6)

        clip_note = label_text(
            "Clipping prevents a single high-reward response from dominating the gradient update",
            color=GREY_MED,
        )
        clip_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(clip_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(clip_title, clip_boxes, clip_arrows, clip_note)
        self.wait(1.0)

        # ── 9. GRPO vs PPO side-by-side comparison table ─────────────────────
        table_title = body_text("GRPO vs PPO: what changes, what stays the same",
                                color=WHITE)
        table_title.to_edge(UP, buff=0.6)
        self.play(Write(table_title), run_time=0.7)

        table_rows = [
            ("Critic / value model",  "Required — same size\nas policy",  "Not needed",),
            ("Advantage estimate",    "Per-token (critic\nforward pass)",  "Per-response\n(group z-score)",),
            ("Models in memory",      "4 models",                          "3 models",),
            ("Clipping mechanism",    "Yes  (epsilon = 0.2)",              "Yes  (epsilon = 0.2)",),
            ("KL penalty",            "Yes  (vs ref policy)",              "Yes  (vs ref policy)",),
            ("Stability risk",        "Critic collapse\n+ reward hacking", "Reward hacking\n+ small-G noise",),
        ]

        header_row = VGroup(
            body_text("Dimension",    color=GREY_LIGHT),
            body_text("PPO",          color=RED_MED),
            body_text("GRPO",         color=GREEN_MED),
        )
        header_row.arrange(RIGHT, buff=0.0)
        header_row[0].set_width(4.2)
        header_row[1].set_width(4.0)
        header_row[2].set_width(4.0)
        header_row.move_to(ORIGIN + UP * 2.1)

        self.play(FadeIn(header_row), run_time=0.4)

        sep = Line(LEFT * 6.2, RIGHT * 6.2, color=GREY_MED, stroke_width=1)
        sep.next_to(header_row, DOWN, buff=0.12)
        self.play(Create(sep), run_time=0.3)

        data_rows = VGroup()
        for dim, ppo_val, grpo_val in table_rows:
            d_txt   = label_text(dim,      color=GREY_LIGHT)
            ppo_txt = label_text(ppo_val,  color=RED_MED)
            grp_txt = label_text(grpo_val, color=GREEN_MED)
            d_txt.set_width(4.0)
            ppo_txt.set_width(4.0)
            grp_txt.set_width(4.0)
            row = VGroup(d_txt, ppo_txt, grp_txt)
            row.arrange(RIGHT, buff=0.2)
            data_rows.add(row)

        data_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        data_rows.next_to(sep, DOWN, buff=0.2)

        self.play(LaggedStart(*[FadeIn(r) for r in data_rows], lag_ratio=0.15),
                  run_time=1.3)
        self.wait(1.5)
        self.fade_all(table_title, header_row, sep, data_rows)
        self.wait(1.0)

        # ── 10. DeepSeek-R1 used this — real numbers ─────────────────────────
        ds_title = body_text("DeepSeek-R1 (2025): GRPO at 671B scale with verifiable rewards",
                             color=WHITE)
        ds_title.to_edge(UP, buff=0.6)
        self.play(Write(ds_title), run_time=0.7)

        ds_facts = [
            ("Model size",       "671B parameters  (MoE architecture)",      BLUE_LIGHT),
            ("Group size G",     "8 to 16 responses per prompt",              YELLOW_MED),
            ("Reward signal",    "Binary correctness — no reward model for math/code", GREEN_MED),
            ("Reward for math",  "Correct final answer = 1.0,  wrong = 0.0", GREEN_MED),
            ("Format reward",    "Follows reasoning chain format = 0.1",      ORANGE_MED),
            ("KL coefficient",   "beta = 0.04  (light penalty on divergence)", BLUE_MED),
            ("Clip epsilon",     "epsilon = 0.2  (same as PPO default)",      BLUE_MED),
            ("Key outcome",      "Emergent chain-of-thought without explicit training", PURPLE_MED),
        ]

        fact_rows = VGroup()
        for key, val, col in ds_facts:
            k_txt = label_text(key + ":", color=col)
            v_txt = label_text(val,       color=GREY_LIGHT)
            v_txt.next_to(k_txt, RIGHT, buff=0.3)
            fact_rows.add(VGroup(k_txt, v_txt))

        fact_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        fact_rows.scale_to_fit_width(12.8)
        fact_rows.move_to(ORIGIN + DOWN * 0.15)

        self.play(LaggedStart(*[FadeIn(r) for r in fact_rows], lag_ratio=0.12),
                  run_time=1.5)

        src_note = label_text(
            "Source: DeepSeek-R1 paper, arXiv 2501.12948 (2025)",
            color=GREY_MED,
        )
        src_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(src_note), run_time=0.4)
        self.wait(1.5)
        self.fade_all(ds_title, fact_rows, src_note)
        self.wait(1.0)

        # ── 11. Length bias — limitation that leads to Dr. GRPO ──────────────
        bias_title = body_text("Limitation: length bias — longer responses dominate the gradient",
                               color=RED_MED)
        bias_title.to_edge(UP, buff=0.6)
        self.play(Write(bias_title), run_time=0.7)

        # Show a short and a long response with their token counts
        short_box = rounded_box(3.5, 0.85, fill_color=str(GREEN_MED) + "22",
                                stroke_color=GREEN_MED,
                                label="Short response\n50 tokens,  advantage = +1.0",
                                label_color=GREEN_MED)
        short_box.shift(LEFT * 3.2 + UP * 1.0)

        long_box = rounded_box(3.5, 0.85, fill_color=str(ORANGE_MED) + "22",
                               stroke_color=ORANGE_MED,
                               label="Long response\n500 tokens,  advantage = +1.0",
                               label_color=ORANGE_MED)
        long_box.shift(RIGHT * 3.2 + UP * 1.0)

        self.play(FadeIn(short_box), FadeIn(long_box), run_time=0.7)

        grad_short = rounded_box(3.5, 0.75, fill_color=str(GREEN_MED) + "11",
                                 stroke_color=GREEN_MED,
                                 label="Gradient contribution\n50 token-level updates",
                                 label_color=GREEN_MED)
        grad_short.shift(LEFT * 3.2 + DOWN * 0.5)

        grad_long = rounded_box(3.5, 0.75, fill_color=str(ORANGE_MED) + "33",
                                stroke_color=ORANGE_MED,
                                label="Gradient contribution\n500 token-level updates",
                                label_color=ORANGE_MED)
        grad_long.shift(RIGHT * 3.2 + DOWN * 0.5)

        arr_s = Arrow(short_box.get_bottom(), grad_short.get_top(), buff=0.05,
                      color=GREEN_MED, stroke_width=1.5,
                      max_tip_length_to_length_ratio=0.2)
        arr_l = Arrow(long_box.get_bottom(), grad_long.get_top(), buff=0.05,
                      color=ORANGE_MED, stroke_width=1.5,
                      max_tip_length_to_length_ratio=0.2)

        self.play(Create(arr_s), Create(arr_l), run_time=0.5)
        self.play(FadeIn(grad_short), FadeIn(grad_long), run_time=0.6)

        bias_note = label_text(
            "Same advantage score, but 10x more gradient steps — model is incentivised to be verbose",
            color=RED_MED,
        )
        bias_note.to_edge(DOWN, buff=0.55)

        fix_note = label_text(
            "Fix: Dr. GRPO normalises gradient by response length — equal weight per response",
            color=GREEN_LIGHT,
        )
        fix_note.to_edge(DOWN, buff=0.28)

        self.play(FadeIn(bias_note), run_time=0.4)
        self.play(FadeIn(fix_note), run_time=0.4)
        self.wait(1.5)
        self.fade_all(bias_title, short_box, long_box, arr_s, arr_l,
                      grad_short, grad_long, bias_note, fix_note)
        self.wait(1.0)

        # ── 12. Summary ───────────────────────────────────────────────────────
        summary_title = body_text("Summary: what GRPO changes and what it keeps",
                                  color=WHITE)
        summary_title.to_edge(UP, buff=0.6)
        self.play(Write(summary_title), run_time=0.7)

        summary_items = [
            ("Removes",  "the critic — no separately learned value function",           RED_MED),
            ("Adds",     "group sampling — G responses per prompt as the baseline",      GREEN_MED),
            ("Adds",     "advantage normalization — z-score within group",               GREEN_MED),
            ("Keeps",    "PPO clipping at epsilon = 0.2",                                BLUE_MED),
            ("Keeps",    "KL penalty against reference policy",                          BLUE_MED),
            ("Saves",    "~25% model memory for 7B policy; more at 70B+",               YELLOW_MED),
            ("Used in",  "DeepSeekMath (2024), DeepSeek-R1 (2025)",                    PURPLE_MED),
            ("Next",     "RLOO — leave-one-out baseline for unbiased group estimates",   ORANGE_MED),
        ]

        sum_rows = VGroup()
        for verb, desc, col in summary_items:
            v_txt = label_text(verb + ":", color=col)
            d_txt = label_text(desc,       color=GREY_LIGHT)
            d_txt.next_to(v_txt, RIGHT, buff=0.3)
            sum_rows.add(VGroup(v_txt, d_txt))

        sum_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.23)
        sum_rows.scale_to_fit_width(12.8)
        sum_rows.move_to(ORIGIN + DOWN * 0.1)

        self.play(LaggedStart(*[FadeIn(r) for r in sum_rows], lag_ratio=0.1),
                  run_time=1.5)
        self.wait(2.0)
        self.fade_all(summary_title, sum_rows)
        self.wait(1.0)
