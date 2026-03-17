"""
Scene 22 — RL Foundations for LLMs
Run: manim -pql 22_rl_foundations.py RLFoundationsScene
"""
from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class RLFoundationsScene(LLMScene):
    def construct(self):

        # ── 1. Title card ─────────────────────────────────────────────────────
        title = self.show_title("RL Foundations for LLMs", "🎮 Agent · State · Action · Policy · Reward")
        self.wait(1.0)
        self.fade_all(title)

        # ── 2. The RL loop diagram ────────────────────────────────────────────
        loop_title = body_text("The Reinforcement Learning Loop", color=WHITE)
        loop_title.to_edge(UP, buff=0.6)
        self.play(Write(loop_title), run_time=0.7)

        agent_box = rounded_box(2.6, 1.0, fill_color=str(BLUE_MED) + "22",
                                stroke_color=BLUE_MED,
                                label="🤖 Agent\n(Language Model)",
                                label_color=BLUE_LIGHT)
        agent_box.move_to(LEFT * 3.5 + DOWN * 0.3)

        env_box = rounded_box(2.6, 1.0, fill_color=str(ORANGE_MED) + "22",
                              stroke_color=ORANGE_MED,
                              label="🌍 Environment\n(Reward Model)",
                              label_color=ORANGE_MED)
        env_box.move_to(RIGHT * 3.5 + DOWN * 0.3)

        # Action arrow: Agent → Environment (top arc)
        action_label = label_text("Action: next token chosen", color=BLUE_LIGHT)
        action_label.move_to(UP * 1.4)

        action_arrow = CurvedArrow(
            agent_box.get_top(),
            env_box.get_top(),
            angle=-TAU / 6,
            color=BLUE_MED,
            stroke_width=2.5,
        )

        # State + Reward arrow: Environment → Agent (bottom arc)
        reward_label = label_text("State + Reward: new context + score", color=ORANGE_MED)
        reward_label.move_to(DOWN * 1.8)

        reward_arrow = CurvedArrow(
            env_box.get_bottom(),
            agent_box.get_bottom(),
            angle=-TAU / 6,
            color=ORANGE_MED,
            stroke_width=2.5,
        )

        self.play(FadeIn(agent_box), FadeIn(env_box), run_time=0.6)
        self.play(Create(action_arrow), FadeIn(action_label), run_time=0.7)
        self.play(Create(reward_arrow), FadeIn(reward_label), run_time=0.7)

        loop_note = label_text(
            "The agent acts → the environment responds → the agent learns → repeat",
            color=GREY_LIGHT,
        )
        loop_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(loop_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(loop_title, agent_box, env_box,
                      action_arrow, action_label,
                      reward_arrow, reward_label, loop_note)

        # ── 3. State, Action, Policy — LLM analogy ───────────────────────────
        sap_title = body_text("State · Action · Policy — What They Mean for LLMs", color=WHITE)
        sap_title.to_edge(UP, buff=0.6)
        self.play(Write(sap_title), run_time=0.7)

        concepts = [
            ("State 📍",
             BLUE_LIGHT,
             "Prompt + all tokens\ngenerated so far"),
            ("Action 🎲",
             GREEN_MED,
             "Pick the next token\nfrom ~32k vocab options"),
            ("Policy π_θ 🧠",
             PURPLE_MED,
             "LLM's softmax dist.\nover the vocabulary"),
        ]

        concept_boxes = VGroup()
        for title_str, col, desc in concepts:
            header = body_text(title_str, color=col)
            detail = label_text(desc, color=WHITE)
            detail.next_to(header, DOWN, buff=0.15)
            content = VGroup(header, detail)
            bg = SurroundingRectangle(content, color=col,
                                      fill_color=str(col) + "11",
                                      fill_opacity=1,
                                      buff=0.3, corner_radius=0.12)
            concept_boxes.add(VGroup(bg, content))

        concept_boxes.arrange(RIGHT, buff=0.7)
        concept_boxes.scale_to_fit_width(12.8)
        concept_boxes.move_to(UP * 0.4)

        self.play(LaggedStart(*[FadeIn(b) for b in concept_boxes], lag_ratio=0.3),
                  run_time=1.2)

        # Worked example strip
        example_prompt = rounded_box(9.0, 0.65, fill_color=GREY_DARK,
                                     stroke_color=GREY_MED,
                                     label='Prompt (state): "Explain why the sky is blue in one sentence."',
                                     label_color=GREY_LIGHT)
        example_prompt.move_to(DOWN * 1.55)

        token_row_label = label_text(
            "Token choices (actions): The  →  sky  →  appears  →  blue  →  because  →  ...",
            color=GREEN_MED,
        )
        token_row_label.move_to(DOWN * 2.35)

        policy_note = label_text(
            "Each step: policy outputs P(next token | all previous tokens)",
            color=PURPLE_MED,
        )
        policy_note.to_edge(DOWN, buff=0.4)

        self.play(FadeIn(example_prompt), run_time=0.5)
        self.play(FadeIn(token_row_label), run_time=0.5)
        self.play(FadeIn(policy_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(sap_title, concept_boxes, example_prompt,
                      token_row_label, policy_note)

        # ── 4. Reward signal — what it is for LLMs ────────────────────────────
        rew_title = body_text("The Reward Signal — LLM's Report Card 📋", color=WHITE)
        rew_title.to_edge(UP, buff=0.6)
        self.play(Write(rew_title), run_time=0.7)

        # Timeline: generate → evaluate → score
        steps = [
            ("Generate\nResponse", BLUE_MED,    "LLM samples tokens\none by one"),
            ("Reward Model\nEvaluates", ORANGE_MED, "Reads full (prompt,\nresponse) pair"),
            ("Scalar\nReward r", GREEN_MED,     "e.g.  r = 7.4 / 10\nor  r = 1  (correct)"),
        ]

        step_boxes = VGroup()
        for lbl, col, note in steps:
            b = rounded_box(2.8, 0.9, fill_color=str(col) + "22",
                            stroke_color=col, label=lbl, label_color=col)
            n = label_text(note, color=GREY_LIGHT)
            n.next_to(b, DOWN, buff=0.2)
            step_boxes.add(VGroup(b, n))

        step_boxes.arrange(RIGHT, buff=0.9)
        step_boxes.scale_to_fit_width(12.5)
        step_boxes.move_to(UP * 0.5)

        step_arrows = VGroup(*[
            Arrow(step_boxes[i][0].get_right(), step_boxes[i + 1][0].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=2,
                  max_tip_length_to_length_ratio=0.18)
            for i in range(2)
        ])

        self.play(LaggedStart(*[FadeIn(b) for b in step_boxes], lag_ratio=0.3),
                  run_time=1.0)
        self.play(LaggedStart(*[Create(a) for a in step_arrows], lag_ratio=0.3),
                  run_time=0.5)

        sparse_note = rounded_box(10.0, 0.7, fill_color=str(RED_MED) + "22",
                                  stroke_color=RED_MED,
                                  label="⚠️  Sparse reward: the model receives ZERO feedback during token generation — only at the end",
                                  label_color=RED_MED)
        sparse_note.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(sparse_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(rew_title, step_boxes, step_arrows, sparse_note)

        # ── 5. Policy gradient — J(θ) in plain words ─────────────────────────
        pg_title = body_text("Policy Gradient — J(θ) in Plain English", color=WHITE)
        pg_title.to_edge(UP, buff=0.6)
        self.play(Write(pg_title), run_time=0.7)

        j_box = rounded_box(11.0, 0.8, fill_color=str(YELLOW_MED) + "22",
                            stroke_color=YELLOW_MED,
                            label="J(θ)  =  Expected total reward across all prompts and responses",
                            label_color=YELLOW_MED)
        j_box.move_to(UP * 1.6)
        self.play(FadeIn(j_box), run_time=0.5)

        steps_pg = [
            ("Sample\ntrajectory", BLUE_LIGHT,   "Prompt → generate\nfull response"),
            ("Compute\nadvantage", GREEN_MED,     "reward − group\naverage reward"),
            ("Scale log\nprob", ORANGE_MED,      "log P(token) × advantage\nfor each token"),
            ("Gradient\nstep ↑J(θ)", PURPLE_MED,  "Update weights to\nincrease J(θ)"),
        ]

        pg_boxes = VGroup()
        for lbl, col, note in steps_pg:
            b = rounded_box(2.5, 0.85, fill_color=str(col) + "22",
                            stroke_color=col, label=lbl, label_color=col)
            n = label_text(note, color=GREY_LIGHT)
            n.next_to(b, DOWN, buff=0.18)
            pg_boxes.add(VGroup(b, n))

        pg_boxes.arrange(RIGHT, buff=0.5)
        pg_boxes.scale_to_fit_width(12.8)
        pg_boxes.move_to(DOWN * 0.2)

        pg_arrows = VGroup(*[
            Arrow(pg_boxes[i][0].get_right(), pg_boxes[i + 1][0].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.2)
            for i in range(3)
        ])

        self.play(LaggedStart(*[FadeIn(b) for b in pg_boxes], lag_ratio=0.2),
                  run_time=1.2)
        self.play(LaggedStart(*[Create(a) for a in pg_arrows], lag_ratio=0.2),
                  run_time=0.5)

        adv_note = label_text(
            "Positive advantage → raise token probabilities  |  Negative advantage → lower them",
            color=GREY_MED,
        )
        adv_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(adv_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(pg_title, j_box, pg_boxes, pg_arrows, adv_note)

        # ── 6. Trajectory and discounted returns ──────────────────────────────
        traj_title = body_text("Trajectory and Discounted Returns 🛤️", color=WHITE)
        traj_title.to_edge(UP, buff=0.6)
        self.play(Write(traj_title), run_time=0.7)

        # Token sequence boxes representing one trajectory
        token_labels = ["The", "sky", "appears", "blue", "because", "...", "<eos>"]
        token_colors = [BLUE_LIGHT, BLUE_MED, BLUE_MED, GREEN_MED, GREEN_MED, GREY_MED, ORANGE_MED]

        token_boxes = VGroup()
        for tok, col in zip(token_labels, token_colors):
            b = rounded_box(1.2, 0.65, fill_color=str(col) + "22",
                            stroke_color=col, label=tok, label_color=col)
            token_boxes.add(b)

        token_boxes.arrange(RIGHT, buff=0.15)
        token_boxes.scale_to_fit_width(12.8)
        token_boxes.move_to(UP * 1.3)

        traj_label = label_text("One trajectory τ  (episode = one complete response)", color=GREY_LIGHT)
        traj_label.next_to(token_boxes, UP, buff=0.25)

        token_arrows_traj = VGroup(*[
            Arrow(token_boxes[i].get_right(), token_boxes[i + 1].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.25)
            for i in range(len(token_boxes) - 1)
        ])

        self.play(FadeIn(traj_label), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(b) for b in token_boxes], lag_ratio=0.1),
                  run_time=0.9)
        self.play(LaggedStart(*[Create(a) for a in token_arrows_traj], lag_ratio=0.05),
                  run_time=0.5)

        # Reward box at the end
        reward_box_traj = rounded_box(2.0, 0.7, fill_color=GREEN_DARK,
                                      stroke_color=GREEN_MED,
                                      label="r = 7.4", label_color=GREEN_LIGHT)
        reward_box_traj.next_to(token_boxes, DOWN, buff=0.45)
        reward_box_traj.align_to(token_boxes, RIGHT)

        reward_arrow_traj = Arrow(token_boxes[-1].get_bottom(),
                                  reward_box_traj.get_top(),
                                  color=GREEN_MED, buff=0.05, stroke_width=2,
                                  max_tip_length_to_length_ratio=0.2)

        self.play(Create(reward_arrow_traj), FadeIn(reward_box_traj), run_time=0.5)

        # Discounting explanation boxes
        gamma_box = rounded_box(4.5, 0.8, fill_color=str(PURPLE_MED) + "22",
                                stroke_color=PURPLE_MED,
                                label="Discount factor  γ = 0.99",
                                label_color=PURPLE_MED)
        gamma_box.move_to(LEFT * 3.2 + DOWN * 1.5)

        gamma_why = label_text(
            "γ < 1 ensures sums converge\nand values near-term rewards more",
            color=GREY_LIGHT,
        )
        gamma_why.next_to(gamma_box, DOWN, buff=0.2)

        return_note = rounded_box(4.5, 0.8, fill_color=str(YELLOW_MED) + "22",
                                  stroke_color=YELLOW_MED,
                                  label="G_t = r + 0.99r + 0.99²r + ...",
                                  label_color=YELLOW_MED)
        return_note.move_to(RIGHT * 2.8 + DOWN * 1.5)

        self.play(FadeIn(gamma_box), FadeIn(gamma_why), run_time=0.5)
        self.play(FadeIn(return_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(traj_title, traj_label, token_boxes, token_arrows_traj,
                      reward_arrow_traj, reward_box_traj,
                      gamma_box, gamma_why, return_note)

        # ── 7. Why RL differs from supervised learning ────────────────────────
        sl_title = body_text("Supervised Learning vs Reinforcement Learning 🔄", color=WHITE)
        sl_title.to_edge(UP, buff=0.6)
        self.play(Write(sl_title), run_time=0.7)

        rows = [
            ("Dimension",       "Supervised Learning",       "Reinforcement Learning"),
            ("Feedback",        "Exact target token",        "Scalar reward at episode end"),
            ("Training data",   "Fixed labeled dataset",     "Generated by the model live"),
            ("Loss signal",     "Cross-entropy vs target",   "Expected reward (J θ)"),
            ("Correct answer",  "Always known",              "Never given — only good/bad"),
            ("Quality ceiling", "Limited by demo quality",   "Can exceed human demos"),
        ]

        row_groups = VGroup()
        col_colors = [GREY_LIGHT, BLUE_LIGHT, ORANGE_MED]
        col_x = [-4.2, 0.0, 4.2]

        for r_idx, row in enumerate(rows):
            row_group = VGroup()
            for c_idx, cell in enumerate(row):
                if r_idx == 0:
                    txt = body_text(cell, color=col_colors[c_idx])
                else:
                    txt = label_text(cell, color=col_colors[c_idx])
                txt.move_to([col_x[c_idx], 0, 0])
                row_group.add(txt)
            row_groups.add(row_group)

        row_groups.arrange(DOWN, buff=0.32)
        row_groups.move_to(DOWN * 0.1)

        # Header underline
        underline = Line(LEFT * 6.3, RIGHT * 6.3, color=GREY_MED, stroke_width=1)
        underline.next_to(row_groups[0], DOWN, buff=0.1)

        self.play(FadeIn(row_groups[0]), FadeIn(underline), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r) for r in row_groups[1:]], lag_ratio=0.15),
                  run_time=1.2)

        ceiling_note = label_text(
            "Key insight: RL can discover responses better than any human demonstration",
            color=GREEN_LIGHT,
        )
        ceiling_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(ceiling_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(sl_title, row_groups, underline, ceiling_note)

        # ── 8. LLM as RL agent — full diagram ────────────────────────────────
        llm_title = body_text("The LLM as RL Agent — Full Data Flow 🤖", color=WHITE)
        llm_title.to_edge(UP, buff=0.6)
        self.play(Write(llm_title), run_time=0.7)

        prompt_box = rounded_box(2.2, 0.85, fill_color=GREY_DARK,
                                 stroke_color=GREY_MED,
                                 label="📨 Prompt\n(initial state)",
                                 label_color=GREY_LIGHT)
        prompt_box.move_to(LEFT * 5.0 + UP * 0.5)

        llm_box = rounded_box(2.2, 0.85, fill_color=str(BLUE_MED) + "22",
                              stroke_color=BLUE_MED,
                              label="🧠 LLM\nPolicy π_θ",
                              label_color=BLUE_LIGHT)
        llm_box.move_to(LEFT * 2.0 + UP * 0.5)

        tokens_box = rounded_box(2.2, 0.85, fill_color=str(GREEN_MED) + "22",
                                 stroke_color=GREEN_MED,
                                 label="🔤 Tokens\n(actions)",
                                 label_color=GREEN_LIGHT)
        tokens_box.move_to(RIGHT * 1.0 + UP * 0.5)

        rm_box = rounded_box(2.2, 0.85, fill_color=str(ORANGE_MED) + "22",
                             stroke_color=ORANGE_MED,
                             label="⚖️  Reward\nModel",
                             label_color=ORANGE_MED)
        rm_box.move_to(RIGHT * 4.2 + UP * 0.5)

        reward_scalar_box = rounded_box(2.2, 0.85, fill_color=GREEN_DARK,
                                        stroke_color=GREEN_MED,
                                        label="r = 7.4\n(scalar reward)",
                                        label_color=GREEN_LIGHT)
        reward_scalar_box.move_to(RIGHT * 4.2 + DOWN * 1.6)

        gradient_box = rounded_box(2.8, 0.8, fill_color=str(PURPLE_MED) + "22",
                                   stroke_color=PURPLE_MED,
                                   label="Gradient update\nΔθ via policy gradient",
                                   label_color=PURPLE_MED)
        gradient_box.move_to(LEFT * 2.0 + DOWN * 1.6)

        e1 = Arrow(prompt_box.get_right(), llm_box.get_left(),
                   color=GREY_MED, buff=0.05, stroke_width=2,
                   max_tip_length_to_length_ratio=0.18)
        e2 = Arrow(llm_box.get_right(), tokens_box.get_left(),
                   color=BLUE_LIGHT, buff=0.05, stroke_width=2,
                   max_tip_length_to_length_ratio=0.18)
        e3 = Arrow(tokens_box.get_right(), rm_box.get_left(),
                   color=GREEN_MED, buff=0.05, stroke_width=2,
                   max_tip_length_to_length_ratio=0.18)
        e4 = Arrow(rm_box.get_bottom(), reward_scalar_box.get_top(),
                   color=ORANGE_MED, buff=0.05, stroke_width=2,
                   max_tip_length_to_length_ratio=0.18)
        e5 = Arrow(reward_scalar_box.get_left(), gradient_box.get_right(),
                   color=GREEN_MED, buff=0.05, stroke_width=2,
                   max_tip_length_to_length_ratio=0.18)
        e6 = Arrow(gradient_box.get_top(), llm_box.get_bottom(),
                   color=PURPLE_MED, buff=0.05, stroke_width=2,
                   max_tip_length_to_length_ratio=0.18)

        all_boxes_llm = VGroup(prompt_box, llm_box, tokens_box, rm_box,
                               reward_scalar_box, gradient_box)
        all_edges_llm = VGroup(e1, e2, e3, e4, e5, e6)

        self.play(LaggedStart(*[FadeIn(b) for b in all_boxes_llm], lag_ratio=0.15),
                  run_time=1.2)
        self.play(LaggedStart(*[Create(e) for e in all_edges_llm], lag_ratio=0.1),
                  run_time=1.2)

        loop_lbl = label_text("Loop repeats: updated LLM generates next batch of responses",
                              color=GREY_MED)
        loop_lbl.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(loop_lbl), run_time=0.5)
        self.wait(1.5)
        self.fade_all(llm_title, all_boxes_llm, all_edges_llm, loop_lbl)

        # ── 9. Exploration vs exploitation in token sampling ──────────────────
        expl_title = body_text("Exploration vs Exploitation — Temperature Controls This 🎲", color=WHITE)
        expl_title.to_edge(UP, buff=0.6)
        self.play(Write(expl_title), run_time=0.7)

        temp_cases = [
            ("Temperature\n0.0  (greedy)", RED_MED,
             "Always top token\nNo exploration\nSame output every run"),
            ("Temperature\n1.0  (balanced)", GREEN_MED,
             "Sample from softmax\nGood exploration\nDefault for RL training"),
            ("Temperature\n2.0  (random)", ORANGE_MED,
             "Flat distribution\nToo much noise\nIncoherent outputs"),
        ]

        temp_boxes = VGroup()
        for lbl, col, note in temp_cases:
            header = body_text(lbl, color=col)
            detail = label_text(note, color=WHITE)
            detail.next_to(header, DOWN, buff=0.18)
            content = VGroup(header, detail)
            bg = SurroundingRectangle(content, color=col,
                                      fill_color=str(col) + "11",
                                      fill_opacity=1,
                                      buff=0.3, corner_radius=0.12)
            temp_boxes.add(VGroup(bg, content))

        temp_boxes.arrange(RIGHT, buff=0.7)
        temp_boxes.scale_to_fit_width(12.8)
        temp_boxes.move_to(UP * 0.25)

        self.play(LaggedStart(*[FadeIn(b) for b in temp_boxes], lag_ratio=0.3),
                  run_time=1.2)

        recommended = rounded_box(7.0, 0.65, fill_color=GREEN_DARK,
                                  stroke_color=GREEN_MED,
                                  label="✅  RL training uses temperature = 1.0 — explore without losing coherence",
                                  label_color=GREEN_LIGHT)
        recommended.to_edge(DOWN, buff=0.55)
        self.play(FadeIn(recommended), run_time=0.5)

        exploit_note = label_text(
            "Exploration = sampling different tokens  |  Exploitation = taking the highest-prob token",
            color=GREY_MED,
        )
        exploit_note.next_to(recommended, UP, buff=0.3)
        self.play(FadeIn(exploit_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(expl_title, temp_boxes, recommended, exploit_note)

        # ── 10. Summary / key takeaways ───────────────────────────────────────
        summary_title = body_text("Key Takeaways — RL Foundations 📋", color=WHITE)
        summary_title.to_edge(UP, buff=0.6)
        self.play(Write(summary_title), run_time=0.7)

        takeaways = [
            ("🤖 Agent",      BLUE_LIGHT,  "The LLM and its weights θ"),
            ("🌍 Environment", ORANGE_MED,  "Reward model + text context"),
            ("📍 State",       BLUE_MED,    "Prompt + tokens so far"),
            ("🎲 Action",      GREEN_MED,   "Next token chosen (~32k options)"),
            ("🧠 Policy π_θ",  PURPLE_MED,  "Softmax probability dist. over vocab"),
            ("📊 Reward r",    YELLOW_MED,  "Scalar from reward model — end only"),
            ("📈 J(θ)",        GREEN_LIGHT, "Expected reward — what we maximise"),
            ("↑ Policy grad.", ORANGE_MED,  "log P(token) × advantage per token"),
        ]

        takeaway_rows = VGroup()
        for icon_label, col, desc in takeaways:
            label_item = body_text(icon_label, color=col)
            label_item.scale(0.75)
            desc_item = label_text(desc, color=WHITE)
            label_item.move_to(LEFT * 3.8)
            desc_item.next_to(label_item, RIGHT, buff=0.5, aligned_edge=LEFT)
            takeaway_rows.add(VGroup(label_item, desc_item))

        takeaway_rows.arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        takeaway_rows.move_to(DOWN * 0.05)

        self.play(LaggedStart(*[FadeIn(r) for r in takeaway_rows], lag_ratio=0.1),
                  run_time=1.8)

        next_lbl = label_text("Up next → REINFORCE Algorithm", color=YELLOW_MED)
        next_lbl.to_edge(DOWN, buff=0.4)
        self.play(Write(next_lbl), run_time=0.6)
        self.wait(2.0)
        self.fade_all(summary_title, takeaway_rows, next_lbl)
