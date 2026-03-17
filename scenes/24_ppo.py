"""
Scene 24 — PPO: Proximal Policy Optimization
Run: manim -pql 24_ppo.py PPOScene
"""
from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class PPOScene(LLMScene):
    def construct(self):

        # ── 0. Title card ──────────────────────────────────────────────────────
        title = self.show_title("PPO", "Proximal Policy Optimization")
        subtitle = body_text("The algorithm behind InstructGPT and RLHF", color=GREY_LIGHT)
        subtitle.next_to(title, DOWN, buff=0.35)
        self.play(FadeIn(subtitle), run_time=0.6)
        self.wait(1.0)
        self.fade_all(title, subtitle)
        self.wait(1.0)

        # ── 1. The problem: policy collapse ────────────────────────────────────
        prob_title = body_text("The Problem: Policy Collapse", color=WHITE)
        prob_title.to_edge(UP, buff=0.6)
        self.play(Write(prob_title), run_time=0.7)

        # Three-box chain: collect data → large gradient update → policy collapse
        prob_steps = [
            ("Collect data\nwith old policy",  BLUE_MED,   "trajectories & rewards"),
            ("Big gradient\nstep",             ORANGE_MED, "no bound on update size"),
            ("Policy\ncollapse!",              RED_MED,    "new policy is very different\ndata is now unreliable"),
        ]
        prob_boxes = VGroup()
        for lbl, col, note in prob_steps:
            b = rounded_box(3.2, 0.9, fill_color=str(col) + "22",
                            stroke_color=col, label=lbl, label_color=col)
            n = label_text(note, color=GREY_LIGHT)
            n.next_to(b, DOWN, buff=0.2)
            prob_boxes.add(VGroup(b, n))

        prob_boxes.arrange(RIGHT, buff=0.8)
        prob_boxes.scale_to_fit_width(13.0)
        prob_boxes.move_to(ORIGIN + UP * 0.4)

        prob_arrows = VGroup(*[
            Arrow(prob_boxes[i][0].get_right(), prob_boxes[i + 1][0].get_left(),
                  color=RED_MED, buff=0.05, stroke_width=2,
                  max_tip_length_to_length_ratio=0.18)
            for i in range(2)
        ])

        self.play(LaggedStart(*[FadeIn(b) for b in prob_boxes], lag_ratio=0.3),
                  run_time=1.2)
        self.play(LaggedStart(*[Create(a) for a in prob_arrows], lag_ratio=0.3),
                  run_time=0.6)

        # The distribution mismatch note
        mismatch_note = label_text(
            "Distribution mismatch: data collected by old policy, gradient applied to new policy",
            color=ORANGE_MED,
        )
        mismatch_note.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(mismatch_note), run_time=0.5)

        solution_lbl = label_text(
            "PPO solution: bound how much the policy can change per step",
            color=GREEN_MED,
        )
        solution_lbl.to_edge(DOWN, buff=0.25)
        self.play(FadeIn(solution_lbl), run_time=0.5)
        self.wait(1.0)
        self.fade_all(prob_title, prob_boxes, prob_arrows, mismatch_note, solution_lbl)
        self.wait(1.0)

        # ── 2. The ratio r(theta) = new / old policy ───────────────────────────
        ratio_title = body_text("The Policy Ratio: measuring how much has changed", color=WHITE)
        ratio_title.to_edge(UP, buff=0.6)
        self.play(Write(ratio_title), run_time=0.7)

        # Visual gauge: ratio bar with three zones
        gauge_label = label_text("ratio  =  new policy probability / old policy probability",
                                 color=GREY_LIGHT)
        gauge_label.shift(UP * 2.0)
        self.play(FadeIn(gauge_label), run_time=0.5)

        # Three ratio zones as coloured boxes
        zone_data = [
            ("ratio = 0.5\nHalf as likely",    RED_MED,    "policy moved\naway fast"),
            ("ratio = 1.0\nNo change",         GREEN_MED,  "policy unchanged"),
            ("ratio = 2.0\nTwice as likely",   ORANGE_MED, "policy moved\ntoward action fast"),
        ]
        zone_boxes = VGroup()
        for lbl, col, sub in zone_data:
            b = rounded_box(3.0, 0.9, fill_color=str(col) + "22",
                            stroke_color=col, label=lbl, label_color=col)
            s = label_text(sub, color=GREY_LIGHT)
            s.next_to(b, DOWN, buff=0.18)
            zone_boxes.add(VGroup(b, s))

        zone_boxes.arrange(RIGHT, buff=0.6)
        zone_boxes.scale_to_fit_width(12.5)
        zone_boxes.move_to(ORIGIN + UP * 0.3)

        self.play(LaggedStart(*[FadeIn(z) for z in zone_boxes], lag_ratio=0.3),
                  run_time=1.0)

        # Advantage interpretation
        adv_row = VGroup(
            label_text("Positive advantage  +  ratio > 1", color=GREEN_MED),
            label_text("→  reinforce this action", color=GREEN_LIGHT),
        )
        adv_row.arrange(RIGHT, buff=0.4)

        bad_row = VGroup(
            label_text("Negative advantage  +  ratio < 1", color=RED_MED),
            label_text("→  discourage this action", color=RED_MED),
        )
        bad_row.arrange(RIGHT, buff=0.4)

        interp = VGroup(adv_row, bad_row)
        interp.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        interp.to_edge(DOWN, buff=0.55)

        self.play(LaggedStart(*[FadeIn(r) for r in interp], lag_ratio=0.4),
                  run_time=0.8)
        self.wait(1.0)
        self.fade_all(ratio_title, gauge_label, zone_boxes, interp)
        self.wait(1.0)

        # ── 3. The clipping mechanism ──────────────────────────────────────────
        clip_title = body_text("Clipping: bound the ratio to [1 - eps, 1 + eps]", color=WHITE)
        clip_title.to_edge(UP, buff=0.6)
        self.play(Write(clip_title), run_time=0.7)

        # Show the axis from 0 to 2+, with clipped zone highlighted
        # Using rectangles to represent the number line zones
        below_zone = RoundedRectangle(width=2.8, height=0.7, corner_radius=0.1,
                                      fill_color=str(RED_MED) + "33",
                                      stroke_color=RED_MED, stroke_width=1.5)
        below_zone.shift(LEFT * 4.2 + UP * 1.0)
        below_label = label_text("ratio < 0.8\nclipped at 0.8", color=RED_MED)
        below_label.next_to(below_zone, DOWN, buff=0.2)

        safe_zone = RoundedRectangle(width=3.2, height=0.7, corner_radius=0.1,
                                     fill_color=str(GREEN_MED) + "44",
                                     stroke_color=GREEN_MED, stroke_width=2)
        safe_zone.move_to(ORIGIN + UP * 1.0)
        safe_label = label_text("ratio in [0.8, 1.2]\nallowed zone", color=GREEN_MED)
        safe_label.next_to(safe_zone, DOWN, buff=0.2)

        above_zone = RoundedRectangle(width=2.8, height=0.7, corner_radius=0.1,
                                      fill_color=str(ORANGE_MED) + "33",
                                      stroke_color=ORANGE_MED, stroke_width=1.5)
        above_zone.shift(RIGHT * 4.2 + UP * 1.0)
        above_label = label_text("ratio > 1.2\nclipped at 1.2", color=ORANGE_MED)
        above_label.next_to(above_zone, DOWN, buff=0.2)

        # Axis dividers (vertical lines)
        div_left = Line(LEFT * 1.6 + UP * 1.4, LEFT * 1.6 + DOWN * 0.2,
                        color=GREY_MED, stroke_width=1.5)
        div_right = Line(RIGHT * 1.6 + UP * 1.4, RIGHT * 1.6 + DOWN * 0.2,
                         color=GREY_MED, stroke_width=1.5)

        # Labels for the boundaries
        lbl_08 = label_text("0.8", color=GREY_LIGHT)
        lbl_08.next_to(div_left, UP, buff=0.1)
        lbl_12 = label_text("1.2", color=GREY_LIGHT)
        lbl_12.next_to(div_right, UP, buff=0.1)

        self.play(FadeIn(below_zone), FadeIn(below_label), run_time=0.5)
        self.play(FadeIn(safe_zone), FadeIn(safe_label), run_time=0.5)
        self.play(FadeIn(above_zone), FadeIn(above_label), run_time=0.5)
        self.play(Create(div_left), Create(div_right),
                  FadeIn(lbl_08), FadeIn(lbl_12), run_time=0.5)

        # Explain the min operation
        min_note = rounded_box(9.5, 0.75, fill_color=str(BLUE_MED) + "22",
                               stroke_color=BLUE_MED,
                               label="Objective = min( ratio * advantage,   clip(ratio, 0.8, 1.2) * advantage )",
                               label_color=BLUE_LIGHT)
        min_note.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(min_note), run_time=0.5)

        eps_note = label_text(
            "epsilon = 0.2 (typical)  —  each step changes any action probability by at most 20%",
            color=GREY_MED,
        )
        eps_note.next_to(min_note, UP, buff=0.2)
        self.play(FadeIn(eps_note), run_time=0.4)
        self.wait(1.0)
        self.fade_all(clip_title, below_zone, below_label, safe_zone, safe_label,
                      above_zone, above_label, div_left, div_right,
                      lbl_08, lbl_12, min_note, eps_note)
        self.wait(1.0)

        # ── 4. GAE: Generalized Advantage Estimation ───────────────────────────
        gae_title = body_text("GAE: How Advantage Is Estimated", color=WHITE)
        gae_title.to_edge(UP, buff=0.6)
        self.play(Write(gae_title), run_time=0.7)

        # Advantage = actual - expected
        adv_def = rounded_box(7.0, 0.75, fill_color=str(YELLOW_MED) + "22",
                              stroke_color=YELLOW_MED,
                              label="Advantage  =  actual reward received  -  expected reward (critic)",
                              label_color=YELLOW_MED)
        adv_def.shift(UP * 2.0)
        self.play(FadeIn(adv_def), run_time=0.5)

        # Two extremes: MC vs TD
        mc_box = rounded_box(4.5, 1.0, fill_color=str(BLUE_MED) + "22",
                             stroke_color=BLUE_MED,
                             label="Monte Carlo (lambda=1)\nWait for full episode\nLow bias, HIGH variance",
                             label_color=BLUE_LIGHT)
        mc_box.shift(LEFT * 3.2 + UP * 0.4)

        td_box = rounded_box(4.5, 1.0, fill_color=str(GREEN_MED) + "22",
                             stroke_color=GREEN_MED,
                             label="TD Estimation (lambda=0)\nBootstrap from value fn\nHIGH bias, low variance",
                             label_color=GREEN_LIGHT)
        td_box.shift(RIGHT * 3.2 + UP * 0.4)

        self.play(FadeIn(mc_box), FadeIn(td_box), run_time=0.7)

        # GAE blends both
        gae_box = rounded_box(6.0, 0.8, fill_color=str(PURPLE_MED) + "22",
                              stroke_color=PURPLE_MED,
                              label="GAE: lambda in (0, 1)  =  weighted blend\nlambda=0.95 typical — mostly low-variance TD",
                              label_color=PURPLE_MED)
        gae_box.shift(DOWN * 0.8)

        blend_arrow_l = Arrow(mc_box.get_bottom(), gae_box.get_top() + LEFT * 1.5,
                              color=GREY_MED, buff=0.05, stroke_width=1.5,
                              max_tip_length_to_length_ratio=0.18)
        blend_arrow_r = Arrow(td_box.get_bottom(), gae_box.get_top() + RIGHT * 1.5,
                              color=GREY_MED, buff=0.05, stroke_width=1.5,
                              max_tip_length_to_length_ratio=0.18)

        self.play(Create(blend_arrow_l), Create(blend_arrow_r), run_time=0.5)
        self.play(FadeIn(gae_box), run_time=0.5)

        gae_note = label_text(
            "Better advantage estimates → more accurate gradient → faster stable learning",
            color=GREY_MED,
        )
        gae_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(gae_note), run_time=0.4)
        self.wait(1.0)
        self.fade_all(gae_title, adv_def, mc_box, td_box,
                      blend_arrow_l, blend_arrow_r, gae_box, gae_note)
        self.wait(1.0)

        # ── 5. Actor-Critic Architecture ───────────────────────────────────────
        ac_title = body_text("Actor-Critic: Two Networks, One Goal", color=WHITE)
        ac_title.to_edge(UP, buff=0.6)
        self.play(Write(ac_title), run_time=0.7)

        # Actor box (left)
        actor_box = rounded_box(3.8, 1.3, fill_color=str(BLUE_MED) + "22",
                                stroke_color=BLUE_MED,
                                label="ACTOR  (policy)\nSame LLM architecture\nOutputs: token distribution",
                                label_color=BLUE_LIGHT)
        actor_box.shift(LEFT * 3.3 + UP * 0.6)

        # Critic box (right)
        critic_box = rounded_box(3.8, 1.3, fill_color=str(ORANGE_MED) + "22",
                                 stroke_color=ORANGE_MED,
                                 label="CRITIC  (value fn)\nSame architecture + scalar head\nOutputs: expected return V(s)",
                                 label_color=ORANGE_MED)
        critic_box.shift(RIGHT * 3.3 + UP * 0.6)

        self.play(FadeIn(actor_box), FadeIn(critic_box), run_time=0.7)

        # Shared input
        input_lbl = label_text("Both receive the same token sequence (state)", color=GREY_LIGHT)
        input_lbl.shift(UP * 2.2)
        self.play(FadeIn(input_lbl), run_time=0.4)

        # Arrows from both to advantage computation
        adv_compute = rounded_box(4.5, 0.75, fill_color=str(YELLOW_MED) + "22",
                                  stroke_color=YELLOW_MED,
                                  label="Advantage  =  reward  -  V(s)\n(used to update Actor)",
                                  label_color=YELLOW_MED)
        adv_compute.shift(DOWN * 0.9)

        a_arrow = Arrow(actor_box.get_bottom(), adv_compute.get_top() + LEFT * 1.0,
                        color=BLUE_MED, buff=0.05, stroke_width=1.5,
                        max_tip_length_to_length_ratio=0.18)
        c_arrow = Arrow(critic_box.get_bottom(), adv_compute.get_top() + RIGHT * 1.0,
                        color=ORANGE_MED, buff=0.05, stroke_width=1.5,
                        max_tip_length_to_length_ratio=0.18)

        self.play(Create(a_arrow), Create(c_arrow), run_time=0.5)
        self.play(FadeIn(adv_compute), run_time=0.5)

        # Note: actor is deployed, critic is training-only
        deploy_note = label_text(
            "Actor is deployed after training.  Critic is discarded — only needed during PPO.",
            color=GREY_MED,
        )
        deploy_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(deploy_note), run_time=0.4)
        self.wait(1.0)
        self.fade_all(ac_title, actor_box, critic_box, input_lbl,
                      a_arrow, c_arrow, adv_compute, deploy_note)
        self.wait(1.0)

        # ── 6. PPO Full Loop ───────────────────────────────────────────────────
        loop_title = body_text("The PPO Training Loop: one iteration", color=WHITE)
        loop_title.to_edge(UP, buff=0.6)
        self.play(Write(loop_title), run_time=0.7)

        loop_steps = [
            ("Sample\nprompts",    GREY_LIGHT,  "from prompt dataset"),
            ("Actor generates\nresponse", BLUE_MED, "full token sequence"),
            ("Reward model\nscores", ORANGE_MED, "scalar r per response"),
            ("Compute\nadvantage", YELLOW_MED,  "GAE with critic estimates"),
            ("Clip + Update\nActor + Critic", PURPLE_MED, "stay within [0.8, 1.2]"),
        ]
        loop_boxes = VGroup()
        for lbl, col, note in loop_steps:
            b = rounded_box(2.2, 0.9, fill_color=str(col) + "22",
                            stroke_color=col, label=lbl, label_color=col)
            n = label_text(note, color=GREY_LIGHT)
            n.next_to(b, DOWN, buff=0.18)
            loop_boxes.add(VGroup(b, n))

        loop_boxes.arrange(RIGHT, buff=0.35)
        loop_boxes.scale_to_fit_width(13.2)
        loop_boxes.move_to(ORIGIN + UP * 0.4)

        loop_arrows = VGroup(*[
            Arrow(loop_boxes[i][0].get_right(), loop_boxes[i + 1][0].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.18)
            for i in range(4)
        ])

        self.play(LaggedStart(*[FadeIn(b) for b in loop_boxes], lag_ratio=0.15),
                  run_time=1.2)
        self.play(LaggedStart(*[Create(a) for a in loop_arrows], lag_ratio=0.1),
                  run_time=0.5)

        # Loopback arrow
        loop_back = CurvedArrow(
            loop_boxes[4][0].get_bottom(),
            loop_boxes[0][0].get_bottom(),
            angle=-TAU / 6, color=PURPLE_MED, stroke_width=1.5,
        )
        loop_back_lbl = label_text("next batch", color=PURPLE_MED)
        loop_back_lbl.next_to(loop_back, DOWN, buff=0.1)

        self.play(Create(loop_back), FadeIn(loop_back_lbl), run_time=0.6)
        self.wait(1.0)
        self.fade_all(loop_title, loop_boxes, loop_arrows, loop_back, loop_back_lbl)
        self.wait(1.0)

        # ── 7. PPO in LLM Training: 4 models + memory ─────────────────────────
        llm_title = body_text("PPO for LLMs: Four Models in GPU Memory", color=WHITE)
        llm_title.to_edge(UP, buff=0.6)
        self.play(Write(llm_title), run_time=0.7)

        # Four model boxes in a 2x2 grid
        model_data = [
            ("Actor\n(RL Policy)",      BLUE_MED,    "Being trained\nFull gradients + optimizer",   LEFT * 3.3 + UP * 1.2),
            ("Reference Policy\n(frozen SFT)", GREY_LIGHT, "Never updated\nKL computation only",   RIGHT * 3.3 + UP * 1.2),
            ("Critic\n(Value Model)",   ORANGE_MED,  "Trained jointly\nEstimates V(state)",         LEFT * 3.3 + DOWN * 0.8),
            ("Reward Model\n(frozen)",  GREEN_MED,   "Never updated\nScores completed responses",   RIGHT * 3.3 + DOWN * 0.8),
        ]
        model_boxes = VGroup()
        for lbl, col, note, pos in model_data:
            b = rounded_box(3.8, 0.95, fill_color=str(col) + "22",
                            stroke_color=col, label=lbl, label_color=col)
            n = label_text(note, color=GREY_LIGHT)
            n.next_to(b, DOWN, buff=0.15)
            grp = VGroup(b, n)
            grp.move_to(pos)
            model_boxes.add(grp)

        self.play(LaggedStart(*[FadeIn(m) for m in model_boxes], lag_ratio=0.2),
                  run_time=1.2)

        mem_note = label_text(
            "7B param model: ~150+ GB total across all four models — requires multi-GPU setup",
            color=RED_MED,
        )
        mem_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(mem_note), run_time=0.5)
        self.wait(1.0)
        self.fade_all(llm_title, model_boxes, mem_note)
        self.wait(1.0)

        # ── 8. KL Penalty as backstop ──────────────────────────────────────────
        kl_title = body_text("KL Penalty: the leash that prevents reward hacking", color=WHITE)
        kl_title.to_edge(UP, buff=0.6)
        self.play(Write(kl_title), run_time=0.7)

        # Show the total reward formula as a box chain
        rm_score = rounded_box(2.8, 0.75, fill_color=str(GREEN_MED) + "22",
                               stroke_color=GREEN_MED,
                               label="Reward model\nscore  r", label_color=GREEN_LIGHT)
        rm_score.shift(LEFT * 4.0 + UP * 1.0)

        minus_lbl = body_text("-", color=WHITE)
        minus_lbl.shift(LEFT * 1.4 + UP * 1.0)

        kl_penalty = rounded_box(2.8, 0.75, fill_color=str(RED_MED) + "22",
                                 stroke_color=RED_MED,
                                 label="KL penalty\nlambda * KL(actor || ref)",
                                 label_color=RED_MED)
        kl_penalty.shift(RIGHT * 1.2 + UP * 1.0)

        equals_lbl = body_text("=", color=WHITE)
        equals_lbl.shift(RIGHT * 3.7 + UP * 1.0)

        total_reward = rounded_box(2.5, 0.75, fill_color=str(YELLOW_MED) + "22",
                                   stroke_color=YELLOW_MED,
                                   label="Total reward\nused for PPO",
                                   label_color=YELLOW_MED)
        total_reward.shift(RIGHT * 5.6 + UP * 1.0)

        self.play(FadeIn(rm_score), run_time=0.4)
        self.play(FadeIn(minus_lbl), FadeIn(kl_penalty), run_time=0.4)
        self.play(FadeIn(equals_lbl), FadeIn(total_reward), run_time=0.4)

        # Two failure mode examples
        hack_box = rounded_box(5.2, 0.85, fill_color=str(RED_MED) + "22",
                               stroke_color=RED_MED,
                               label="Without KL:\nmodel learns bizarre styles that fool reward model",
                               label_color=RED_MED)
        hack_box.shift(LEFT * 3.2 + DOWN * 0.8)

        good_box = rounded_box(5.2, 0.85, fill_color=str(GREEN_MED) + "22",
                               stroke_color=GREEN_MED,
                               label="With KL:\nmodel improves while staying close to SFT behaviour",
                               label_color=GREEN_LIGHT)
        good_box.shift(RIGHT * 3.0 + DOWN * 0.8)

        self.play(FadeIn(hack_box), FadeIn(good_box), run_time=0.6)

        kl_coef_note = label_text(
            "KL coefficient lambda: too small = reward hacking,  too large = model cannot improve",
            color=GREY_MED,
        )
        kl_coef_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(kl_coef_note), run_time=0.4)
        self.wait(1.0)
        self.fade_all(kl_title, rm_score, minus_lbl, kl_penalty, equals_lbl,
                      total_reward, hack_box, good_box, kl_coef_note)
        self.wait(1.0)

        # ── 9. Limitations: expensive, unstable → motivation for GRPO ─────────
        lim_title = body_text("Limitations: why the field moved beyond PPO", color=WHITE)
        lim_title.to_edge(UP, buff=0.6)
        self.play(Write(lim_title), run_time=0.7)

        lim_data = [
            ("Memory cost",    RED_MED,    "4 models loaded simultaneously\n~150+ GB for a 7B actor"),
            ("Training instability", ORANGE_MED, "Reward hacking, KL explosion\nValue model collapse"),
            ("Engineering complexity", YELLOW_MED, "37+ implementation details\nDozens of hyperparameters"),
            ("Sample efficiency", BLUE_MED, "Must generate responses\nfor every gradient step"),
        ]
        lim_boxes = VGroup()
        for name, col, desc in lim_data:
            b = rounded_box(2.9, 1.05, fill_color=str(col) + "22",
                            stroke_color=col, label=name, label_color=col)
            d = label_text(desc, color=GREY_LIGHT)
            d.next_to(b, DOWN, buff=0.18)
            lim_boxes.add(VGroup(b, d))

        lim_boxes.arrange(RIGHT, buff=0.5)
        lim_boxes.scale_to_fit_width(13.2)
        lim_boxes.move_to(ORIGIN + UP * 0.3)

        self.play(LaggedStart(*[FadeIn(b) for b in lim_boxes], lag_ratio=0.2),
                  run_time=1.2)

        grpo_tease = rounded_box(8.0, 0.65, fill_color=str(PURPLE_MED) + "22",
                                 stroke_color=PURPLE_MED,
                                 label="GRPO (next): no critic, 2 models, group-relative advantage — simpler and cheaper",
                                 label_color=PURPLE_MED)
        grpo_tease.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(grpo_tease), run_time=0.5)
        self.wait(1.0)
        self.fade_all(lim_title, lim_boxes, grpo_tease)
        self.wait(1.0)

        # ── 10. Summary with key numbers ──────────────────────────────────────
        summary_title = body_text("PPO at a Glance — Key Numbers", color=WHITE)
        summary_title.to_edge(UP, buff=0.6)
        self.play(Write(summary_title), run_time=0.7)

        summary_rows = [
            ("Clipping epsilon",        "0.2",         "max 20% change per action per step",   BLUE_LIGHT),
            ("GAE lambda",              "0.95",         "mostly TD, some Monte Carlo",          GREEN_LIGHT),
            ("Models in memory",        "4",            "actor, critic, reference, reward",     ORANGE_MED),
            ("Memory (7B model)",       "150+ GB",      "requires multi-GPU setup",             RED_MED),
            ("KL coefficient",          "0.01 - 0.1",   "tuned per run to balance improvement", YELLOW_MED),
            ("PPO epochs per batch",    "1 - 4",        "how many gradient steps per rollout",  PURPLE_MED),
            ("Training steps (RLHF)",   "1k - 10k",     "short vs pretraining billions",        GREY_LIGHT),
        ]

        row_group = VGroup()
        for concept, value, desc, col in summary_rows:
            c_txt = label_text(concept, color=col)
            v_txt = label_text(value, color=WHITE)
            d_txt = label_text(desc, color=GREY_MED)
            v_txt.next_to(c_txt, RIGHT, buff=0.4)
            d_txt.next_to(v_txt, RIGHT, buff=0.4)
            row_group.add(VGroup(c_txt, v_txt, d_txt))

        row_group.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        row_group.scale_to_fit_width(12.8)
        row_group.move_to(ORIGIN + DOWN * 0.1)

        self.play(LaggedStart(*[FadeIn(r) for r in row_group], lag_ratio=0.1),
                  run_time=1.5)

        up_next = label_text("Up next: GRPO — same clipping idea, no critic, half the memory",
                             color=PURPLE_MED)
        up_next.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(up_next), run_time=0.5)
        self.wait(1.0)
        self.fade_all(summary_title, row_group, up_next)
        self.wait(1.0)
