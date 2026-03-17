"""
Scene 31 — Trust Regions & Open Problems
Run: manim -pql 31_trust_regions_open_problems.py TrustRegionsScene
"""
from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class TrustRegionsScene(LLMScene):
    def construct(self):

        # ── 1. Title ──────────────────────────────────────────────────────────
        title = self.show_title("Trust Regions & Open Problems",
                                "RL for LLMs — Final Chapter")
        self.wait(1.0)
        self.fade_all(title)

        # ── 2. Trust region concept: safe step size ───────────────────────────
        tr_title = body_text("Trust Regions: constrain how far the policy can move per update",
                             color=WHITE)
        tr_title.to_edge(UP, buff=0.6)
        self.play(Write(tr_title), run_time=0.8)

        # Current policy dot at centre
        current_dot = Circle(radius=0.15, color=BLUE_MED,
                             fill_color=BLUE_MED, fill_opacity=1.0)
        current_dot.move_to(ORIGIN + UP * 0.2)
        current_lbl = label_text("Current\nPolicy", color=BLUE_MED)
        current_lbl.next_to(current_dot, DOWN, buff=0.2)

        # Trust region circle
        trust_circle = Circle(radius=2.0, color=GREEN_MED,
                              stroke_width=2.5, fill_opacity=0.0)
        trust_circle.move_to(ORIGIN + UP * 0.2)
        trust_lbl = label_text("Trust Region\n(safe update zone)", color=GREEN_MED)
        trust_lbl.next_to(trust_circle, UP, buff=0.15)

        # Good update inside trust region
        good_dot = Circle(radius=0.12, color=GREEN_LIGHT,
                          fill_color=GREEN_LIGHT, fill_opacity=1.0)
        good_dot.move_to(ORIGIN + UP * 0.2 + RIGHT * 1.4 + UP * 0.8)
        good_lbl = label_text("Safe update\n(inside region)", color=GREEN_LIGHT)
        good_lbl.next_to(good_dot, RIGHT, buff=0.15)

        # Bad update outside trust region
        bad_dot = Circle(radius=0.12, color=RED_MED,
                         fill_color=RED_MED, fill_opacity=1.0)
        bad_dot.move_to(ORIGIN + UP * 0.2 + LEFT * 3.5 + DOWN * 1.2)
        bad_lbl = label_text("Unsafe update\n(outside region)", color=RED_MED)
        bad_lbl.next_to(bad_dot, LEFT, buff=0.15)

        # Arrow from current to good
        arrow_good = Arrow(current_dot.get_center(), good_dot.get_center(),
                           color=GREEN_MED, buff=0.15, stroke_width=2.0,
                           max_tip_length_to_length_ratio=0.2)
        # Arrow from current to bad
        arrow_bad = Arrow(current_dot.get_center(), bad_dot.get_center(),
                          color=RED_MED, buff=0.15, stroke_width=2.0,
                          max_tip_length_to_length_ratio=0.2)

        # epsilon label on radius
        eps_line = Line(ORIGIN + UP * 0.2,
                        ORIGIN + UP * 0.2 + RIGHT * 2.0,
                        color=YELLOW_MED, stroke_width=1.5)
        eps_lbl = label_text("epsilon (clip range)", color=YELLOW_MED)
        eps_lbl.next_to(eps_line, DOWN, buff=0.1)

        kl_note = label_text(
            "KL divergence measures distance between old policy and new policy",
            color=GREY_LIGHT,
        )
        kl_note.to_edge(DOWN, buff=0.4)

        self.play(Create(trust_circle), FadeIn(trust_lbl), run_time=0.7)
        self.play(FadeIn(current_dot), FadeIn(current_lbl), run_time=0.5)
        self.play(Create(eps_line), FadeIn(eps_lbl), run_time=0.5)
        self.play(Create(arrow_good), FadeIn(good_dot), FadeIn(good_lbl), run_time=0.6)
        self.play(Create(arrow_bad), FadeIn(bad_dot), FadeIn(bad_lbl), run_time=0.6)
        self.play(FadeIn(kl_note), run_time=0.5)
        self.wait(1.0)
        self.fade_all(tr_title, trust_circle, trust_lbl, current_dot, current_lbl,
                      eps_line, eps_lbl, arrow_good, good_dot, good_lbl,
                      arrow_bad, bad_dot, bad_lbl, kl_note)

        # ── 3. DPPO: decoupled actor and critic pipeline ──────────────────────
        dppo_title = body_text("DPPO: Decoupled Actor and Critic for Better GPU Utilization",
                               color=WHITE)
        dppo_title.to_edge(UP, buff=0.6)
        self.play(Write(dppo_title), run_time=0.8)

        # Standard PPO: sequential boxes
        std_lbl = label_text("Standard PPO — Sequential (GPU idle time)", color=RED_MED)
        std_lbl.move_to(UP * 2.0 + LEFT * 2.5)

        std_steps = ["Generate", "Score", "Compute\nAdvantage", "Update"]
        std_colors = [BLUE_MED, ORANGE_MED, YELLOW_MED, PURPLE_MED]
        std_boxes = VGroup()
        for s, c in zip(std_steps, std_colors):
            b = rounded_box(2.0, 0.7, fill_color=str(c) + "22",
                            stroke_color=c, label=s, label_color=c)
            std_boxes.add(b)
        std_boxes.arrange(RIGHT, buff=0.25)
        std_boxes.move_to(UP * 1.0)

        std_arrows = VGroup(*[
            Arrow(std_boxes[i].get_right(), std_boxes[i + 1].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.2)
            for i in range(3)
        ])

        # DPPO: async boxes
        dppo_lbl = label_text("DPPO — Async (actor and critic run in parallel)", color=GREEN_MED)
        dppo_lbl.move_to(DOWN * 0.3 + LEFT * 2.5)

        actor_box = rounded_box(3.5, 0.7, fill_color=str(BLUE_MED) + "22",
                                stroke_color=BLUE_MED,
                                label="Actor workers — Generate continuously",
                                label_color=BLUE_LIGHT)
        actor_box.move_to(LEFT * 1.5 + DOWN * 1.3)

        critic_box = rounded_box(3.5, 0.7, fill_color=str(ORANGE_MED) + "22",
                                 stroke_color=ORANGE_MED,
                                 label="Critic — Updates asynchronously",
                                 label_color=ORANGE_MED)
        critic_box.move_to(RIGHT * 2.8 + DOWN * 1.3)

        async_arrow = Arrow(actor_box.get_right(), critic_box.get_left(),
                            color=GREY_MED, buff=0.05, stroke_width=1.5,
                            max_tip_length_to_length_ratio=0.2)

        policy_lag_note = label_text(
            "Policy lag: actor may be ahead of critic — KL check discards stale samples",
            color=GREY_MED,
        )
        policy_lag_note.to_edge(DOWN, buff=0.4)

        self.play(FadeIn(std_lbl), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(b) for b in std_boxes], lag_ratio=0.15),
                  run_time=0.9)
        self.play(LaggedStart(*[Create(a) for a in std_arrows], lag_ratio=0.1),
                  run_time=0.5)
        self.play(FadeIn(dppo_lbl), run_time=0.4)
        self.play(FadeIn(actor_box), FadeIn(critic_box), run_time=0.6)
        self.play(Create(async_arrow), run_time=0.4)
        self.play(FadeIn(policy_lag_note), run_time=0.4)
        self.wait(1.0)
        self.fade_all(dppo_title, std_lbl, std_boxes, std_arrows,
                      dppo_lbl, actor_box, critic_box, async_arrow, policy_lag_note)

        # ── 4. ScaleRL: distributed training ─────────────────────────────────
        scale_title = body_text("ScaleRL: Distributed RL Across Thousands of GPUs",
                                color=WHITE)
        scale_title.to_edge(UP, buff=0.6)
        self.play(Write(scale_title), run_time=0.8)

        # Four roles
        role_data = [
            ("Actor\nWorkers\n(x many)", BLUE_MED,  [-5.0, 0.3, 0]),
            ("Reward\nModel",            ORANGE_MED, [-1.5, 0.3, 0]),
            ("Reference\nPolicy",        GREY_LIGHT, [ 1.5, 0.3, 0]),
            ("Trainer\n(optimizer)",     PURPLE_MED, [ 5.0, 0.3, 0]),
        ]

        role_boxes = {}
        role_group = VGroup()
        for name, col, pos in role_data:
            b = rounded_box(2.2, 1.0, fill_color=str(col) + "22",
                            stroke_color=col, label=name, label_color=col)
            b.move_to(pos)
            role_boxes[name] = b
            role_group.add(b)

        r1 = Arrow(role_boxes["Actor\nWorkers\n(x many)"].get_right(),
                   role_boxes["Reward\nModel"].get_left(),
                   color=GREY_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.18)
        r2 = Arrow(role_boxes["Reward\nModel"].get_right(),
                   role_boxes["Reference\nPolicy"].get_left(),
                   color=GREY_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.18)
        r3 = Arrow(role_boxes["Reference\nPolicy"].get_right(),
                   role_boxes["Trainer\n(optimizer)"].get_left(),
                   color=GREY_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.18)

        # Feedback arrow from trainer back to actors
        feedback = CurvedArrow(
            role_boxes["Trainer\n(optimizer)"].get_bottom(),
            role_boxes["Actor\nWorkers\n(x many)"].get_bottom(),
            angle=-TAU / 6, color=PURPLE_MED, stroke_width=1.5,
        )
        feedback_lbl = label_text("updated weights", color=PURPLE_MED)
        feedback_lbl.next_to(feedback, DOWN, buff=0.1)

        bw_note = label_text(
            "Bandwidth challenge: syncing 70B weights across actors = 140GB per update",
            color=GREY_MED,
        )
        bw_note.to_edge(DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(b) for b in role_group], lag_ratio=0.15),
                  run_time=1.0)
        self.play(Create(r1), Create(r2), Create(r3), run_time=0.6)
        self.play(Create(feedback), FadeIn(feedback_lbl), run_time=0.6)
        self.play(FadeIn(bw_note), run_time=0.4)
        self.wait(1.0)
        self.fade_all(scale_title, role_group, r1, r2, r3,
                      feedback, feedback_lbl, bw_note)

        # ── 5. Credit assignment problem ──────────────────────────────────────
        ca_title = body_text("Credit Assignment: which token caused the wrong answer?",
                             color=WHITE)
        ca_title.to_edge(UP, buff=0.6)
        self.play(Write(ca_title), run_time=0.8)

        # Token chain — 8 representative blocks for 1000 tokens
        token_labels = ["Tok 1", "Tok 50", "Tok 200", "Tok 400",
                        "...", "Tok 700", "Tok 950", "ANSWER"]
        token_colors = [GREEN_MED, GREEN_MED, YELLOW_MED, RED_MED,
                        GREY_MED, GREEN_MED, GREEN_MED, RED_MED]

        token_boxes = VGroup()
        for lbl, col in zip(token_labels, token_colors):
            b = rounded_box(1.35, 0.6, fill_color=str(col) + "33",
                            stroke_color=col, label=lbl, label_color=col)
            token_boxes.add(b)
        token_boxes.arrange(RIGHT, buff=0.15)
        token_boxes.move_to(UP * 0.6)

        chain_arrows = VGroup(*[
            Arrow(token_boxes[i].get_right(), token_boxes[i + 1].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.2,
                  max_tip_length_to_length_ratio=0.25)
            for i in range(7)
        ])

        # Single reward at end
        reward_box = rounded_box(2.0, 0.6, fill_color=str(RED_MED) + "22",
                                 stroke_color=RED_MED,
                                 label="Reward = 0\n(wrong answer)", label_color=RED_MED)
        reward_box.next_to(token_boxes, DOWN, buff=0.5)
        reward_arrow = Arrow(token_boxes[-1].get_bottom(), reward_box.get_top(),
                             color=RED_MED, buff=0.05, stroke_width=2.0,
                             max_tip_length_to_length_ratio=0.2)

        question = body_text("Which token was responsible?", color=YELLOW_MED)
        question.next_to(reward_box, DOWN, buff=0.35)

        ca_note = label_text(
            "Trajectory-level reward: all 1000 tokens penalized equally — even the correct ones",
            color=GREY_MED,
        )
        ca_note.to_edge(DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(b) for b in token_boxes], lag_ratio=0.08),
                  run_time=1.0)
        self.play(LaggedStart(*[Create(a) for a in chain_arrows], lag_ratio=0.06),
                  run_time=0.7)
        self.play(Create(reward_arrow), FadeIn(reward_box), run_time=0.5)
        self.play(Write(question), run_time=0.6)
        self.play(FadeIn(ca_note), run_time=0.4)
        self.wait(1.0)
        self.fade_all(ca_title, token_boxes, chain_arrows,
                      reward_arrow, reward_box, question, ca_note)

        # ── 6. Reward overoptimization — Goodhart's Law ───────────────────────
        ro_title = body_text("Reward Overoptimization: Goodhart's Law in RL for LLMs",
                             color=WHITE)
        ro_title.to_edge(UP, buff=0.6)
        self.play(Write(ro_title), run_time=0.8)

        # Two diverging lines: proxy reward going up, true reward going up then down
        # Represented as labelled boxes at different positions

        axis_origin = LEFT * 4.5 + DOWN * 1.0
        axis_end_x  = RIGHT * 3.5 + DOWN * 1.0
        axis_end_y  = LEFT * 4.5 + UP * 1.8

        x_axis = Arrow(axis_origin, axis_end_x, color=GREY_MED,
                       buff=0.0, stroke_width=1.5,
                       max_tip_length_to_length_ratio=0.05)
        y_axis = Arrow(axis_origin, axis_end_y, color=GREY_MED,
                       buff=0.0, stroke_width=1.5,
                       max_tip_length_to_length_ratio=0.08)
        x_axis_lbl = label_text("RL training steps (KL from reference)", color=GREY_MED)
        x_axis_lbl.next_to(x_axis, DOWN, buff=0.18)
        y_axis_lbl = label_text("Reward", color=GREY_MED)
        y_axis_lbl.next_to(y_axis, LEFT, buff=0.15)

        # Proxy reward: rises consistently
        proxy_points = [
            axis_origin + RIGHT * 0.0 + UP * 0.1,
            axis_origin + RIGHT * 1.5 + UP * 0.8,
            axis_origin + RIGHT * 3.0 + UP * 1.5,
            axis_origin + RIGHT * 5.0 + UP * 2.1,
            axis_origin + RIGHT * 7.0 + UP * 2.5,
        ]
        proxy_line = VMobject(color=ORANGE_MED, stroke_width=2.5)
        proxy_line.set_points_smoothly([p for p in proxy_points])
        proxy_end_lbl = label_text("Proxy reward\n(reward model score)", color=ORANGE_MED)
        proxy_end_lbl.next_to(proxy_points[-1], RIGHT, buff=0.15)

        # True reward: rises then falls
        true_points = [
            axis_origin + RIGHT * 0.0 + UP * 0.1,
            axis_origin + RIGHT * 1.5 + UP * 0.9,
            axis_origin + RIGHT * 3.0 + UP * 1.6,
            axis_origin + RIGHT * 4.5 + UP * 1.4,
            axis_origin + RIGHT * 6.0 + UP * 0.9,
            axis_origin + RIGHT * 7.0 + UP * 0.4,
        ]
        true_line = VMobject(color=GREEN_MED, stroke_width=2.5)
        true_line.set_points_smoothly([p for p in true_points])
        true_end_lbl = label_text("True reward\n(human eval)", color=GREEN_MED)
        true_end_lbl.next_to(true_points[-1], RIGHT, buff=0.15)

        goodhart_note = label_text(
            "Goodhart's Law: when a measure becomes the target, it ceases to be a good measure",
            color=YELLOW_MED,
        )
        goodhart_note.to_edge(DOWN, buff=0.4)

        self.play(Create(x_axis), Create(y_axis),
                  FadeIn(x_axis_lbl), FadeIn(y_axis_lbl), run_time=0.6)
        self.play(Create(proxy_line), FadeIn(proxy_end_lbl), run_time=0.8)
        self.play(Create(true_line), FadeIn(true_end_lbl), run_time=0.8)
        self.play(FadeIn(goodhart_note), run_time=0.5)
        self.wait(1.0)
        self.fade_all(ro_title, x_axis, y_axis, x_axis_lbl, y_axis_lbl,
                      proxy_line, proxy_end_lbl, true_line, true_end_lbl, goodhart_note)

        # ── 7. Length hacking ────────────────────────────────────────────────
        lh_title = body_text("Length Hacking: verbosity rewarded even without quality",
                             color=WHITE)
        lh_title.to_edge(UP, buff=0.6)
        self.play(Write(lh_title), run_time=0.8)

        # Scatter-like display using rectangles: x=length, y=quality
        # Short + high quality (good)
        lh_axis_origin = LEFT * 5.0 + DOWN * 1.5
        lh_x_end = RIGHT * 5.5 + DOWN * 1.5
        lh_y_end = LEFT * 5.0 + UP * 2.0

        lh_x = Arrow(lh_axis_origin, lh_x_end, color=GREY_MED,
                     buff=0.0, stroke_width=1.5,
                     max_tip_length_to_length_ratio=0.04)
        lh_y = Arrow(lh_axis_origin, lh_y_end, color=GREY_MED,
                     buff=0.0, stroke_width=1.5,
                     max_tip_length_to_length_ratio=0.06)
        lh_x_lbl = label_text("Response length (tokens)", color=GREY_MED)
        lh_x_lbl.next_to(lh_x, DOWN, buff=0.18)
        lh_y_lbl = label_text("Actual quality", color=GREY_MED)
        lh_y_lbl.next_to(lh_y, LEFT, buff=0.15)

        # Good responses: short-medium, high quality
        good_pts = [
            lh_axis_origin + RIGHT * 1.0 + UP * 2.8,
            lh_axis_origin + RIGHT * 1.8 + UP * 2.5,
            lh_axis_origin + RIGHT * 2.5 + UP * 3.0,
            lh_axis_origin + RIGHT * 3.0 + UP * 2.7,
        ]
        good_dots_lh = VGroup(*[
            Dot(point=p, radius=0.12, color=GREEN_MED, fill_opacity=0.9)
            for p in good_pts
        ])

        # Length-hacked responses: very long, mediocre quality
        hack_pts = [
            lh_axis_origin + RIGHT * 6.5 + UP * 1.4,
            lh_axis_origin + RIGHT * 7.5 + UP * 1.0,
            lh_axis_origin + RIGHT * 8.5 + UP * 0.8,
            lh_axis_origin + RIGHT * 9.0 + UP * 1.2,
        ]
        hack_dots_lh = VGroup(*[
            Dot(point=p, radius=0.12, color=RED_MED, fill_opacity=0.9)
            for p in hack_pts
        ])

        good_region_lbl = label_text("Concise + correct", color=GREEN_MED)
        good_region_lbl.next_to(good_dots_lh, UP, buff=0.15)
        hack_region_lbl = label_text("Verbose + mediocre\n(reward-hacked)", color=RED_MED)
        hack_region_lbl.next_to(hack_dots_lh, DOWN, buff=0.2)

        dr_grpo_note = label_text(
            "Dr. GRPO fix: normalize advantages within length-matched groups",
            color=YELLOW_MED,
        )
        dr_grpo_note.to_edge(DOWN, buff=0.4)

        self.play(Create(lh_x), Create(lh_y),
                  FadeIn(lh_x_lbl), FadeIn(lh_y_lbl), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(d) for d in good_dots_lh], lag_ratio=0.1),
                  FadeIn(good_region_lbl), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(d) for d in hack_dots_lh], lag_ratio=0.1),
                  FadeIn(hack_region_lbl), run_time=0.7)
        self.play(FadeIn(dr_grpo_note), run_time=0.4)
        self.wait(1.0)
        self.fade_all(lh_title, lh_x, lh_y, lh_x_lbl, lh_y_lbl,
                      good_dots_lh, good_region_lbl, hack_dots_lh,
                      hack_region_lbl, dr_grpo_note)

        # ── 8. Sparse rewards ─────────────────────────────────────────────────
        sr_title = body_text("Sparse Rewards: signal arrives only at end of long trajectory",
                             color=WHITE)
        sr_title.to_edge(UP, buff=0.6)
        self.play(Write(sr_title), run_time=0.8)

        # Long token bar with reward signal only at the end
        bar_y = UP * 0.5
        traj_bar = Rectangle(width=11.0, height=0.7,
                             fill_color=str(BLUE_MED) + "22",
                             stroke_color=BLUE_MED, stroke_width=1.5,
                             fill_opacity=1.0)
        traj_bar.move_to(bar_y)
        traj_lbl = label_text("1000-token reasoning chain", color=BLUE_LIGHT)
        traj_lbl.next_to(traj_bar, UP, buff=0.2)

        # Reward box at the right end
        reward_sr = rounded_box(1.8, 0.65, fill_color=str(ORANGE_MED) + "33",
                                stroke_color=ORANGE_MED,
                                label="Reward\n= 1", label_color=ORANGE_MED)
        reward_sr.next_to(traj_bar, RIGHT, buff=0.1)
        reward_arrow_sr = Arrow(traj_bar.get_right(), reward_sr.get_left(),
                                color=ORANGE_MED, buff=0.05, stroke_width=2.0,
                                max_tip_length_to_length_ratio=0.2)

        # "No signal here" markers
        no_sig_1 = label_text("No signal here", color=RED_MED)
        no_sig_1.next_to(traj_bar, DOWN, buff=0.25)
        no_sig_1.shift(LEFT * 3.5)
        no_sig_2 = label_text("No signal here", color=RED_MED)
        no_sig_2.next_to(traj_bar, DOWN, buff=0.25)

        prm_box = rounded_box(8.0, 0.7, fill_color=str(GREEN_MED) + "22",
                              stroke_color=GREEN_MED,
                              label="Process Reward Model: per-step signal — denser but expensive to train",
                              label_color=GREEN_LIGHT)
        prm_box.move_to(DOWN * 2.0)

        sr_note = label_text(
            "Discount factor at 1.0: all tokens treated equally — avoids vanishing signal",
            color=GREY_MED,
        )
        sr_note.to_edge(DOWN, buff=0.4)

        self.play(FadeIn(traj_bar), FadeIn(traj_lbl), run_time=0.6)
        self.play(Create(reward_arrow_sr), FadeIn(reward_sr), run_time=0.5)
        self.play(FadeIn(no_sig_1), FadeIn(no_sig_2), run_time=0.5)
        self.play(FadeIn(prm_box), run_time=0.5)
        self.play(FadeIn(sr_note), run_time=0.4)
        self.wait(1.0)
        self.fade_all(sr_title, traj_bar, traj_lbl, reward_arrow_sr, reward_sr,
                      no_sig_1, no_sig_2, prm_box, sr_note)

        # ── 9. Open problems as cards ─────────────────────────────────────────
        op_title = body_text("Open Research Questions — What We Still Don't Know",
                             color=WHITE)
        op_title.to_edge(UP, buff=0.6)
        self.play(Write(op_title), run_time=0.8)

        open_problems = [
            ("PRM without\nhuman labels", BLUE_MED,
             "Can we train process\nreward models without\nexpensive annotations?"),
            ("Credit assign\nlong contexts", ORANGE_MED,
             "32k+ token chains:\nwhich tokens caused\nwhich outcomes?"),
            ("Overoptimization\nat scale", RED_MED,
             "Scaling laws for\nreward hacking in\n70B+ models?"),
            ("Multi-turn RL", PURPLE_MED,
             "Reward across many\nconversation turns,\nnot just one response"),
            ("Non-verifiable\nrewards", YELLOW_MED,
             "RL for writing/advice\nwithout auto-checkers\nfor correctness"),
            ("Reasoning vs\nmemorization", GREEN_MED,
             "Does RL improve genuine\nreasoning or pattern\ncaching?"),
        ]

        cards = VGroup()
        for name, col, detail in open_problems:
            header = body_text(name, color=col)
            detail_txt = label_text(detail, color=GREY_LIGHT)
            detail_txt.next_to(header, DOWN, buff=0.18)
            content = VGroup(header, detail_txt)
            bg = SurroundingRectangle(content, color=col,
                                      fill_color=str(col) + "11",
                                      fill_opacity=1.0, buff=0.25,
                                      corner_radius=0.12)
            cards.add(VGroup(bg, content))

        cards.arrange_in_grid(rows=2, cols=3, buff=0.35)
        cards.scale_to_fit_width(13.0)
        cards.move_to(DOWN * 0.3)

        self.play(LaggedStart(*[FadeIn(c) for c in cards], lag_ratio=0.12),
                  run_time=1.8)
        self.wait(1.0)
        self.fade_all(op_title, cards)

        # ── 10. The 2024-2025 landscape ───────────────────────────────────────
        land_title = body_text("The 2024-2025 Landscape: DeepSeek-R1 and OpenAI o1",
                               color=WHITE)
        land_title.to_edge(UP, buff=0.6)
        self.play(Write(land_title), run_time=0.8)

        model_data = [
            ("OpenAI o1\nSep 2024",      BLUE_MED,   "RL + verifiable rewards\nLong chain-of-thought\nProprietary training"),
            ("OpenAI o3\nDec 2024",      BLUE_LIGHT,  "Extended test-time compute\nARC-AGI breakthrough\nProcess supervision (likely)"),
            ("DeepSeek-R1\nJan 2025",    GREEN_MED,  "GRPO, open weights\nEmergent CoT from RL\nMatches o1 on benchmarks"),
        ]

        model_boxes = VGroup()
        for name, col, detail in model_data:
            header = body_text(name, color=col)
            detail_txt = label_text(detail, color=GREY_LIGHT)
            detail_txt.next_to(header, DOWN, buff=0.2)
            content = VGroup(header, detail_txt)
            bg = SurroundingRectangle(content, color=col,
                                      fill_color=str(col) + "11",
                                      fill_opacity=1.0, buff=0.3,
                                      corner_radius=0.12)
            model_boxes.add(VGroup(bg, content))

        model_boxes.arrange(RIGHT, buff=0.5)
        model_boxes.scale_to_fit_width(13.0)
        model_boxes.move_to(UP * 0.5)

        known_box = rounded_box(6.0, 0.7, fill_color=str(GREEN_MED) + "22",
                                stroke_color=GREEN_MED,
                                label="Known: RL improves math/code; KL constraints essential; CoT emerges",
                                label_color=GREEN_LIGHT)
        known_box.move_to(DOWN * 1.9)

        unknown_box = rounded_box(6.5, 0.7, fill_color=str(RED_MED) + "22",
                                  stroke_color=RED_MED,
                                  label="Unknown: scaling laws for RL; optimal RL:SFT ratio; general task transfer",
                                  label_color=RED_MED)
        unknown_box.move_to(DOWN * 2.9)

        self.play(LaggedStart(*[FadeIn(b) for b in model_boxes], lag_ratio=0.2),
                  run_time=1.2)
        self.play(FadeIn(known_box), run_time=0.5)
        self.play(FadeIn(unknown_box), run_time=0.5)
        self.wait(1.0)
        self.fade_all(land_title, model_boxes, known_box, unknown_box)

        # ── 11. Full RL algorithm family tree ────────────────────────────────
        tree_title = body_text("The RL Algorithm Family Tree — Series Summary",
                               color=WHITE)
        tree_title.to_edge(UP, buff=0.6)
        self.play(Write(tree_title), run_time=0.8)

        # Root
        pg_box = rounded_box(2.8, 0.65, fill_color=str(GREY_LIGHT) + "22",
                             stroke_color=GREY_LIGHT,
                             label="Policy Gradient\n(REINFORCE)", label_color=GREY_LIGHT)
        pg_box.move_to(UP * 1.8)

        # Level 2
        ppo_fam = rounded_box(2.5, 0.65, fill_color=str(BLUE_MED) + "22",
                              stroke_color=BLUE_MED,
                              label="PPO\n(clipped ratio)", label_color=BLUE_LIGHT)
        ppo_fam.move_to(UP * 0.4 + LEFT * 3.5)

        grpo_fam = rounded_box(2.5, 0.65, fill_color=str(ORANGE_MED) + "22",
                               stroke_color=ORANGE_MED,
                               label="GRPO\n(no critic)", label_color=ORANGE_MED)
        grpo_fam.move_to(UP * 0.4 + RIGHT * 0.0)

        dapo_fam = rounded_box(2.5, 0.65, fill_color=str(PURPLE_MED) + "22",
                               stroke_color=PURPLE_MED,
                               label="DAPO\n(dynamic clip)", label_color=PURPLE_MED)
        dapo_fam.move_to(UP * 0.4 + RIGHT * 3.5)

        # Level 3
        dppo_fam = rounded_box(2.5, 0.65, fill_color=str(BLUE_DARK) + "88",
                               stroke_color=BLUE_MED,
                               label="DPPO\n(decoupled)", label_color=BLUE_LIGHT)
        dppo_fam.move_to(DOWN * 1.0 + LEFT * 3.5)

        drgrpo_fam = rounded_box(2.5, 0.65, fill_color=str(ORANGE_DARK) + "88",
                                 stroke_color=ORANGE_MED,
                                 label="Dr. GRPO\n(length norm)", label_color=ORANGE_MED)
        drgrpo_fam.move_to(DOWN * 1.0 + RIGHT * 0.0)

        scalerl_fam = rounded_box(2.5, 0.65, fill_color=str(PURPLE_MED) + "33",
                                  stroke_color=PURPLE_MED,
                                  label="ScaleRL\n(1000s of GPUs)", label_color=PURPLE_MED)
        scalerl_fam.move_to(DOWN * 1.0 + RIGHT * 3.5)

        tree_arrows = VGroup(
            Arrow(pg_box.get_bottom(), ppo_fam.get_top(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.2),
            Arrow(pg_box.get_bottom(), grpo_fam.get_top(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.2),
            Arrow(pg_box.get_bottom(), dapo_fam.get_top(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.2),
            Arrow(ppo_fam.get_bottom(), dppo_fam.get_top(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.2),
            Arrow(grpo_fam.get_bottom(), drgrpo_fam.get_top(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.2),
            Arrow(dapo_fam.get_bottom(), scalerl_fam.get_top(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.2),
        )

        all_tree = VGroup(pg_box, ppo_fam, grpo_fam, dapo_fam,
                          dppo_fam, drgrpo_fam, scalerl_fam)
        all_tree.move_to(ORIGIN + DOWN * 0.2)

        self.play(FadeIn(pg_box), run_time=0.5)
        self.play(LaggedStart(
            *[Create(a) for a in tree_arrows[:3]],
            *[FadeIn(b) for b in [ppo_fam, grpo_fam, dapo_fam]],
            lag_ratio=0.15), run_time=1.0)
        self.play(LaggedStart(
            *[Create(a) for a in tree_arrows[3:]],
            *[FadeIn(b) for b in [dppo_fam, drgrpo_fam, scalerl_fam]],
            lag_ratio=0.15), run_time=0.9)
        self.wait(1.0)
        self.fade_all(tree_title, all_tree, tree_arrows)

        # ── 12. Closing card ──────────────────────────────────────────────────
        close_title = title_text("Series Complete", color=WHITE)
        close_title.move_to(UP * 1.8)
        self.play(Write(close_title), run_time=1.0)

        summary_lines = [
            "Pretraining gives capability",
            "SFT gives format and behaviour",
            "Reward models give preferences",
            "RL gives alignment and reasoning",
            "Trust regions make RL stable",
        ]
        summary_group = VGroup()
        for i, line in enumerate(summary_lines):
            col = [GREY_LIGHT, BLUE_LIGHT, ORANGE_MED, GREEN_MED, YELLOW_MED][i]
            t = body_text(line, color=col)
            summary_group.add(t)
        summary_group.arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        summary_group.move_to(DOWN * 0.3)

        frontier_note = label_text(
            "Open problems remain — the field is advancing rapidly",
            color=GREY_MED,
        )
        frontier_note.to_edge(DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(t) for t in summary_group], lag_ratio=0.18),
                  run_time=1.4)
        self.play(FadeIn(frontier_note), run_time=0.5)
        self.wait(2.0)
        self.fade_all(close_title, summary_group, frontier_note)
