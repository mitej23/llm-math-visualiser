"""
Scene 20 — KL Divergence
Run: manim -pql 20_kl_divergence.py KLDivergenceScene
"""

from manim import *
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class KLDivergenceScene(LLMScene):
    def construct(self):
        title = self.show_title("KL Divergence", "Don't Stray Too Far")
        self.wait(0.5)
        self.fade_all(title)

        # ── 1. Two distributions — similar vs different ────────────────────────
        dist_title = body_text("KL Divergence measures how different two distributions are:",
                               color=WHITE)
        dist_title.to_edge(UP, buff=0.6)
        self.play(Write(dist_title), run_time=0.7)

        tokens = ["cat", "dog", "fish", "bird"]
        ref_probs  = [0.45, 0.30, 0.15, 0.10]   # reference policy
        close_probs = [0.42, 0.32, 0.16, 0.10]  # small KL
        far_probs   = [0.10, 0.05, 0.70, 0.15]  # large KL

        bar_w = 0.6
        y_scale = 3.0

        def make_dist_bars(probs, col, x_offset):
            bars = VGroup()
            for i, (tok, p) in enumerate(zip(tokens, probs)):
                h = p * y_scale
                bar = Rectangle(width=bar_w, height=h,
                                 fill_color=col, fill_opacity=0.8,
                                 stroke_color=col, stroke_width=1)
                bar.shift(RIGHT * (i * (bar_w + 0.2) + x_offset) + UP * (h / 2 - 1.0))
                lbl = label_text(tok, color=col)
                lbl.next_to(bar, DOWN, buff=0.1)
                bars.add(VGroup(bar, lbl))
            return bars

        ref_bars   = make_dist_bars(ref_probs,   BLUE_MED, -5.5)
        close_bars = make_dist_bars(close_probs, GREEN_MED, -1.2)
        far_bars   = make_dist_bars(far_probs,   RED_MED,   3.0)

        ref_label   = label_text("Reference\n(frozen SFT)", color=BLUE_MED)
        close_label = label_text("RL policy\n(small KL ✅)", color=GREEN_MED)
        far_label   = label_text("RL policy\n(large KL ❌)", color=RED_MED)

        ref_label.next_to(ref_bars, UP, buff=0.2)
        close_label.next_to(close_bars, UP, buff=0.2)
        far_label.next_to(far_bars, UP, buff=0.2)

        self.play(LaggedStart(*[FadeIn(b) for b in ref_bars],   lag_ratio=0.1),
                  FadeIn(ref_label), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(b) for b in close_bars], lag_ratio=0.1),
                  FadeIn(close_label), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(b) for b in far_bars],   lag_ratio=0.1),
                  FadeIn(far_label), run_time=0.8)

        kl_note = label_text(
            "Small KL → model improved but still 'itself'     "
            "Large KL → model has changed too dramatically",
            color=GREY_LIGHT,
        )
        kl_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(kl_note), run_time=0.5)
        self.wait(1.2)
        self.fade_all(dist_title, ref_bars, ref_label,
                      close_bars, close_label, far_bars, far_label, kl_note)

        # ── 2. Two smooth distributions on axes — area between = divergence ──
        curves_title = body_text("P = reference distribution   Q = new (RL) distribution",
                                 color=WHITE)
        curves_title.to_edge(UP, buff=0.6)
        self.play(Write(curves_title), run_time=0.7)

        ax = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 0.55, 0.1],
            x_length=10,
            y_length=4.5,
            axis_config={"color": GREY_MED, "stroke_width": 1.5},
            tips=False,
        )
        ax.move_to(ORIGIN + DOWN * 0.4)

        def gaussian(x, mu, sigma):
            return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

        p_curve = ax.plot(lambda x: gaussian(x, 2.0, 0.8), color=BLUE_MED, stroke_width=2.5)
        q_curve = ax.plot(lambda x: gaussian(x, 2.6, 0.9), color=GREEN_MED, stroke_width=2.5)

        p_label = label_text("P  (reference)", color=BLUE_MED)
        q_label = label_text("Q  (RL policy)", color=GREEN_MED)
        p_label.next_to(ax.i2gp(1.5, p_curve), UP, buff=0.2)
        q_label.next_to(ax.i2gp(3.2, q_curve), UP, buff=0.2)

        # Shade the area between curves to represent divergence
        area = ax.get_area(p_curve, x_range=[0, 6], color=BLUE_MED, opacity=0.15)
        area_q = ax.get_area(q_curve, x_range=[0, 6], color=GREEN_MED, opacity=0.15)

        self.play(Create(ax), run_time=0.6)
        self.play(Create(p_curve), FadeIn(p_label), FadeIn(area), run_time=0.7)
        self.play(Create(q_curve), FadeIn(q_label), FadeIn(area_q), run_time=0.7)

        div_note = label_text(
            "Where P is high but Q is low → P gets 'surprised' → large KL contribution",
            color=ORANGE_MED,
        )
        div_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(div_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(curves_title, ax, p_curve, q_curve, p_label, q_label,
                      area, area_q, div_note)

        # ── 3. KL is asymmetric ───────────────────────────────────────────────
        asym_title = body_text("KL is asymmetric: KL(P || Q)  ≠  KL(Q || P)", color=WHITE)
        asym_title.to_edge(UP, buff=0.6)
        self.play(Write(asym_title), run_time=0.7)

        # Show two boxes side by side explaining the two directions
        kl_pq = rounded_box(4.5, 1.6, fill_color=str(BLUE_MED) + "22",
                            stroke_color=BLUE_MED,
                            label="KL( P || Q )\n\"Forward KL\"\nIf reality is P, using Q\nPenalises: P high, Q low",
                            label_color=BLUE_LIGHT)
        kl_pq.shift(LEFT * 3.3 + DOWN * 0.2)

        kl_qp = rounded_box(4.5, 1.6, fill_color=str(GREEN_MED) + "22",
                            stroke_color=GREEN_MED,
                            label="KL( Q || P )\n\"Reverse KL\"\nIf reality is Q, using P\nPenalises: Q high, P low",
                            label_color=GREEN_LIGHT)
        kl_qp.shift(RIGHT * 3.3 + DOWN * 0.2)

        neq_label = body_text("≠", color=YELLOW_MED)
        neq_label.move_to(ORIGIN + DOWN * 0.2)

        self.play(FadeIn(kl_pq), FadeIn(kl_qp), run_time=0.7)
        self.play(FadeIn(neq_label), run_time=0.4)

        rlhf_note = label_text(
            "RLHF uses KL(RL policy || Reference)  — penalises RL policy for tokens the reference finds very unlikely",
            color=ORANGE_MED,
        )
        rlhf_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(rlhf_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(asym_title, kl_pq, kl_qp, neq_label, rlhf_note)

        # ── 4. KL penalty in the reward formula ───────────────────────────────
        formula_title = body_text("The RLHF reward formula:", color=WHITE)
        formula_title.to_edge(UP, buff=0.6)
        self.play(Write(formula_title), run_time=0.6)

        parts = [
            ("Total reward", WHITE),
            ("  =  ", GREY_LIGHT),
            ("Reward model score", GREEN_MED),
            ("  -  beta x  ", GREY_LIGHT),
            ("KL( RL policy || Reference )", ORANGE_MED),
        ]
        formula_row = VGroup(*[body_text(t, color=c) for t, c in parts])
        formula_row.arrange(RIGHT, buff=0.05)
        formula_row.move_to(ORIGIN + UP * 0.5)

        reward_note = label_text("be helpful",       color=GREEN_MED)
        kl_note2    = label_text("don't change too much", color=ORANGE_MED)
        reward_note.next_to(formula_row[2], DOWN, buff=0.3)
        kl_note2.next_to(formula_row[4], DOWN, buff=0.3)

        self.play(LaggedStart(*[FadeIn(p) for p in formula_row], lag_ratio=0.1),
                  run_time=1.0)
        self.play(FadeIn(reward_note), FadeIn(kl_note2), run_time=0.6)
        self.wait(1.2)

        source_note = label_text(
            "Source: Lambert et al., HuggingFace RLHF Blog (2022)  &  "
            "Huang et al., N+ Implementation Details (2023)",
            color=GREY_MED,
        )
        source_note.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(source_note), run_time=0.5)
        self.wait(1)
        self.fade_all(formula_title, formula_row, reward_note, kl_note2, source_note)

        # ── 5. KL in RLHF — two bar charts over tokens ────────────────────────
        rl_kl_title = body_text("KL penalty: compare RL policy vs reference per token",
                                color=WHITE)
        rl_kl_title.to_edge(UP, buff=0.6)
        self.play(Write(rl_kl_title), run_time=0.7)

        tok_labels = ["Paris", "great", "city", "amazing", "!"]
        ref_tok  = [0.52, 0.18, 0.16, 0.08, 0.06]
        rl_tok   = [0.50, 0.17, 0.15, 0.12, 0.06]  # small change
        hack_tok = [0.05, 0.03, 0.04, 0.70, 0.18]  # reward-hacked

        bar_w2 = 0.55
        ys = 3.5

        def make_bars_centered(probs, col, x_off, y_floor):
            grp = VGroup()
            for i, (lbl, p) in enumerate(zip(tok_labels, probs)):
                h = p * ys
                bar = Rectangle(width=bar_w2, height=max(h, 0.03),
                                 fill_color=col, fill_opacity=0.8,
                                 stroke_color=col, stroke_width=1)
                bar.move_to([x_off + i * (bar_w2 + 0.18), y_floor + h / 2, 0])
                l = label_text(lbl, color=col)
                l.next_to(bar, DOWN, buff=0.08)
                grp.add(VGroup(bar, l))
            return grp

        ref_bars2 = make_bars_centered(ref_tok,  BLUE_MED,  -5.8, -1.5)
        rl_bars2  = make_bars_centered(rl_tok,   GREEN_MED, -1.0, -1.5)
        hack_bars = make_bars_centered(hack_tok, RED_MED,    3.5, -1.5)

        ref_hdr  = label_text("Reference policy", color=BLUE_MED)
        rl_hdr   = label_text("RL policy (low KL)", color=GREEN_MED)
        hack_hdr = label_text("Reward-hacked (high KL)", color=RED_MED)

        ref_hdr.next_to(ref_bars2, UP, buff=0.25)
        rl_hdr.next_to(rl_bars2, UP, buff=0.25)
        hack_hdr.next_to(hack_bars, UP, buff=0.25)

        self.play(LaggedStart(*[FadeIn(b) for b in ref_bars2], lag_ratio=0.07),
                  FadeIn(ref_hdr), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(b) for b in rl_bars2], lag_ratio=0.07),
                  FadeIn(rl_hdr), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(b) for b in hack_bars], lag_ratio=0.07),
                  FadeIn(hack_hdr), run_time=0.7)

        kl_compare_note = label_text(
            "Per-token KL summed over the response.  Reward-hacked policy: high KL → large penalty",
            color=GREY_LIGHT,
        )
        kl_compare_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(kl_compare_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(rl_kl_title, ref_bars2, ref_hdr, rl_bars2, rl_hdr,
                      hack_bars, hack_hdr, kl_compare_note)

        # ── 6. What happens when KL is too large ──────────────────────────────
        large_kl_title = body_text("When KL is too large: reward hacking breakdown",
                                   color=RED_MED)
        large_kl_title.to_edge(UP, buff=0.6)
        self.play(Write(large_kl_title), run_time=0.7)

        stages = [
            ("Stage 1\nEarly RL", GREEN_MED, "KL small\nModel improves genuinely"),
            ("Stage 2\nMid RL",   YELLOW_MED, "KL growing\nReward scores rising"),
            ("Stage 3\nLate RL",  RED_MED,   "KL large\nReward high, quality crashes"),
        ]
        stage_boxes = VGroup()
        for lbl, col, note in stages:
            b = rounded_box(3.0, 1.0, fill_color=str(col) + "22",
                            stroke_color=col, label=lbl, label_color=WHITE)
            n = label_text(note, color=GREY_LIGHT)
            n.next_to(b, DOWN, buff=0.2)
            stage_boxes.add(VGroup(b, n))

        stage_boxes.arrange(RIGHT, buff=0.8)
        stage_boxes.move_to(ORIGIN + UP * 0.5)

        st_arrows = VGroup(*[
            Arrow(stage_boxes[i][0].get_right(), stage_boxes[i + 1][0].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.2)
            for i in range(2)
        ])

        self.play(LaggedStart(*[FadeIn(b) for b in stage_boxes], lag_ratio=0.3),
                  run_time=1.0)
        self.play(LaggedStart(*[Create(a) for a in st_arrows], lag_ratio=0.3),
                  run_time=0.5)

        kl_solution = rounded_box(5.5, 0.65, fill_color=GREEN_DARK,
                                  stroke_color=GREEN_MED,
                                  label="KL penalty prevents Stage 3 — keeps model in useful range",
                                  label_color=WHITE)
        kl_solution.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(kl_solution), run_time=0.5)
        self.wait(1.5)
        self.fade_all(large_kl_title, stage_boxes, st_arrows, kl_solution)

        # ── 7. Adaptive KL control ────────────────────────────────────────────
        adapt_title = body_text("Adaptive KL: auto-tune the leash tightness", color=WHITE)
        adapt_title.to_edge(UP, buff=0.6)
        self.play(Write(adapt_title), run_time=0.7)

        cases = [
            ("KL > target",  RED_MED,   "Tighten leash  (increase beta) 🐕‍🦺"),
            ("KL ≈ target",  GREEN_MED, "Keep as-is  ✅"),
            ("KL < target",  BLUE_MED,  "Loosen leash  (decrease beta) 🐕"),
        ]
        case_rows = VGroup()
        for cond, col, action in cases:
            c = body_text(cond, color=col)
            a = label_text(action, color=WHITE)
            a.next_to(c, RIGHT, buff=0.5)
            case_rows.add(VGroup(c, a))

        case_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        case_rows.move_to(ORIGIN + DOWN * 0.2)
        box = SurroundingRectangle(case_rows, color=GREY_MED, buff=0.3, corner_radius=0.12)

        target_note = label_text("Target KL ≈ 6.0 (typical default)", color=GREY_LIGHT)
        target_note.next_to(box, UP, buff=0.25)

        self.play(Write(target_note), Create(box), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(r) for r in case_rows], lag_ratio=0.25),
                  run_time=1.0)
        self.wait(1.5)
        self.fade_all(adapt_title, case_rows, box, target_note)

        # ── 8. DPO — replacing explicit KL with implicit ──────────────────────
        dpo_title = body_text("DPO: KL baked into loss — no explicit reward model needed",
                              color=WHITE)
        dpo_title.to_edge(UP, buff=0.6)
        self.play(Write(dpo_title), run_time=0.7)

        rlhf_flow = VGroup(
            rounded_box(2.2, 0.75, fill_color=str(BLUE_MED) + "22",
                        stroke_color=BLUE_MED, label="SFT\nmodel", label_color=BLUE_LIGHT),
            rounded_box(2.2, 0.75, fill_color=str(ORANGE_MED) + "22",
                        stroke_color=ORANGE_MED, label="Reward\nModel", label_color=WHITE),
            rounded_box(2.2, 0.75, fill_color=str(PURPLE_MED) + "22",
                        stroke_color=PURPLE_MED, label="PPO +\nKL penalty", label_color=WHITE),
            rounded_box(2.2, 0.75, fill_color=GREEN_DARK,
                        stroke_color=GREEN_MED, label="Aligned\nLLM", label_color=GREEN_LIGHT),
        )
        rlhf_flow.arrange(RIGHT, buff=0.4)
        rlhf_flow.move_to(ORIGIN + UP * 1.2)

        rlhf_arrows = VGroup(*[
            Arrow(rlhf_flow[i].get_right(), rlhf_flow[i + 1].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.2)
            for i in range(3)
        ])

        rlhf_lbl = label_text("RLHF: 3 stages", color=GREY_LIGHT)
        rlhf_lbl.next_to(rlhf_flow, LEFT, buff=0.3)

        self.play(LaggedStart(*[FadeIn(b) for b in rlhf_flow], lag_ratio=0.15),
                  FadeIn(rlhf_lbl), run_time=0.9)
        self.play(LaggedStart(*[Create(a) for a in rlhf_arrows], lag_ratio=0.15),
                  run_time=0.5)

        dpo_flow = VGroup(
            rounded_box(2.2, 0.75, fill_color=str(BLUE_MED) + "22",
                        stroke_color=BLUE_MED, label="SFT\nmodel", label_color=BLUE_LIGHT),
            rounded_box(3.0, 0.75, fill_color=str(GREEN_MED) + "22",
                        stroke_color=GREEN_MED,
                        label="DPO loss\n(KL implicit)", label_color=GREEN_LIGHT),
            rounded_box(2.2, 0.75, fill_color=GREEN_DARK,
                        stroke_color=GREEN_MED, label="Aligned\nLLM", label_color=GREEN_LIGHT),
        )
        dpo_flow.arrange(RIGHT, buff=0.4)
        dpo_flow.move_to(ORIGIN + DOWN * 0.9)

        dpo_arrows = VGroup(*[
            Arrow(dpo_flow[i].get_right(), dpo_flow[i + 1].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.2)
            for i in range(2)
        ])

        dpo_lbl = label_text("DPO: 2 stages", color=GREEN_MED)
        dpo_lbl.next_to(dpo_flow, LEFT, buff=0.3)

        self.play(LaggedStart(*[FadeIn(b) for b in dpo_flow], lag_ratio=0.15),
                  FadeIn(dpo_lbl), run_time=0.9)
        self.play(LaggedStart(*[Create(a) for a in dpo_arrows], lag_ratio=0.15),
                  run_time=0.4)

        dpo_note = label_text(
            "DPO re-parameterises the RLHF objective — KL is still there in the maths",
            color=GREY_MED,
        )
        dpo_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(dpo_note), run_time=0.5)
        self.wait(2)
        self.fade_all(dpo_title, rlhf_flow, rlhf_arrows, rlhf_lbl,
                      dpo_flow, dpo_arrows, dpo_lbl, dpo_note)
