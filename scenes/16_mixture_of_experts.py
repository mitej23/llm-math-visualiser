"""
Scene 16 — Mixture of Experts (MoE)
Run: manim -pql 16_mixture_of_experts.py MixtureOfExpertsScene
"""

from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class MixtureOfExpertsScene(LLMScene):
    def construct(self):
        title = self.show_title("Mixture of Experts", "Specialization at Scale")
        self.wait(0.5)
        self.fade_all(title)

        # ── 1. Dense vs sparse ────────────────────────────────────────────────
        compare_title = body_text("Dense model: every token uses every parameter",
                                  color=WHITE)
        compare_title.to_edge(UP, buff=0.6)
        self.play(Write(compare_title), run_time=0.7)

        # Dense — all neurons lit
        dense_nodes = VGroup(*[
            Circle(radius=0.22, color=BLUE_MED, fill_color=BLUE_MED, fill_opacity=0.8)
            for _ in range(12)
        ])
        dense_nodes.arrange_in_grid(3, 4, buff=0.3)
        dense_nodes.shift(LEFT * 3.5)
        dense_lbl = label_text("Dense FFN\n(all active)", color=BLUE_MED)
        dense_lbl.next_to(dense_nodes, DOWN, buff=0.2)

        # MoE — only 2 of 8 experts lit
        expert_colors = [GREY_DARK] * 8
        expert_colors[2] = GREEN_MED
        expert_colors[5] = ORANGE_MED
        moe_nodes = VGroup(*[
            Circle(radius=0.22,
                   color=col if col != GREY_DARK else GREY_MED,
                   fill_color=col if col != GREY_DARK else GREY_DARK,
                   fill_opacity=0.8 if col != GREY_DARK else 0.3)
            for col in expert_colors
        ])
        moe_nodes.arrange_in_grid(2, 4, buff=0.3)
        moe_nodes.shift(RIGHT * 3.0)
        moe_lbl = label_text("MoE  (8 experts, 2 active)", color=GREEN_MED)
        moe_lbl.next_to(moe_nodes, DOWN, buff=0.2)

        active_lbl = label_text("Active", color=GREEN_MED)
        active_lbl.next_to(moe_nodes[2], UL, buff=0.08)
        active_lbl2 = label_text("Active", color=ORANGE_MED)
        active_lbl2.next_to(moe_nodes[5], UL, buff=0.08)

        vs = body_text("vs.", color=GREY_LIGHT).move_to(ORIGIN + LEFT * 0.5)

        self.play(FadeIn(dense_nodes), FadeIn(dense_lbl), run_time=0.7)
        self.play(Write(vs), FadeIn(moe_nodes), FadeIn(moe_lbl),
                  FadeIn(active_lbl), FadeIn(active_lbl2), run_time=0.9)
        self.wait(1)
        self.fade_all(compare_title, dense_nodes, dense_lbl,
                      vs, moe_nodes, moe_lbl, active_lbl, active_lbl2)

        # ── 2. Router mechanism ────────────────────────────────────────────────
        router_title = body_text("The Router decides which experts handle each token:",
                                  color=WHITE)
        router_title.to_edge(UP, buff=0.6)
        self.play(Write(router_title), run_time=0.7)

        token_box = rounded_box(1.4, 0.65,
                                fill_color=BLUE_DARK, stroke_color=BLUE_MED,
                                label='"chemistry"', label_color=BLUE_LIGHT)
        token_box.shift(LEFT * 5)

        router_box = rounded_box(1.6, 0.65,
                                 fill_color=str(PURPLE_MED) + "33", stroke_color=PURPLE_MED,
                                 label="Router\n(small linear)", label_color=PURPLE_MED)
        router_box.shift(LEFT * 2)

        n_experts = 8
        expert_boxes = VGroup()
        colors_e = [GREY_MED, GREY_MED, GREEN_MED, GREY_MED,
                    GREY_MED, ORANGE_MED, GREY_MED, GREY_MED]
        opacities = [0.2, 0.2, 0.9, 0.2, 0.2, 0.9, 0.2, 0.2]
        for i, (col, op) in enumerate(zip(colors_e, opacities)):
            e = rounded_box(0.85, 0.55,
                            fill_color=str(col) + "33" if op > 0.5 else GREY_DARK,
                            stroke_color=col,
                            label=f"E{i+1}", label_color=col)
            e.set_opacity(op)
            expert_boxes.add(e)

        expert_boxes.arrange(DOWN, buff=0.15)
        expert_boxes.shift(RIGHT * 1.5)

        combine_box = rounded_box(1.6, 0.65,
                                  fill_color=GREEN_DARK, stroke_color=GREEN_MED,
                                  label="Weighted\nCombine", label_color=GREEN_MED)
        combine_box.shift(RIGHT * 4.0)

        out_box = rounded_box(1.4, 0.65,
                              fill_color=BLUE_DARK, stroke_color=BLUE_MED,
                              label="Output", label_color=BLUE_MED)
        out_box.shift(RIGHT * 6.2)

        # Arrows
        a0 = Arrow(token_box.get_right(), router_box.get_left(),
                   color=GREY_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.18)
        active_experts = [expert_boxes[2], expert_boxes[5]]
        a1 = Arrow(router_box.get_right(),
                   active_experts[0].get_left() + LEFT * 0.1,
                   color=GREEN_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        a2 = Arrow(router_box.get_right(),
                   active_experts[1].get_left() + LEFT * 0.1,
                   color=ORANGE_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        a3 = Arrow(active_experts[0].get_right(),
                   combine_box.get_left() + UP * 0.15,
                   color=GREEN_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        a4 = Arrow(active_experts[1].get_right(),
                   combine_box.get_left() + DOWN * 0.15,
                   color=ORANGE_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        a5 = Arrow(combine_box.get_right(), out_box.get_left(),
                   color=GREY_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.18)

        diagram = VGroup(token_box, a0, router_box, expert_boxes,
                         a1, a2, a3, a4, combine_box, a5, out_box)
        diagram.scale_to_fit_width(13.5)
        diagram.move_to(ORIGIN + DOWN * 0.2)

        self.play(LaggedStart(*[FadeIn(m) for m in [
            token_box, a0, router_box, expert_boxes,
            a1, a2, a3, a4, combine_box, a5, out_box
        ]], lag_ratio=0.08), run_time=2.2)
        self.wait(1)
        self.fade_all(router_title, diagram)

        # ── 3. Real-world models ───────────────────────────────────────────────
        models_title = body_text("MoE in the wild:", color=WHITE)
        models_title.to_edge(UP, buff=0.6)
        self.play(Write(models_title), run_time=0.5)

        model_data = [
            ("Mixtral 8x7B",  BLUE_MED,    "8 experts, 2 active · Total 47B, Active 13B"),
            ("Grok-1",        ORANGE_MED,  "64 experts, 8 active · 314B total, 86B active"),
            ("DeepSeek-V3",   GREEN_MED,   "256 experts, 8 active · frontier quality"),
            ("GPT-4 (est.)",  PURPLE_MED,  "~8 experts · ~1.8T total parameters"),
        ]
        rows = VGroup()
        for name, col, desc in model_data:
            n = body_text(name, color=col)
            d = label_text(desc, color=WHITE)
            d.next_to(n, RIGHT, buff=0.4)
            rows.add(VGroup(n, d))

        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        rows.move_to(ORIGIN + DOWN * 0.3)
        box = SurroundingRectangle(rows, color=GREY_MED, buff=0.35, corner_radius=0.15)

        self.play(Create(box), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.2),
                  run_time=1.3)

        summary = label_text(
            "Key insight: more total knowledge, same compute per token!",
            color=YELLOW_MED,
        )
        summary.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(summary), run_time=0.7)
        self.wait(2)
        self.fade_all(models_title, box, rows, summary)

        # ── 4. The capacity problem ────────────────────────────────────────────
        cap_title = body_text("10x more params — but NOT 10x more compute", color=WHITE)
        cap_title.to_edge(UP, buff=0.6)
        self.play(Write(cap_title), run_time=0.7)

        # Two columns: dense scaling vs MoE scaling
        dense_col_lbl = body_text("Dense model", color=BLUE_MED)
        dense_col_lbl.move_to(LEFT * 3.5 + UP * 1.8)

        moe_col_lbl = body_text("MoE model", color=GREEN_MED)
        moe_col_lbl.move_to(RIGHT * 2.5 + UP * 1.8)

        dense_rows = VGroup()
        dense_data = [
            ("7B params",  "7B compute/token",  BLUE_MED),
            ("70B params", "70B compute/token", ORANGE_MED),
            ("700B params","700B compute/token", RED_MED),
        ]
        for params, compute, col in dense_data:
            p = label_text(params, color=col)
            arr = label_text("->", color=GREY_MED)
            c = label_text(compute, color=col)
            arr.next_to(p, RIGHT, buff=0.2)
            c.next_to(arr, RIGHT, buff=0.2)
            dense_rows.add(VGroup(p, arr, c))
        dense_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        dense_rows.next_to(dense_col_lbl, DOWN, buff=0.3)

        moe_row_data = [
            ("8 x 7B = 56B total", "Only 2 active -> 14B compute/token", GREEN_MED),
        ]
        moe_rows = VGroup()
        for total, compute, col in moe_row_data:
            t = label_text(total, color=col)
            arr = label_text("->", color=GREY_MED)
            c = label_text(compute, color=col)
            arr.next_to(t, RIGHT, buff=0.2)
            c.next_to(arr, RIGHT, buff=0.2)
            moe_rows.add(VGroup(t, arr, c))
        moe_rows.next_to(moe_col_lbl, DOWN, buff=0.3)

        win_label = label_text("4x more knowledge, only 2x compute!", color=YELLOW_MED)
        win_label.next_to(moe_rows, DOWN, buff=0.4)

        divider = Line(UP * 2.5, DOWN * 1.5, color=GREY_MED, stroke_width=1)
        divider.move_to(ORIGIN + LEFT * 0.3)

        self.play(FadeIn(dense_col_lbl), FadeIn(moe_col_lbl), Create(divider), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(r) for r in dense_rows], lag_ratio=0.25), run_time=1.0)
        self.play(FadeIn(moe_rows), run_time=0.6)
        self.play(FadeIn(win_label), run_time=0.5)
        self.wait(1.5)
        self.fade_all(cap_title, dense_col_lbl, moe_col_lbl, divider,
                      dense_rows, moe_rows, win_label)

        # ── 5. Expert specialisation over training ─────────────────────────────
        spec_title = body_text("Experts develop specialisations during training:",
                               color=WHITE)
        spec_title.to_edge(UP, buff=0.6)
        self.play(Write(spec_title), run_time=0.7)

        expert_spec = [
            ("Expert 1", BLUE_MED,   "Code & syntax"),
            ("Expert 2", GREEN_MED,  "Reasoning & logic"),
            ("Expert 3", ORANGE_MED, "Languages & translation"),
            ("Expert 4", PURPLE_MED, "Math & numbers"),
        ]

        spec_boxes = VGroup()
        for name, col, spec in expert_spec:
            outer = rounded_box(3.2, 0.9,
                                fill_color=str(col) + "22",
                                stroke_color=col,
                                label="", label_color=col)
            name_lbl = body_text(name, color=col)
            name_lbl.move_to(outer.get_center() + LEFT * 0.6)
            spec_lbl = label_text(spec, color=WHITE)
            spec_lbl.next_to(name_lbl, RIGHT, buff=0.3)
            spec_boxes.add(VGroup(outer, name_lbl, spec_lbl))

        spec_boxes.arrange(DOWN, buff=0.25)
        spec_boxes.move_to(ORIGIN + DOWN * 0.3)

        emerge_note = label_text(
            "Specialisation is emergent — nobody programs it in!",
            color=YELLOW_MED,
        )
        emerge_note.to_edge(DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(b) for b in spec_boxes], lag_ratio=0.2),
                  run_time=1.5)
        self.play(FadeIn(emerge_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(spec_title, spec_boxes, emerge_note)

        # ── 6. Load balancing problem ──────────────────────────────────────────
        lb_title = body_text("Load balancing problem: all tokens rush to Expert 1",
                             color=WHITE)
        lb_title.to_edge(UP, buff=0.6)
        self.play(Write(lb_title), run_time=0.7)

        # 8 expert boxes arranged in a row
        lb_expert_colors = [RED_MED] + [GREY_MED] * 7
        lb_expert_labels = ["E1\n(OVERLOADED)"] + [f"E{i+2}\n(idle)" for i in range(7)]
        lb_boxes = VGroup()
        for i, (col, lbl) in enumerate(zip(lb_expert_colors, lb_expert_labels)):
            b = rounded_box(1.35, 1.0,
                            fill_color=str(col) + "33",
                            stroke_color=col,
                            label=lbl, label_color=col)
            lb_boxes.add(b)

        lb_boxes.arrange(RIGHT, buff=0.18)
        lb_boxes.scale_to_fit_width(13.5)
        lb_boxes.move_to(ORIGIN + UP * 0.2)

        # Token arrows all pointing to E1
        token_lbls = ["tok1", "tok2", "tok3", "tok4", "tok5"]
        token_arrows = VGroup()
        for i, t in enumerate(token_lbls):
            start_x = -2.5 + i * 1.1
            start = np.array([start_x, -2.2, 0])
            end = lb_boxes[0].get_bottom() + DOWN * 0.05
            arr = Arrow(start, end, color=RED_MED, stroke_width=1.5, buff=0.05,
                        max_tip_length_to_length_ratio=0.18)
            tok_lbl = label_text(t, color=RED_MED)
            tok_lbl.move_to(start + DOWN * 0.25)
            token_arrows.add(VGroup(arr, tok_lbl))

        fix_note = label_text(
            "Fix: auxiliary loss penalises uneven routing -> forces balanced distribution",
            color=GREEN_MED,
        )
        fix_note.to_edge(DOWN, buff=0.4)

        self.play(FadeIn(lb_boxes), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(ta) for ta in token_arrows], lag_ratio=0.1),
                  run_time=1.0)
        self.play(FadeIn(fix_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(lb_title, lb_boxes, token_arrows, fix_note)

        # ── 7. Token dropping ──────────────────────────────────────────────────
        drop_title = body_text("Token dropping: expert is full — overflow tokens skip it",
                               color=WHITE)
        drop_title.to_edge(UP, buff=0.6)
        self.play(Write(drop_title), run_time=0.7)

        expert_full = rounded_box(2.8, 2.0,
                                  fill_color=str(RED_MED) + "22",
                                  stroke_color=RED_MED,
                                  label="Expert 3\n(FULL: 4/4 slots used)",
                                  label_color=RED_MED)
        expert_full.move_to(ORIGIN + RIGHT * 1.5)

        # Already-queued tokens inside
        queued = VGroup()
        for i in range(4):
            tb = rounded_box(0.7, 0.4, fill_color=BLUE_DARK,
                             stroke_color=BLUE_MED, label=f"t{i+1}", label_color=BLUE_MED)
            queued.add(tb)
        queued.arrange(DOWN, buff=0.15)
        queued.move_to(expert_full.get_center())

        # New arriving token
        new_tok = rounded_box(0.8, 0.45, fill_color=str(YELLOW_MED) + "33",
                              stroke_color=YELLOW_MED, label="t5 NEW", label_color=YELLOW_MED)
        new_tok.move_to(LEFT * 3.5)
        arrow_to_full = Arrow(new_tok.get_right(), expert_full.get_left(),
                              color=YELLOW_MED, stroke_width=1.5, buff=0.05,
                              max_tip_length_to_length_ratio=0.2)

        drop_x = Label = label_text("DROPPED", color=RED_MED)
        drop_x.move_to(LEFT * 3.5 + DOWN * 1.5)
        drop_arr = Arrow(new_tok.get_bottom(), drop_x.get_top(),
                         color=RED_MED, stroke_width=1.5, buff=0.05,
                         max_tip_length_to_length_ratio=0.25)

        tradeoff = label_text(
            "Trade-off: higher capacity factor -> less dropping -> more memory",
            color=GREY_LIGHT,
        )
        tradeoff.to_edge(DOWN, buff=0.4)

        self.play(FadeIn(expert_full), FadeIn(queued), run_time=0.6)
        self.play(FadeIn(new_tok), Create(arrow_to_full), run_time=0.6)
        self.play(FadeOut(arrow_to_full), Create(drop_arr), FadeIn(drop_x), run_time=0.6)
        self.play(FadeIn(tradeoff), run_time=0.5)
        self.wait(1.5)
        self.fade_all(drop_title, expert_full, queued, new_tok, drop_arr, drop_x, tradeoff)

        # ── 8. MoE in the FFN layer ────────────────────────────────────────────
        ffn_title = body_text("MoE replaces the FFN in each transformer block:",
                              color=WHITE)
        ffn_title.to_edge(UP, buff=0.6)
        self.play(Write(ffn_title), run_time=0.7)

        # Left: regular transformer block
        reg_lbl = body_text("Regular block", color=BLUE_MED)
        reg_lbl.move_to(LEFT * 3.8 + UP * 1.6)

        reg_attn = rounded_box(1.8, 0.65, fill_color=str(BLUE_MED) + "22",
                               stroke_color=BLUE_MED, label="Attention", label_color=BLUE_MED)
        reg_ffn = rounded_box(1.8, 0.65, fill_color=str(BLUE_MED) + "22",
                              stroke_color=BLUE_MED, label="Single FFN", label_color=BLUE_MED)
        reg_attn.move_to(LEFT * 3.8 + UP * 0.4)
        reg_ffn.move_to(LEFT * 3.8 + DOWN * 0.7)
        reg_arr = Arrow(reg_attn.get_bottom(), reg_ffn.get_top(),
                        color=BLUE_MED, stroke_width=1.5, buff=0.05,
                        max_tip_length_to_length_ratio=0.25)

        # Right: MoE block
        moe_lbl2 = body_text("MoE block", color=GREEN_MED)
        moe_lbl2.move_to(RIGHT * 2.5 + UP * 1.6)

        moe_attn = rounded_box(1.8, 0.65, fill_color=str(GREEN_MED) + "22",
                               stroke_color=GREEN_MED, label="Attention", label_color=GREEN_MED)
        moe_attn.move_to(RIGHT * 2.5 + UP * 0.4)

        moe_router_b = rounded_box(1.4, 0.55, fill_color=str(PURPLE_MED) + "22",
                                   stroke_color=PURPLE_MED, label="Router", label_color=PURPLE_MED)
        moe_router_b.move_to(RIGHT * 2.5 + DOWN * 0.5)

        moe_experts_mini = VGroup()
        for i in range(4):
            eb = rounded_box(1.0, 0.45, fill_color=str(ORANGE_MED) + "22",
                             stroke_color=ORANGE_MED, label=f"FFN {i+1}", label_color=ORANGE_MED)
            moe_experts_mini.add(eb)
        moe_experts_mini.arrange(RIGHT, buff=0.15)
        moe_experts_mini.next_to(moe_router_b, DOWN, buff=0.3)

        arr_attn_router = Arrow(moe_attn.get_bottom(), moe_router_b.get_top(),
                                color=GREEN_MED, stroke_width=1.5, buff=0.05,
                                max_tip_length_to_length_ratio=0.25)
        arr_router_exp = Arrow(moe_router_b.get_bottom(), moe_experts_mini.get_top(),
                               color=PURPLE_MED, stroke_width=1.5, buff=0.05,
                               max_tip_length_to_length_ratio=0.25)

        same_note = label_text("Attention is unchanged — only FFN is replaced with experts",
                               color=YELLOW_MED)
        same_note.to_edge(DOWN, buff=0.4)

        divider2 = Line(UP * 2.2, DOWN * 2.2, color=GREY_MED, stroke_width=1)

        self.play(FadeIn(reg_lbl), FadeIn(moe_lbl2), Create(divider2), run_time=0.5)
        self.play(FadeIn(reg_attn), FadeIn(moe_attn), run_time=0.5)
        self.play(Create(reg_arr), FadeIn(reg_ffn), run_time=0.5)
        self.play(Create(arr_attn_router), FadeIn(moe_router_b), run_time=0.5)
        self.play(Create(arr_router_exp),
                  LaggedStart(*[FadeIn(e) for e in moe_experts_mini], lag_ratio=0.1),
                  run_time=0.8)
        self.play(FadeIn(same_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(ffn_title, reg_lbl, moe_lbl2, divider2,
                      reg_attn, reg_ffn, reg_arr,
                      moe_attn, moe_router_b, arr_attn_router,
                      moe_experts_mini, arr_router_exp, same_note)

        # ── 9. Training challenges ─────────────────────────────────────────────
        train_title = body_text("Training MoE is harder than training dense models:",
                                color=WHITE)
        train_title.to_edge(UP, buff=0.6)
        self.play(Write(train_title), run_time=0.7)

        challenges = [
            ("Experts don't share gradients",
             "Each expert trained on ~1/N of data", RED_MED),
            ("Routing instability",
             "Router can oscillate — needs careful warmup", ORANGE_MED),
            ("More data needed",
             "~2-3x more tokens to converge vs dense", YELLOW_MED),
            ("Harder to fine-tune",
             "LoRA works better than full fine-tuning for MoE", BLUE_MED),
        ]

        ch_group = VGroup()
        for challenge, detail, col in challenges:
            ch_box = rounded_box(11.5, 0.85,
                                 fill_color=str(col) + "22",
                                 stroke_color=col, label="", label_color=col)
            ch_lbl = body_text(challenge, color=col)
            det_lbl = label_text(detail, color=WHITE)
            ch_lbl.move_to(ch_box.get_center() + LEFT * 2.5)
            det_lbl.move_to(ch_box.get_center() + RIGHT * 1.5)
            ch_group.add(VGroup(ch_box, ch_lbl, det_lbl))

        ch_group.arrange(DOWN, buff=0.2)
        ch_group.move_to(ORIGIN + DOWN * 0.2)

        self.play(LaggedStart(*[FadeIn(c) for c in ch_group], lag_ratio=0.2),
                  run_time=1.5)
        self.wait(1.5)
        self.fade_all(train_title, ch_group)
