"""
Scene 29 — CISPO: Clipped IS Policy Optimization
Run: manim -pql 29_cispo.py CISPOScene
"""
from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class CISPOScene(LLMScene):
    def construct(self):

        # ── 1. Title ──────────────────────────────────────────────────────────
        title = self.show_title("CISPO", "Clipped IS Policy Optimization")
        self.wait(0.8)
        tagline = label_text(
            "Reuse old samples  |  Block gradient through IS weights  |  ~2x throughput",
            color=GREY_LIGHT,
        )
        tagline.next_to(title, DOWN, buff=0.45)
        self.play(FadeIn(tagline), run_time=0.6)
        self.wait(1.2)
        self.fade_all(title, tagline)

        # ── 2. The off-policy problem ─────────────────────────────────────────
        offpol_title = body_text("Problem: fresh samples after every update is expensive",
                                 color=WHITE)
        offpol_title.to_edge(UP, buff=0.6)
        self.play(Write(offpol_title), run_time=0.7)

        # Three-step cycle: generate → score → update → repeat
        steps = [
            ("Generate\nresponses",  BLUE_MED),
            ("Score with\nreward model", ORANGE_MED),
            ("Gradient\nupdate", GREEN_MED),
            ("Discard\nsamples", RED_MED),
        ]
        step_boxes = VGroup()
        for lbl, col in steps:
            b = rounded_box(2.6, 1.0,
                            fill_color=col + "22",
                            stroke_color=col,
                            label=lbl, label_color=WHITE)
            step_boxes.add(b)
        step_boxes.arrange(RIGHT, buff=0.55)
        step_boxes.move_to(ORIGIN + UP * 0.4)

        cycle_arrows = VGroup(*[
            Arrow(step_boxes[i].get_right(), step_boxes[i + 1].get_left(),
                  buff=0.05, color=GREY_MED, stroke_width=2,
                  max_tip_length_to_length_ratio=0.25)
            for i in range(3)
        ])

        self.play(LaggedStart(*[FadeIn(b) for b in step_boxes], lag_ratio=0.2),
                  run_time=0.9)
        self.play(LaggedStart(*[Create(a) for a in cycle_arrows], lag_ratio=0.2),
                  run_time=0.6)

        waste_note = label_text(
            "GPU idle during generation  |  expensive forward passes wasted after one update",
            color=RED_MED,
        )
        waste_note.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(waste_note), run_time=0.5)
        self.wait(1.2)

        question = body_text("Can we reuse old samples instead of always generating fresh ones?",
                             color=YELLOW_MED)
        question.next_to(offpol_title, DOWN, buff=0.35)
        self.play(Write(question), run_time=0.8)
        self.wait(1.5)
        self.fade_all(offpol_title, question, step_boxes, cycle_arrows, waste_note)

        # ── 3. Importance sampling: the correction factor ─────────────────────
        is_title = body_text("Importance Sampling: correcting for stale data", color=WHITE)
        is_title.to_edge(UP, buff=0.6)
        self.play(Write(is_title), run_time=0.7)

        # Receipt analogy boxes
        old_box = rounded_box(3.2, 1.2,
                              fill_color=BLUE_MED + "22", stroke_color=BLUE_MED,
                              label="Old policy  (data collector)\nrecorded log_prob_old",
                              label_color=BLUE_LIGHT)
        old_box.shift(LEFT * 3.5 + UP * 0.6)

        new_box = rounded_box(3.2, 1.2,
                              fill_color=GREEN_MED + "22", stroke_color=GREEN_MED,
                              label="Current policy  (learner)\ncomputes log_prob_new",
                              label_color=WHITE)
        new_box.shift(RIGHT * 3.5 + UP * 0.6)

        ratio_box = rounded_box(3.8, 0.9,
                                fill_color=ORANGE_MED + "22", stroke_color=ORANGE_MED,
                                label="IS ratio  =  prob_new / prob_old",
                                label_color=WHITE)
        ratio_box.move_to(ORIGIN + DOWN * 0.7)

        a1 = Arrow(old_box.get_right(), ratio_box.get_left(), buff=0.05,
                   color=GREY_MED, stroke_width=2, max_tip_length_to_length_ratio=0.2)
        a2 = Arrow(new_box.get_left(), ratio_box.get_right(), buff=0.05,
                   color=GREY_MED, stroke_width=2, max_tip_length_to_length_ratio=0.2)

        ratio_meaning = label_text(
            "ratio ~ 1  →  data still relevant     ratio >> 1 or << 1  →  data is stale",
            color=GREY_LIGHT,
        )
        ratio_meaning.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(old_box), FadeIn(new_box), run_time=0.7)
        self.play(Create(a1), Create(a2), run_time=0.5)
        self.play(FadeIn(ratio_box), run_time=0.5)
        self.play(FadeIn(ratio_meaning), run_time=0.5)
        self.wait(1.5)
        self.fade_all(is_title, old_box, new_box, a1, a2, ratio_box, ratio_meaning)

        # ── 4. Variance explosion: when IS ratio is large ─────────────────────
        var_title = body_text("The problem: large IS ratios explode gradient variance",
                              color=RED_MED)
        var_title.to_edge(UP, buff=0.6)
        self.play(Write(var_title), run_time=0.7)

        # Show three ratio values and their effect on the gradient weight
        ratio_examples = [
            ("ratio = 0.95", GREEN_MED,  "gradient scaled by 0.95  —  fine"),
            ("ratio = 3.0",  ORANGE_MED, "gradient scaled by 3.0   —  3x amplification"),
            ("ratio = 15.0", RED_MED,    "gradient scaled by 15.0  —  dominates batch"),
        ]
        ex_rows = VGroup()
        for ratio_str, col, effect_str in ratio_examples:
            r_lbl = body_text(ratio_str, color=col)
            arrow = Arrow(ORIGIN, RIGHT * 1.4, buff=0.0, color=GREY_MED,
                          stroke_width=1.5, max_tip_length_to_length_ratio=0.2)
            e_lbl = label_text(effect_str, color=col)
            row = VGroup(r_lbl, arrow, e_lbl)
            row.arrange(RIGHT, buff=0.3)
            ex_rows.add(row)

        ex_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        ex_rows.move_to(ORIGIN + UP * 0.2)
        surround = SurroundingRectangle(ex_rows, color=GREY_MED, buff=0.35,
                                        corner_radius=0.12)

        var_note = label_text(
            "A single sample with ratio = 15 can dominate an entire batch  —  training becomes unstable",
            color=GREY_LIGHT,
        )
        var_note.to_edge(DOWN, buff=0.5)

        self.play(Create(surround), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r) for r in ex_rows], lag_ratio=0.3),
                  run_time=1.0)
        self.play(FadeIn(var_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(var_title, ex_rows, surround, var_note)

        # ── 5. Stop-gradient visualization ────────────────────────────────────
        sg_title = body_text("CISPO: compute IS ratio but block its gradient", color=WHITE)
        sg_title.to_edge(UP, buff=0.6)
        self.play(Write(sg_title), run_time=0.7)

        # Computation graph nodes
        old_log = rounded_box(2.4, 0.75, fill_color=BLUE_MED + "22",
                              stroke_color=BLUE_MED, label="log_prob_old",
                              label_color=BLUE_LIGHT)
        new_log = rounded_box(2.4, 0.75, fill_color=GREEN_MED + "22",
                              stroke_color=GREEN_MED, label="log_prob_new",
                              label_color=GREEN_LIGHT)
        ratio_node = rounded_box(2.6, 0.75, fill_color=ORANGE_MED + "22",
                                 stroke_color=ORANGE_MED, label="IS ratio",
                                 label_color=WHITE)
        sg_node = rounded_box(3.0, 0.75, fill_color=RED_MED + "22",
                              stroke_color=RED_MED, label="stop_gradient( ratio )",
                              label_color=WHITE)
        loss_node = rounded_box(2.4, 0.75, fill_color=PURPLE_MED + "22",
                                stroke_color=PURPLE_MED, label="clipped loss",
                                label_color=WHITE)

        old_log.move_to(LEFT * 5.2 + UP * 1.2)
        new_log.move_to(LEFT * 2.5 + UP * 1.2)
        ratio_node.move_to(ORIGIN + UP * 1.2)
        sg_node.move_to(RIGHT * 3.2 + UP * 1.2)
        loss_node.move_to(RIGHT * 3.2 + DOWN * 0.5)

        # Forward arrows
        f1 = Arrow(old_log.get_right(), ratio_node.get_left(), buff=0.05,
                   color=GREY_MED, stroke_width=2, max_tip_length_to_length_ratio=0.2)
        f2 = Arrow(new_log.get_right(), ratio_node.get_left(), buff=0.05,
                   color=GREY_MED, stroke_width=2, max_tip_length_to_length_ratio=0.2)
        f3 = Arrow(ratio_node.get_right(), sg_node.get_left(), buff=0.05,
                   color=ORANGE_MED, stroke_width=2, max_tip_length_to_length_ratio=0.2)
        f4 = Arrow(sg_node.get_bottom(), loss_node.get_top(), buff=0.05,
                   color=PURPLE_MED, stroke_width=2, max_tip_length_to_length_ratio=0.2)

        # Blocked backward arrow (dashed red X)
        blocked_arrow = DashedLine(
            sg_node.get_left(), ratio_node.get_right(),
            color=RED_MED, dash_length=0.12, stroke_width=2.5,
        )
        block_x = Text("X", color=RED_MED, font_size=28, weight=BOLD)
        block_x.move_to(blocked_arrow.get_center() + UP * 0.35)
        blocked_label = label_text("gradient blocked here", color=RED_MED)
        blocked_label.next_to(blocked_arrow, DOWN, buff=0.15)

        grad_note = label_text(
            "Forward pass: ratio is computed normally     Backward pass: ratio is treated as a constant",
            color=GREY_LIGHT,
        )
        grad_note.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(old_log), FadeIn(new_log), run_time=0.5)
        self.play(Create(f1), Create(f2), run_time=0.4)
        self.play(FadeIn(ratio_node), run_time=0.4)
        self.play(Create(f3), FadeIn(sg_node), run_time=0.5)
        self.play(Create(f4), FadeIn(loss_node), run_time=0.4)
        self.play(Create(blocked_arrow), FadeIn(block_x), FadeIn(blocked_label),
                  run_time=0.6)
        self.play(FadeIn(grad_note), run_time=0.5)
        self.wait(1.8)
        self.fade_all(sg_title, old_log, new_log, ratio_node, sg_node, loss_node,
                      f1, f2, f3, f4, blocked_arrow, block_x, blocked_label, grad_note)

        # ── 6. The clipping part ──────────────────────────────────────────────
        clip_title = body_text("Clipping: still prevent huge policy updates", color=WHITE)
        clip_title.to_edge(UP, buff=0.6)
        self.play(Write(clip_title), run_time=0.7)

        # Number line showing clip window
        line = Line(LEFT * 5.5, RIGHT * 5.5, color=GREY_MED, stroke_width=2)
        line.move_to(ORIGIN + UP * 0.3)

        # Markers at 0.8, 1.0, 1.2
        tick_vals = [(-3.5, "0.8"), (0.0, "1.0"), (3.5, "1.2")]
        ticks = VGroup()
        for x, lbl in tick_vals:
            tick = Line(UP * 0.18, DOWN * 0.18, color=WHITE, stroke_width=2)
            tick.move_to(line.get_center() + RIGHT * x)
            t_lbl = label_text(lbl, color=WHITE)
            t_lbl.next_to(tick, DOWN, buff=0.18)
            ticks.add(VGroup(tick, t_lbl))

        # Shaded clip window
        clip_zone = Rectangle(width=7.0, height=0.55,
                              fill_color=GREEN_MED + "33", fill_opacity=1.0,
                              stroke_width=0)
        clip_zone.move_to(line.get_center())

        clip_lbl = label_text("clip window  [0.8, 1.2]  — data in here passes through",
                              color=GREEN_MED)
        clip_lbl.next_to(clip_zone, UP, buff=0.35)

        out_left = label_text("ratio too small  —  excluded", color=RED_MED)
        out_left.move_to(LEFT * 5.0 + DOWN * 0.85)
        out_right = label_text("ratio too large  —  excluded", color=RED_MED)
        out_right.move_to(RIGHT * 5.0 + DOWN * 0.85)

        clip_note = label_text(
            "Clipping acts as a gate: only fresh-enough data shapes the gradient update",
            color=GREY_LIGHT,
        )
        clip_note.to_edge(DOWN, buff=0.5)

        self.play(Create(line), run_time=0.4)
        self.play(FadeIn(clip_zone), run_time=0.3)
        self.play(LaggedStart(*[FadeIn(t) for t in ticks], lag_ratio=0.2), run_time=0.5)
        self.play(FadeIn(clip_lbl), run_time=0.4)
        self.play(FadeIn(out_left), FadeIn(out_right), run_time=0.5)
        self.play(FadeIn(clip_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(clip_title, line, clip_zone, ticks, clip_lbl,
                      out_left, out_right, clip_note)

        # ── 7. How this enables 2x throughput: async data collection ──────────
        async_title = body_text("Result: async pipeline gives ~2x training throughput",
                                color=GREEN_MED)
        async_title.to_edge(UP, buff=0.6)
        self.play(Write(async_title), run_time=0.7)

        # Synchronous timeline (PPO)
        sync_lbl = label_text("PPO  (synchronous):", color=BLUE_MED)
        sync_lbl.move_to(LEFT * 5.8 + UP * 1.5)

        gen_bar = Rectangle(width=3.2, height=0.55,
                            fill_color=BLUE_MED + "55", fill_opacity=1.0,
                            stroke_color=BLUE_MED, stroke_width=1.5)
        gen_bar.move_to(LEFT * 1.8 + UP * 1.5)
        gen_lbl = label_text("Generate", color=BLUE_MED)
        gen_lbl.move_to(gen_bar)

        train_bar_ppo = Rectangle(width=3.2, height=0.55,
                                  fill_color=PURPLE_MED + "55", fill_opacity=1.0,
                                  stroke_color=PURPLE_MED, stroke_width=1.5)
        train_bar_ppo.move_to(RIGHT * 1.8 + UP * 1.5)
        train_lbl_ppo = label_text("Train (PPO)", color=PURPLE_MED)
        train_lbl_ppo.move_to(train_bar_ppo)

        gen_bar2 = gen_bar.copy().shift(DOWN * 0.0 + RIGHT * 6.4)
        gen_lbl2 = label_text("Generate", color=BLUE_MED)
        gen_lbl2.move_to(gen_bar2)

        idle_note = label_text("idle", color=RED_MED)
        idle_note.move_to(gen_bar2.get_center())

        # Asynchronous timeline (CISPO)
        async_lbl = label_text("CISPO  (asynchronous):", color=GREEN_MED)
        async_lbl.move_to(LEFT * 5.8 + DOWN * 0.2)

        gen_async = Rectangle(width=6.5, height=0.55,
                              fill_color=BLUE_MED + "44", fill_opacity=1.0,
                              stroke_color=BLUE_MED, stroke_width=1.5)
        gen_async.move_to(ORIGIN + RIGHT * 0.1 + DOWN * 0.2)
        gen_async_lbl = label_text("Generate continuously (worker)", color=BLUE_MED)
        gen_async_lbl.move_to(gen_async)

        train_async = Rectangle(width=6.5, height=0.55,
                                fill_color=GREEN_MED + "44", fill_opacity=1.0,
                                stroke_color=GREEN_MED, stroke_width=1.5)
        train_async.move_to(ORIGIN + RIGHT * 0.1 + DOWN * 1.0)
        train_async_lbl = label_text("Train continuously (CISPO + clip)", color=GREEN_MED)
        train_async_lbl.move_to(train_async)

        overlap_note = label_text(
            "Generation and training overlap in time  —  GPU stays busy in both phases",
            color=GREY_LIGHT,
        )
        overlap_note.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(sync_lbl), FadeIn(gen_bar), FadeIn(gen_lbl), run_time=0.5)
        self.play(FadeIn(train_bar_ppo), FadeIn(train_lbl_ppo), run_time=0.4)
        self.play(FadeIn(gen_bar2), FadeIn(idle_note), run_time=0.4)

        self.play(FadeIn(async_lbl), FadeIn(gen_async), FadeIn(gen_async_lbl),
                  run_time=0.5)
        self.play(FadeIn(train_async), FadeIn(train_async_lbl), run_time=0.4)
        self.play(FadeIn(overlap_note), run_time=0.5)
        self.wait(1.8)
        self.fade_all(async_title, sync_lbl, gen_bar, gen_lbl, train_bar_ppo,
                      train_lbl_ppo, gen_bar2, gen_lbl2, idle_note,
                      async_lbl, gen_async, gen_async_lbl, train_async,
                      train_async_lbl, overlap_note)

        # ── 8. CISPO vs PPO computation graph comparison ──────────────────────
        comp_title = body_text("PPO vs CISPO: what the backward pass touches", color=WHITE)
        comp_title.to_edge(UP, buff=0.6)
        self.play(Write(comp_title), run_time=0.7)

        # PPO column
        ppo_header = body_text("PPO", color=BLUE_MED)
        ppo_header.move_to(LEFT * 3.5 + UP * 2.2)

        ppo_items = [
            ("Old policy forward  (grad tracking)", BLUE_MED),
            ("IS ratio  (in gradient)", ORANGE_MED),
            ("Clip  (IS x advantage)", ORANGE_MED),
            ("Current policy backward", PURPLE_MED),
            ("IS ratio backward  (extra pass)", RED_MED),
        ]
        ppo_col = VGroup()
        for txt, col in ppo_items:
            row = label_text(txt, color=col)
            ppo_col.add(row)
        ppo_col.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        ppo_col.next_to(ppo_header, DOWN, buff=0.3)
        ppo_col.shift(LEFT * 0.2)

        ppo_border = SurroundingRectangle(VGroup(ppo_header, ppo_col),
                                          color=BLUE_MED, buff=0.25, corner_radius=0.12)

        # CISPO column
        cispo_header = body_text("CISPO", color=GREEN_MED)
        cispo_header.move_to(RIGHT * 3.5 + UP * 2.2)

        cispo_items = [
            ("Old policy forward  (no grad)", BLUE_MED),
            ("IS ratio  (stop-gradient)", GREEN_MED),
            ("Clip gate  (include / exclude)", GREEN_MED),
            ("Current policy backward", PURPLE_MED),
            ("IS ratio backward  — skipped", GREY_MED),
        ]
        cispo_col = VGroup()
        for txt, col in cispo_items:
            row = label_text(txt, color=col)
            cispo_col.add(row)
        cispo_col.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        cispo_col.next_to(cispo_header, DOWN, buff=0.3)
        cispo_col.shift(LEFT * 0.2)

        cispo_border = SurroundingRectangle(VGroup(cispo_header, cispo_col),
                                            color=GREEN_MED, buff=0.25, corner_radius=0.12)

        saved_lbl = label_text("One full backward pass eliminated per step",
                               color=GREEN_MED)
        saved_lbl.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(ppo_header), FadeIn(cispo_header), run_time=0.4)
        self.play(Create(ppo_border), Create(cispo_border), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(r) for r in ppo_col], lag_ratio=0.15),
                  LaggedStart(*[FadeIn(r) for r in cispo_col], lag_ratio=0.15),
                  run_time=1.0)
        self.play(FadeIn(saved_lbl), run_time=0.5)
        self.wait(1.8)
        self.fade_all(comp_title, ppo_header, ppo_col, ppo_border,
                      cispo_header, cispo_col, cispo_border, saved_lbl)

        # ── 9. Memory savings analysis ────────────────────────────────────────
        mem_title = body_text("Memory: inference-mode old policy = much less VRAM",
                              color=WHITE)
        mem_title.to_edge(UP, buff=0.6)
        self.play(Write(mem_title), run_time=0.7)

        mem_items = [
            ("PPO: old policy forward", "stores full activation graph for backward",
             BLUE_MED, RED_MED),
            ("CISPO: old policy forward", "inference mode — activations not retained",
             GREEN_MED, GREEN_MED),
        ]

        mem_rows = VGroup()
        for method, detail, col1, col2 in mem_items:
            m_lbl = body_text(method, color=col1)
            d_lbl = label_text(detail, color=col2)
            d_lbl.next_to(m_lbl, RIGHT, buff=0.4)
            mem_rows.add(VGroup(m_lbl, d_lbl))

        mem_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        mem_rows.move_to(ORIGIN + UP * 0.4)

        # Bar chart showing relative memory use
        bar_y = -1.5
        ppo_mem = Rectangle(width=5.5, height=0.5,
                            fill_color=RED_MED + "55", fill_opacity=1.0,
                            stroke_color=RED_MED, stroke_width=1.5)
        ppo_mem.move_to(LEFT * 0.25 + UP * bar_y)
        ppo_mem_lbl = label_text("PPO activation memory (training mode)", color=RED_MED)
        ppo_mem_lbl.next_to(ppo_mem, LEFT, buff=0.15)

        cispo_mem = Rectangle(width=2.6, height=0.5,
                              fill_color=GREEN_MED + "55", fill_opacity=1.0,
                              stroke_color=GREEN_MED, stroke_width=1.5)
        cispo_mem.align_to(ppo_mem, LEFT)
        cispo_mem.shift(DOWN * 0.7)
        cispo_mem_lbl = label_text("CISPO activation memory (inference mode)", color=GREEN_MED)
        cispo_mem_lbl.next_to(cispo_mem, LEFT, buff=0.15)

        mem_note = label_text(
            "Larger models benefit more: activation memory scales with model size and sequence length",
            color=GREY_LIGHT,
        )
        mem_note.to_edge(DOWN, buff=0.5)

        self.play(LaggedStart(*[FadeIn(r) for r in mem_rows], lag_ratio=0.3),
                  run_time=0.8)
        self.play(FadeIn(ppo_mem), FadeIn(ppo_mem_lbl), run_time=0.4)
        self.play(FadeIn(cispo_mem), FadeIn(cispo_mem_lbl), run_time=0.4)
        self.play(FadeIn(mem_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(mem_title, mem_rows, ppo_mem, ppo_mem_lbl,
                      cispo_mem, cispo_mem_lbl, mem_note)

        # ── 10. CISPO limitations: when does the approximation break down? ────
        limit_title = body_text("Limitations: when does CISPO's approximation fail?",
                                color=YELLOW_MED)
        limit_title.to_edge(UP, buff=0.6)
        self.play(Write(limit_title), run_time=0.7)

        limits = [
            ("Large policy drift", ORANGE_MED,
             "If the policy changes rapidly, most IS ratios fall outside clip bounds.\n"
             "Clipping rejects most old data — batch becomes very small, training slows."),
            ("Very sparse rewards", RED_MED,
             "When only a few responses score well, old samples may all be near clip boundary.\n"
             "CISPO's gate provides less signal than PPO's full IS correction."),
            ("Tiny clip window", YELLOW_MED,
             "An aggressive clip window rejects too much data; too wide accepts too much stale data.\n"
             "Tuning clip bounds requires balancing data freshness against batch size."),
        ]

        limit_rows = VGroup()
        for title_str, col, detail_str in limits:
            t = body_text(title_str, color=col)
            d = label_text(detail_str, color=GREY_LIGHT)
            d.next_to(t, DOWN, buff=0.1)
            d.align_to(t, LEFT)
            limit_rows.add(VGroup(t, d))

        limit_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        limit_rows.move_to(ORIGIN + DOWN * 0.1)
        limit_border = SurroundingRectangle(limit_rows, color=GREY_MED,
                                            buff=0.3, corner_radius=0.12)

        self.play(Create(limit_border), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r) for r in limit_rows], lag_ratio=0.3),
                  run_time=1.2)
        self.wait(1.8)
        self.fade_all(limit_title, limit_rows, limit_border)

        # ── 11. Summary and comparison table ──────────────────────────────────
        summary_title = body_text("Summary: CISPO vs PPO", color=WHITE)
        summary_title.to_edge(UP, buff=0.6)
        self.play(Write(summary_title), run_time=0.7)

        rows = [
            ("IS ratio in gradient",   "Yes — full IS scaling",     "No — stop-gradient"),
            ("Old policy forward",      "Training mode (grad track)", "Inference mode (no grad)"),
            ("Backward passes / step",  "2  (policy + IS ratio)",    "1  (policy only)"),
            ("Async sampling",          "Fragile — high IS variance", "Robust — IS gated out"),
            ("Throughput",              "Baseline  (1x)",             "~2x faster"),
            ("Final performance",       "Strong baseline",            "Comparable to PPO"),
            ("Gradient bias",           "Low within clip region",     "Slightly higher bias"),
            ("Memory (old policy fw)",  "High  (activations stored)", "Low  (inference mode)"),
        ]

        col_headers = ["Property", "PPO", "CISPO"]
        col_colors  = [GREY_LIGHT, BLUE_MED, GREEN_MED]
        col_x       = [-4.5, 0.2, 4.0]

        header_grp = VGroup()
        for hdr, col, x in zip(col_headers, col_colors, col_x):
            h = body_text(hdr, color=col)
            h.move_to([x, 2.5, 0])
            header_grp.add(h)

        divider = Line(LEFT * 6.2, RIGHT * 6.2, color=GREY_MED, stroke_width=1)
        divider.move_to([0, 2.1, 0])

        table_rows = VGroup()
        for i, (prop, ppo_val, cispo_val) in enumerate(rows):
            y = 1.7 - i * 0.48
            p_cell = label_text(prop,      color=GREY_LIGHT)
            b_cell = label_text(ppo_val,   color=BLUE_MED)
            c_cell = label_text(cispo_val, color=GREEN_MED)
            p_cell.move_to([col_x[0], y, 0])
            b_cell.move_to([col_x[1], y, 0])
            c_cell.move_to([col_x[2], y, 0])
            table_rows.add(VGroup(p_cell, b_cell, c_cell))

        self.play(FadeIn(header_grp), Create(divider), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(r) for r in table_rows], lag_ratio=0.12),
                  run_time=1.4)

        next_lbl = label_text("Up next:  MaxRL — maximizing throughput across the full training pipeline",
                              color=ORANGE_MED)
        next_lbl.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(next_lbl), run_time=0.5)
        self.wait(2.5)
        self.fade_all(summary_title, header_grp, divider, table_rows, next_lbl)
