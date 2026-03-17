"""
Scene 27 — Dr. GRPO: GRPO Done Right
Run: manim -pql 27_dr_grpo.py DrGRPOScene
"""
from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class DrGRPOScene(LLMScene):
    def construct(self):

        # ── 1. Title ──────────────────────────────────────────────────────────
        title = self.show_title("Dr. GRPO", "GRPO Done Right — Fixing the Length Bias Bug")
        self.wait(0.8)
        self.fade_all(title)
        self.wait(1.0)

        # ── 2. The bug — short correct vs long wrong ──────────────────────────
        bug_title = body_text("The Bug: Which response gets more gradient?", color=WHITE)
        bug_title.to_edge(UP, buff=0.6)
        self.play(Write(bug_title), run_time=0.7)

        prompt_box = rounded_box(7.0, 0.65,
                                 fill_color=str(GREY_MED) + "33",
                                 stroke_color=GREY_MED,
                                 label="Prompt: Is 7 a prime number?",
                                 label_color=GREY_LIGHT)
        prompt_box.move_to(UP * 2.2)
        self.play(FadeIn(prompt_box), run_time=0.5)

        # Short correct answer
        short_box = rounded_box(4.0, 1.2,
                                fill_color=str(GREEN_MED) + "22",
                                stroke_color=GREEN_MED,
                                label="Response A\n\"Yes.\"  (1 token)\nReward = +1",
                                label_color=GREEN_MED)
        short_box.move_to(LEFT * 3.2 + UP * 0.6)

        # Long wrong answer
        long_box = rounded_box(4.0, 1.2,
                               fill_color=str(RED_MED) + "22",
                               stroke_color=RED_MED,
                               label="Response B\n500-token wrong answer\nReward = -1",
                               label_color=RED_MED)
        long_box.move_to(RIGHT * 3.2 + UP * 0.6)

        self.play(FadeIn(short_box), FadeIn(long_box), run_time=0.7)

        # Gradient bars
        short_grad = Rectangle(width=0.5, height=0.3,
                               fill_color=GREEN_MED, fill_opacity=0.9,
                               stroke_color=GREEN_MED, stroke_width=1)
        short_grad.move_to(LEFT * 3.2 + DOWN * 0.5)
        short_grad_lbl = label_text("Gradient: tiny", color=GREEN_MED)
        short_grad_lbl.next_to(short_grad, DOWN, buff=0.15)

        long_grad = Rectangle(width=4.5, height=0.3,
                              fill_color=RED_MED, fill_opacity=0.9,
                              stroke_color=RED_MED, stroke_width=1)
        long_grad.move_to(RIGHT * 3.2 + DOWN * 0.5)
        long_grad_lbl = label_text("Gradient: HUGE (500x)", color=RED_MED)
        long_grad_lbl.next_to(long_grad, DOWN, buff=0.15)

        self.play(FadeIn(short_grad), FadeIn(short_grad_lbl), run_time=0.5)
        self.play(FadeIn(long_grad), FadeIn(long_grad_lbl), run_time=0.5)

        bug_note = label_text(
            "Original GRPO sums loss over tokens — longer responses get proportionally more gradient",
            color=ORANGE_MED,
        )
        bug_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(bug_note), run_time=0.5)
        self.wait(1.0)
        self.fade_all(bug_title, prompt_box, short_box, long_box,
                      short_grad, short_grad_lbl, long_grad, long_grad_lbl, bug_note)
        self.wait(1.0)

        # ── 3. Visual: two responses side by side — token-by-token loss ────────
        tokens_title = body_text("Loss is summed per token — length determines scale",
                                 color=WHITE)
        tokens_title.to_edge(UP, buff=0.6)
        self.play(Write(tokens_title), run_time=0.7)

        # Short response: 1 token block
        short_lbl = label_text("Response A  |  Reward = +1  |  1 token", color=GREEN_MED)
        short_lbl.move_to(LEFT * 3.3 + UP * 2.0)
        self.play(FadeIn(short_lbl), run_time=0.4)

        tok_a = Rectangle(width=0.6, height=0.55,
                          fill_color=str(GREEN_MED) + "88", fill_opacity=1,
                          stroke_color=GREEN_MED, stroke_width=1.5)
        tok_a.move_to(LEFT * 3.3 + UP * 1.1)
        tok_a_lbl = label_text("\"Yes\"", color=WHITE)
        tok_a_lbl.move_to(tok_a)
        self.play(FadeIn(tok_a), FadeIn(tok_a_lbl), run_time=0.4)

        total_a = label_text("Total loss = 1 unit", color=GREEN_MED)
        total_a.move_to(LEFT * 3.3 + UP * 0.3)
        self.play(FadeIn(total_a), run_time=0.4)

        # Long response: many token blocks (show 10 representative)
        long_lbl = label_text("Response B  |  Reward = -1  |  500 tokens", color=RED_MED)
        long_lbl.move_to(RIGHT * 1.8 + UP * 2.0)
        self.play(FadeIn(long_lbl), run_time=0.4)

        tok_blocks = VGroup()
        n_shown = 10
        for i in range(n_shown):
            blk = Rectangle(width=0.55, height=0.55,
                            fill_color=str(RED_MED) + "77", fill_opacity=1,
                            stroke_color=RED_MED, stroke_width=1)
            blk.move_to(RIGHT * (0.0 + i * 0.62) + UP * 1.1)
            tok_blocks.add(blk)

        ellipsis = label_text("... x500", color=RED_MED)
        ellipsis.next_to(tok_blocks, RIGHT, buff=0.2)

        self.play(LaggedStart(*[FadeIn(b) for b in tok_blocks], lag_ratio=0.07),
                  run_time=0.7)
        self.play(FadeIn(ellipsis), run_time=0.3)

        total_b = label_text("Total loss = 500 units", color=RED_MED)
        total_b.move_to(RIGHT * 2.8 + UP * 0.3)
        self.play(FadeIn(total_b), run_time=0.4)

        scale_note = label_text(
            "500x more gradient from the WRONG answer just because it is longer",
            color=ORANGE_MED,
        )
        scale_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(scale_note), run_time=0.5)
        self.wait(1.0)
        self.fade_all(tokens_title, short_lbl, tok_a, tok_a_lbl, total_a,
                      long_lbl, tok_blocks, ellipsis, total_b, scale_note)
        self.wait(1.0)

        # ── 4. The fix: per-token loss normalisation ───────────────────────────
        fix_title = body_text("The Fix: Divide by response length (per-token mean)",
                              color=WHITE)
        fix_title.to_edge(UP, buff=0.6)
        self.play(Write(fix_title), run_time=0.7)

        # Old approach box
        old_box = rounded_box(4.8, 1.4,
                              fill_color=str(RED_MED) + "22",
                              stroke_color=RED_MED,
                              label="Original GRPO\nLoss = SUM over tokens\nLonger = more gradient",
                              label_color=RED_MED)
        old_box.move_to(LEFT * 3.0 + UP * 0.4)

        vs_lbl = body_text("vs", color=GREY_LIGHT)
        vs_lbl.move_to(ORIGIN + UP * 0.4)

        # New approach box
        new_box = rounded_box(4.8, 1.4,
                              fill_color=str(GREEN_MED) + "22",
                              stroke_color=GREEN_MED,
                              label="Dr. GRPO\nLoss = MEAN over tokens\nAll responses equal weight",
                              label_color=GREEN_MED)
        new_box.move_to(RIGHT * 3.0 + UP * 0.4)

        self.play(FadeIn(old_box), FadeIn(vs_lbl), FadeIn(new_box), run_time=0.8)

        norm_note = label_text(
            "Divide total token-loss by number of effective (non-padding) generated tokens",
            color=GREY_LIGHT,
        )
        norm_note.to_edge(DOWN, buff=0.5)

        mask_note = label_text(
            "Mask out prompt tokens and padding tokens — only generated content counts",
            color=GREY_MED,
        )
        mask_note.next_to(norm_note, UP, buff=0.2)

        self.play(FadeIn(mask_note), run_time=0.4)
        self.play(FadeIn(norm_note), run_time=0.4)
        self.wait(1.0)
        self.fade_all(fix_title, old_box, vs_lbl, new_box, norm_note, mask_note)
        self.wait(1.0)

        # ── 5. Before vs after: bar chart — gradient contribution per response ─
        bars_title = body_text("Gradient contribution per response: Before vs After",
                               color=WHITE)
        bars_title.to_edge(UP, buff=0.6)
        self.play(Write(bars_title), run_time=0.7)

        # Response labels and their (unnorm, norm) gradient sizes
        responses = ["A\n1 tok\n+reward", "B\n50 tok\n+reward", "C\n200 tok\n-reward", "D\n500 tok\n-reward"]
        unnorm = [1, 50, 200, 500]   # raw sums
        normed = [1,  1,   1,   1]   # after per-token norm (same advantage => same contribution)
        colors_resp = [GREEN_MED, GREEN_MED, RED_MED, RED_MED]

        bar_w = 0.7
        max_h = 2.8
        max_val = max(unnorm)

        before_lbl = label_text("Before (original GRPO)", color=ORANGE_MED)
        before_lbl.move_to(LEFT * 3.2 + UP * 2.8)
        self.play(FadeIn(before_lbl), run_time=0.4)

        before_bars = VGroup()
        for i, (resp, val, col) in enumerate(zip(responses, unnorm, colors_resp)):
            h = max_h * (val / max_val)
            bar = Rectangle(width=bar_w, height=max(h, 0.05),
                            fill_color=col, fill_opacity=0.8,
                            stroke_color=col, stroke_width=1)
            bar.move_to([LEFT.get_x() * 3.2 + i * (bar_w + 0.45), -1.8 + h / 2, 0])
            val_lbl = label_text(str(val), color=col)
            val_lbl.next_to(bar, UP, buff=0.08)
            resp_lbl = label_text(resp, color=GREY_LIGHT)
            resp_lbl.next_to(bar, DOWN, buff=0.1)
            before_bars.add(VGroup(bar, val_lbl, resp_lbl))

        self.play(LaggedStart(*[FadeIn(b) for b in before_bars], lag_ratio=0.15),
                  run_time=0.8)

        after_lbl = label_text("After (Dr. GRPO)", color=GREEN_MED)
        after_lbl.move_to(RIGHT * 2.4 + UP * 2.8)
        self.play(FadeIn(after_lbl), run_time=0.4)

        after_bars = VGroup()
        equal_h = max_h * 0.18
        for i, (resp, col) in enumerate(zip(responses, colors_resp)):
            bar = Rectangle(width=bar_w, height=equal_h,
                            fill_color=col, fill_opacity=0.8,
                            stroke_color=col, stroke_width=1)
            bar.move_to([RIGHT.get_x() * 2.4 + i * (bar_w + 0.45), -1.8 + equal_h / 2, 0])
            val_lbl = label_text("1", color=col)
            val_lbl.next_to(bar, UP, buff=0.08)
            resp_lbl = label_text(resp, color=GREY_LIGHT)
            resp_lbl.next_to(bar, DOWN, buff=0.1)
            after_bars.add(VGroup(bar, val_lbl, resp_lbl))

        self.play(LaggedStart(*[FadeIn(b) for b in after_bars], lag_ratio=0.15),
                  run_time=0.8)

        equal_note = label_text(
            "Each response contributes equally — reward signal, not length, decides the gradient",
            color=GREY_LIGHT,
        )
        equal_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(equal_note), run_time=0.5)
        self.wait(1.0)
        self.fade_all(bars_title, before_lbl, before_bars, after_lbl, after_bars, equal_note)
        self.wait(1.0)

        # ── 6. Why this matters: reasoning models + length-hacking ─────────────
        reason_title = body_text("Why it matters: reasoning models and long chains",
                                 color=WHITE)
        reason_title.to_edge(UP, buff=0.6)
        self.play(Write(reason_title), run_time=0.7)

        chain_box = rounded_box(7.5, 1.6,
                                fill_color=str(BLUE_MED) + "22",
                                stroke_color=BLUE_MED,
                                label="Reasoning model generates chain-of-thought:\n"
                                      "Step 1 ... Step 2 ... Step 3 ... [300 tokens total]",
                                label_color=BLUE_LIGHT)
        chain_box.move_to(UP * 1.5)
        self.play(FadeIn(chain_box), run_time=0.6)

        problems = [
            ("Without fix", RED_MED,
             "Gradient amplified by 300x\nModel learns: more tokens = more gradient\nDrifts toward verbosity"),
            ("With fix", GREEN_MED,
             "Gradient averaged over 300 tokens\nModel learns: correct reasoning = reward\nLength follows task needs"),
        ]

        prob_boxes = VGroup()
        for lbl, col, note in problems:
            b = rounded_box(4.8, 1.5,
                            fill_color=str(col) + "22",
                            stroke_color=col,
                            label=f"{lbl}\n{note}",
                            label_color=col)
            prob_boxes.add(b)

        prob_boxes.arrange(RIGHT, buff=0.6)
        prob_boxes.move_to(DOWN * 0.8)
        self.play(FadeIn(prob_boxes[0]), run_time=0.5)
        self.play(FadeIn(prob_boxes[1]), run_time=0.5)

        chain_note = label_text(
            "Reasoning models are most vulnerable — their chains are hundreds of tokens long",
            color=ORANGE_MED,
        )
        chain_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(chain_note), run_time=0.5)
        self.wait(1.0)
        self.fade_all(reason_title, chain_box, prob_boxes, chain_note)
        self.wait(1.0)

        # ── 7. Length hacking explained ────────────────────────────────────────
        hack_title = body_text("Length Hacking: the model learns to pad", color=RED_MED)
        hack_title.to_edge(UP, buff=0.6)
        self.play(Write(hack_title), run_time=0.7)

        stages = [
            ("Early training\nStep 100", GREEN_MED,
             "Avg length: 80 tokens\nReasonable responses"),
            ("Mid training\nStep 1000", YELLOW_MED,
             "Avg length: 200 tokens\nSome padding emerges"),
            ("Late training\nStep 5000", RED_MED,
             "Avg length: 500+ tokens\nVerbosity drift — quality per token falls"),
        ]

        stage_boxes = VGroup()
        for lbl, col, note in stages:
            b = rounded_box(3.2, 1.1,
                            fill_color=str(col) + "22",
                            stroke_color=col,
                            label=lbl,
                            label_color=col)
            n = label_text(note, color=GREY_LIGHT)
            n.next_to(b, DOWN, buff=0.18)
            stage_boxes.add(VGroup(b, n))

        stage_boxes.arrange(RIGHT, buff=0.5)
        stage_boxes.move_to(UP * 0.5)

        st_arrows = VGroup(*[
            Arrow(stage_boxes[i][0].get_right(), stage_boxes[i + 1][0].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.2)
            for i in range(2)
        ])

        self.play(LaggedStart(*[FadeIn(b) for b in stage_boxes], lag_ratio=0.3),
                  run_time=0.9)
        self.play(LaggedStart(*[Create(a) for a in st_arrows], lag_ratio=0.3),
                  run_time=0.5)

        hack_insight = rounded_box(7.5, 0.7,
                                   fill_color=str(RED_MED) + "22",
                                   stroke_color=RED_MED,
                                   label="Model internalises: write more tokens -> more gradient -> better training signal",
                                   label_color=RED_MED)
        hack_insight.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(hack_insight), run_time=0.5)
        self.wait(1.0)
        self.fade_all(hack_title, stage_boxes, st_arrows, hack_insight)
        self.wait(1.0)

        # ── 8. Denominator normalisation: group-level ──────────────────────────
        denom_title = body_text("Denominator Normalisation: group-level token counting",
                                color=WHITE)
        denom_title.to_edge(UP, buff=0.6)
        self.play(Write(denom_title), run_time=0.7)

        # Show group of 4 responses with different lengths
        group_label = label_text("Group of 4 responses sampled for one prompt:", color=GREY_LIGHT)
        group_label.move_to(UP * 2.5)
        self.play(FadeIn(group_label), run_time=0.4)

        resp_data = [
            ("R1", 10,  GREEN_MED),
            ("R2", 10,  GREEN_MED),
            ("R3", 10,  GREEN_MED),
            ("R4", 500, RED_MED),
        ]

        resp_vis = VGroup()
        for i, (name, toks, col) in enumerate(resp_data):
            bar_w_r = min(toks / 100.0, 5.0)
            bar = Rectangle(width=bar_w_r, height=0.45,
                            fill_color=col, fill_opacity=0.75,
                            stroke_color=col, stroke_width=1)
            lbl = label_text(f"{name}: {toks} tokens", color=col)
            lbl.next_to(bar, LEFT, buff=0.25)
            bar.move_to([bar_w_r / 2 - 1.5, 1.6 - i * 0.75, 0])
            lbl.move_to([-3.8, 1.6 - i * 0.75, 0])
            resp_vis.add(VGroup(bar, lbl))

        self.play(LaggedStart(*[FadeIn(r) for r in resp_vis], lag_ratio=0.2), run_time=0.8)

        old_denom = rounded_box(4.8, 0.9,
                                fill_color=str(RED_MED) + "22",
                                stroke_color=RED_MED,
                                label="Old: divide by N=4 responses\nR4 (500 tok) still dominates",
                                label_color=RED_MED)
        old_denom.move_to(LEFT * 2.8 + DOWN * 1.5)

        new_denom = rounded_box(4.8, 0.9,
                                fill_color=str(GREEN_MED) + "22",
                                stroke_color=GREEN_MED,
                                label="New: divide by total effective tokens (530)\nEach token weighted equally",
                                label_color=GREEN_MED)
        new_denom.move_to(RIGHT * 2.8 + DOWN * 1.5)

        self.play(FadeIn(old_denom), run_time=0.5)
        self.play(FadeIn(new_denom), run_time=0.5)

        denom_note = label_text(
            "Group-level normalisation removes the last remaining length scale advantage",
            color=GREY_LIGHT,
        )
        denom_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(denom_note), run_time=0.5)
        self.wait(1.0)
        self.fade_all(denom_title, group_label, resp_vis, old_denom, new_denom, denom_note)
        self.wait(1.0)

        # ── 9. Impact on response quality and length distributions ─────────────
        impact_title = body_text("Impact: response length distribution before vs after",
                                 color=WHITE)
        impact_title.to_edge(UP, buff=0.6)
        self.play(Write(impact_title), run_time=0.7)

        ax_before = Axes(
            x_range=[0, 600, 100],
            y_range=[0, 1.0, 0.2],
            x_length=5.0,
            y_length=3.0,
            axis_config={"color": GREY_MED, "stroke_width": 1.5},
            tips=False,
        )
        ax_before.move_to(LEFT * 3.0 + DOWN * 0.3)

        import numpy as np

        def skewed_dist(x, peak, spread, skew):
            base = np.exp(-0.5 * ((x - peak) / spread) ** 2)
            tail = np.exp(-0.5 * ((x - (peak + skew)) / (spread * 2)) ** 2) * 0.4
            return base + tail

        p_before = ax_before.plot(
            lambda x: skewed_dist(x, 200, 60, 250),
            color=RED_MED, stroke_width=2.5,
        )
        before_area = ax_before.get_area(p_before, x_range=[0, 600], color=RED_MED, opacity=0.15)
        before_ax_lbl = label_text("Before Dr. GRPO\nHeavy tail — verbosity drift", color=RED_MED)
        before_ax_lbl.next_to(ax_before, DOWN, buff=0.25)
        before_x_lbl = label_text("Response length (tokens)", color=GREY_MED)
        before_x_lbl.next_to(ax_before, DOWN, buff=0.8)

        ax_after = Axes(
            x_range=[0, 600, 100],
            y_range=[0, 1.0, 0.2],
            x_length=5.0,
            y_length=3.0,
            axis_config={"color": GREY_MED, "stroke_width": 1.5},
            tips=False,
        )
        ax_after.move_to(RIGHT * 3.0 + DOWN * 0.3)

        p_after = ax_after.plot(
            lambda x: np.exp(-0.5 * ((x - 150) / 55) ** 2),
            color=GREEN_MED, stroke_width=2.5,
        )
        after_area = ax_after.get_area(p_after, x_range=[0, 600], color=GREEN_MED, opacity=0.15)
        after_ax_lbl = label_text("After Dr. GRPO\nTighter — task-driven length", color=GREEN_MED)
        after_ax_lbl.next_to(ax_after, DOWN, buff=0.25)

        self.play(Create(ax_before), Create(ax_after), run_time=0.6)
        self.play(Create(p_before), FadeIn(before_area), FadeIn(before_ax_lbl), run_time=0.6)
        self.play(Create(p_after), FadeIn(after_area), FadeIn(after_ax_lbl), run_time=0.6)

        dist_note = label_text(
            "Distribution tightens around task-appropriate lengths — model stops padding",
            color=GREY_LIGHT,
        )
        dist_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(dist_note), run_time=0.5)
        self.wait(1.0)
        self.fade_all(impact_title, ax_before, ax_after, p_before, p_after,
                      before_area, after_area, before_ax_lbl, after_ax_lbl, dist_note)
        self.wait(1.0)

        # ── 10. Comparison table: Dr. GRPO vs original GRPO ───────────────────
        table_title = body_text("Dr. GRPO vs Original GRPO — Summary", color=WHITE)
        table_title.to_edge(UP, buff=0.6)
        self.play(Write(table_title), run_time=0.7)

        rows = [
            ("Loss computation",   "Sum over tokens",           "Mean over tokens"),
            ("Group aggregation",  "Divide by # responses",     "Divide by total tokens"),
            ("Gradient scale",     "Proportional to length",    "Proportional to reward"),
            ("Length bias",        "Present",                   "Removed"),
            ("Verbosity drift",    "Occurs without controls",   "Prevented"),
            ("Training stability", "Spikes on long batches",    "Bounded gradient norms"),
        ]

        header_row = VGroup(
            label_text("Property",         color=GREY_LIGHT),
            label_text("Original GRPO",    color=RED_MED),
            label_text("Dr. GRPO",         color=GREEN_MED),
        )
        header_row.arrange(RIGHT, buff=0.0)
        header_row[0].set_x(-4.2)
        header_row[1].set_x(0.4)
        header_row[2].set_x(4.2)
        header_row.move_to(UP * 2.3)
        self.play(FadeIn(header_row), run_time=0.4)

        divider = Line(LEFT * 6.4, RIGHT * 6.4, color=GREY_MED, stroke_width=0.8)
        divider.move_to(UP * 1.95)
        self.play(Create(divider), run_time=0.3)

        row_groups = VGroup()
        for idx, (prop, orig, fixed) in enumerate(rows):
            p_txt = label_text(prop,  color=GREY_LIGHT)
            o_txt = label_text(orig,  color=RED_MED)
            f_txt = label_text(fixed, color=GREEN_MED)
            p_txt.set_x(-4.2)
            o_txt.set_x(0.4)
            f_txt.set_x(4.2)
            y_pos = 1.55 - idx * 0.52
            p_txt.set_y(y_pos)
            o_txt.set_y(y_pos)
            f_txt.set_y(y_pos)
            row_groups.add(VGroup(p_txt, o_txt, f_txt))

        self.play(LaggedStart(*[FadeIn(r) for r in row_groups], lag_ratio=0.12),
                  run_time=1.0)

        table_note = label_text(
            "All other aspects of GRPO are unchanged — this is a targeted correction",
            color=GREY_MED,
        )
        table_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(table_note), run_time=0.5)
        self.wait(1.0)
        self.fade_all(table_title, header_row, divider, row_groups, table_note)
        self.wait(1.0)

        # ── 11. Summary ────────────────────────────────────────────────────────
        summary_title = self.show_title("Dr. GRPO", "The One Fix That Changes Everything")
        self.wait(0.5)

        summary_points = [
            ("Bug",  RED_MED,   "Original GRPO sums loss over tokens — longer = more gradient"),
            ("Fix",  GREEN_MED, "Divide by token count — mean instead of sum"),
            ("Why",  BLUE_MED,  "Reasoning chains are long; without fix, model pads to gain gradient"),
            ("Also", YELLOW_MED,"Denominator normalisation: divide by total group tokens too"),
            ("Next", PURPLE_MED,"DAPO extends this with decoupled clipping and dynamic sampling"),
        ]

        point_rows = VGroup()
        for tag, col, text in summary_points:
            tag_txt = label_text(f"[{tag}]", color=col)
            desc_txt = label_text(text, color=WHITE)
            tag_txt.set_x(-5.8)
            desc_txt.set_x(0.6)
            point_rows.add(VGroup(tag_txt, desc_txt))

        point_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        point_rows.move_to(DOWN * 0.5)
        for row in point_rows:
            row[0].align_to(point_rows, LEFT)
            row[1].next_to(row[0], RIGHT, buff=0.4)

        self.play(LaggedStart(*[FadeIn(r) for r in point_rows], lag_ratio=0.2),
                  run_time=1.2)

        next_note = label_text("Up next: DAPO — Decoupled Clip and Dynamic Sampling",
                               color=GREY_MED)
        next_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(next_note), run_time=0.5)
        self.wait(2.0)
        self.fade_all(summary_title, point_rows, next_note)
        self.wait(0.5)
