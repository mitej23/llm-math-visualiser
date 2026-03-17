"""
Scene 26 — RLOO: REINFORCE Leave-One-Out
Run: manim -pql 26_rloo.py RLOOScene
"""
from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class RLOOScene(LLMScene):
    def construct(self):

        # ── 1. Title ──────────────────────────────────────────────────────────
        title = self.show_title("RLOO", "REINFORCE Leave-One-Out")
        sub = label_text("Simpler. Unbiased. No clipping. No value model.", color=GREY_LIGHT)
        sub.next_to(title, DOWN, buff=0.35)
        self.play(FadeIn(sub), run_time=0.6)
        self.wait(1.0)
        self.fade_all(title, sub)

        # ── 2. Setup: multiple responses to one prompt ─────────────────────
        setup_title = body_text("Setup: sample k responses for every prompt", color=WHITE)
        setup_title.to_edge(UP, buff=0.6)
        self.play(Write(setup_title), run_time=0.7)

        prompt_box = rounded_box(3.8, 0.75, fill_color=str(BLUE_MED) + "22",
                                 stroke_color=BLUE_MED,
                                 label="Prompt: Explain gravity\nin one sentence.",
                                 label_color=BLUE_LIGHT)
        prompt_box.move_to(LEFT * 3.8 + UP * 0.2)

        response_labels = [
            ("R1", "Gravity pulls objects\ntoward each other.", GREEN_MED),
            ("R2", "Mass curves spacetime,\ncausing attraction.", GREEN_MED),
            ("R3", "Things fall down\nbecause of gravity.", ORANGE_MED),
            ("R4", "Gravity is a fundamental\nforce of nature.", GREEN_MED),
        ]

        resp_boxes = VGroup()
        for tag, txt, col in response_labels:
            b = rounded_box(3.2, 0.9, fill_color=str(col) + "22",
                            stroke_color=col,
                            label=f"{tag}: {txt}", label_color=col)
            resp_boxes.add(b)

        resp_boxes.arrange(DOWN, buff=0.22)
        resp_boxes.move_to(RIGHT * 2.2 + UP * 0.0)

        arrows_setup = VGroup(*[
            Arrow(prompt_box.get_right(), b.get_left(), color=GREY_MED,
                  buff=0.05, stroke_width=1.5, max_tip_length_to_length_ratio=0.15)
            for b in resp_boxes
        ])

        self.play(FadeIn(prompt_box), run_time=0.5)
        self.play(LaggedStart(*[Create(a) for a in arrows_setup], lag_ratio=0.12),
                  LaggedStart(*[FadeIn(b) for b in resp_boxes], lag_ratio=0.12),
                  run_time=1.2)

        k_note = label_text(
            "k = 4 responses per prompt — each scored by the reward model",
            color=GREY_LIGHT)
        k_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(k_note), run_time=0.5)
        self.wait(1.0)
        self.fade_all(setup_title, prompt_box, resp_boxes, arrows_setup, k_note)

        # ── 3. The leave-one-out idea ──────────────────────────────────────
        loo_title = body_text("Leave-One-Out: exclude your own reward from your baseline",
                              color=WHITE)
        loo_title.to_edge(UP, buff=0.6)
        self.play(Write(loo_title), run_time=0.7)

        grpo_box = rounded_box(5.2, 1.3,
                               fill_color=str(RED_MED) + "22",
                               stroke_color=RED_MED,
                               label="GRPO baseline for R1:\naverage of R1 + R2 + R3 + R4\nBiased — R1 contaminates its own baseline",
                               label_color=RED_MED)
        grpo_box.move_to(LEFT * 3.0 + DOWN * 0.1)

        rloo_box = rounded_box(5.2, 1.3,
                               fill_color=str(GREEN_MED) + "22",
                               stroke_color=GREEN_MED,
                               label="RLOO baseline for R1:\naverage of R2 + R3 + R4 only\nUnbiased — R1 excluded from its own baseline",
                               label_color=GREEN_LIGHT)
        rloo_box.move_to(RIGHT * 3.0 + DOWN * 0.1)

        vs = body_text("vs", color=GREY_MED)
        vs.move_to(ORIGIN + DOWN * 0.1)

        grpo_lbl = label_text("GRPO", color=RED_MED)
        grpo_lbl.next_to(grpo_box, UP, buff=0.18)
        rloo_lbl = label_text("RLOO", color=GREEN_MED)
        rloo_lbl.next_to(rloo_box, UP, buff=0.18)

        self.play(FadeIn(grpo_box), FadeIn(grpo_lbl), run_time=0.5)
        self.play(FadeIn(vs), run_time=0.3)
        self.play(FadeIn(rloo_box), FadeIn(rloo_lbl), run_time=0.5)

        bias_note = label_text(
            "GRPO shrinks advantages by factor (k-1)/k.  RLOO has zero systematic error.",
            color=YELLOW_MED)
        bias_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(bias_note), run_time=0.5)
        self.wait(1.0)
        self.fade_all(loo_title, grpo_box, grpo_lbl, rloo_box, rloo_lbl, vs, bias_note)

        # ── 4. Visual: 4 responses, highlight each baseline ───────────────
        vis_title = body_text("For each response: baseline = average of the other three",
                              color=WHITE)
        vis_title.to_edge(UP, buff=0.6)
        self.play(Write(vis_title), run_time=0.7)

        reward_data = [
            ("R1", 0.82, GREEN_MED),
            ("R2", 0.91, GREEN_MED),
            ("R3", 0.45, ORANGE_MED),
            ("R4", 0.78, GREEN_MED),
        ]

        card_boxes = VGroup()
        for tag, reward, col in reward_data:
            reward_str = str(reward)
            b = rounded_box(2.6, 1.1,
                            fill_color=str(col) + "22",
                            stroke_color=col,
                            label=f"{tag}\nReward: {reward_str}", label_color=col)
            card_boxes.add(b)

        card_boxes.arrange(RIGHT, buff=0.4)
        card_boxes.move_to(ORIGIN + UP * 0.5)
        self.play(LaggedStart(*[FadeIn(b) for b in card_boxes], lag_ratio=0.15),
                  run_time=0.9)

        # Animate highlighting each response's baseline in turn
        baseline_values = [
            (0.91 + 0.45 + 0.78) / 3,   # baseline for R1
            (0.82 + 0.45 + 0.78) / 3,   # baseline for R2
            (0.82 + 0.91 + 0.78) / 3,   # baseline for R3
            (0.82 + 0.91 + 0.45) / 3,   # baseline for R4
        ]

        highlight_rects = []
        for i in range(4):
            indices = [j for j in range(4) if j != i]
            highlight = SurroundingRectangle(
                VGroup(*[card_boxes[j] for j in indices]),
                color=BLUE_MED, stroke_width=2.5, buff=0.08, corner_radius=0.1)
            highlight_rects.append(highlight)

        adv_labels = []
        for i, (tag, reward, col) in enumerate(reward_data):
            bl = baseline_values[i]
            adv = reward - bl
            sign = "+" if adv >= 0 else ""
            adv_str = f"Baseline: {bl:.2f}   Advantage: {sign}{adv:.2f}"
            lbl = label_text(adv_str, color=YELLOW_MED)
            lbl.next_to(card_boxes, DOWN, buff=0.3)
            adv_labels.append(lbl)

        active_highlight = None
        active_lbl = None
        for i in range(4):
            anims_in = [Create(highlight_rects[i]), FadeIn(adv_labels[i])]
            anims_out = []
            if active_highlight is not None:
                anims_out = [FadeOut(active_highlight), FadeOut(active_lbl)]
            if anims_out:
                self.play(*anims_out, run_time=0.3)
            self.play(*anims_in, run_time=0.5)
            self.wait(0.7)
            active_highlight = highlight_rects[i]
            active_lbl = adv_labels[i]

        if active_highlight:
            self.play(FadeOut(active_highlight), FadeOut(active_lbl), run_time=0.3)
        self.wait(0.5)
        self.fade_all(vis_title, card_boxes)

        # ── 5. GRPO baseline vs RLOO baseline — why RLOO is unbiased ──────
        bias_title = body_text("Why GRPO is biased and RLOO is not", color=WHITE)
        bias_title.to_edge(UP, buff=0.6)
        self.play(Write(bias_title), run_time=0.7)

        grpo_detail = rounded_box(5.4, 2.0,
                                  fill_color=str(RED_MED) + "22",
                                  stroke_color=RED_MED,
                                  label="GRPO  (k = 4)\nBaseline = (r1 + r2 + r3 + r4) / 4\nAdvantage = r1 - baseline\n= r1 - r1/4 - avg(others)\n= (3/4) * (r1 - avg(others))\nEffective signal scaled by 0.75",
                                  label_color=RED_MED)
        grpo_detail.move_to(LEFT * 3.0 + DOWN * 0.1)

        rloo_detail = rounded_box(5.4, 2.0,
                                  fill_color=str(GREEN_MED) + "22",
                                  stroke_color=GREEN_MED,
                                  label="RLOO  (k = 4)\nBaseline = (r2 + r3 + r4) / 3\nAdvantage = r1 - baseline\n= r1 - avg(others)\nFull signal, no shrinkage\nZero systematic error",
                                  label_color=GREEN_LIGHT)
        rloo_detail.move_to(RIGHT * 3.0 + DOWN * 0.1)

        vs2 = body_text("vs", color=GREY_MED)
        vs2.move_to(ORIGIN + DOWN * 0.1)

        self.play(FadeIn(grpo_detail), run_time=0.5)
        self.play(FadeIn(vs2), run_time=0.3)
        self.play(FadeIn(rloo_detail), run_time=0.5)

        shrink_note = label_text(
            "GRPO shrinkage factor: (k-1)/k.  At k=4 this is 0.75.  At k=2 it is only 0.50.",
            color=GREY_LIGHT)
        shrink_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(shrink_note), run_time=0.5)
        self.wait(1.0)
        self.fade_all(bias_title, grpo_detail, rloo_detail, vs2, shrink_note)

        # ── 6. No clipping = simpler and faster ───────────────────────────
        clip_title = body_text("No importance sampling = no clipping needed", color=WHITE)
        clip_title.to_edge(UP, buff=0.6)
        self.play(Write(clip_title), run_time=0.7)

        ppo_row = rounded_box(5.4, 1.6,
                              fill_color=str(ORANGE_MED) + "22",
                              stroke_color=ORANGE_MED,
                              label="PPO / GRPO\nOff-policy: reuse data across steps\nNeeds importance sampling ratio\nRatio must be clipped to [0.8, 1.2]\nExtra hyperparameter to tune",
                              label_color=ORANGE_MED)
        ppo_row.move_to(LEFT * 3.0 + DOWN * 0.0)

        rloo_row = rounded_box(5.4, 1.6,
                               fill_color=str(GREEN_MED) + "22",
                               stroke_color=GREEN_MED,
                               label="RLOO\nOn-policy: fresh data every step\nNo importance sampling ratio\nNothing to clip\nFewer hyperparameters, cleaner code",
                               label_color=GREEN_LIGHT)
        rloo_row.move_to(RIGHT * 3.0 + DOWN * 0.0)

        vs3 = body_text("vs", color=GREY_MED)
        vs3.move_to(ORIGIN + DOWN * 0.0)

        self.play(FadeIn(ppo_row), FadeIn(vs3), FadeIn(rloo_row), run_time=0.7)

        clip_note = label_text(
            "On-policy ratio is always 1.0 — clipping [1-e, 1+e] does nothing, so it is dropped.",
            color=GREY_LIGHT)
        clip_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(clip_note), run_time=0.5)
        self.wait(1.0)
        self.fade_all(clip_title, ppo_row, rloo_row, vs3, clip_note)

        # ── 7. Advantage score visualisation for a batch ──────────────────
        adv_title = body_text("Advantage scores across a batch: who outperformed their siblings?",
                              color=WHITE)
        adv_title.to_edge(UP, buff=0.6)
        self.play(Write(adv_title), run_time=0.7)

        batch_entries = [
            ("R1", +0.30, GREEN_MED),
            ("R2", +0.09, GREEN_MED),
            ("R3", -0.37, RED_MED),
            ("R4", +0.07, GREEN_MED),
            ("R5", +0.44, GREEN_MED),
            ("R6", -0.22, RED_MED),
            ("R7", -0.15, ORANGE_MED),
            ("R8", +0.18, GREEN_MED),
        ]

        bar_group = VGroup()
        bar_width = 1.1
        for i, (tag, adv, col) in enumerate(batch_entries):
            bar_h = abs(adv) * 3.5
            bar = Rectangle(width=bar_width, height=max(bar_h, 0.05),
                            fill_color=col, fill_opacity=0.85,
                            stroke_color=col, stroke_width=1)
            if adv >= 0:
                bar.move_to([i * (bar_width + 0.2) - 3.85, bar_h / 2 + 0.05, 0])
            else:
                bar.move_to([i * (bar_width + 0.2) - 3.85, -(bar_h / 2) - 0.05, 0])

            tag_lbl = label_text(tag, color=col)
            tag_lbl.next_to(bar, DOWN if adv >= 0 else UP, buff=0.1)

            sign = "+" if adv >= 0 else ""
            adv_lbl = label_text(f"{sign}{adv:.2f}", color=col)
            adv_lbl.next_to(bar, UP if adv >= 0 else DOWN, buff=0.08)

            bar_group.add(VGroup(bar, tag_lbl, adv_lbl))

        bar_group.move_to(ORIGIN + DOWN * 0.2)

        baseline_line = DashedLine(
            LEFT * 5.5, RIGHT * 5.5, color=GREY_MED,
            stroke_width=1.5, dash_length=0.15)
        baseline_line.move_to(ORIGIN + DOWN * 0.2)
        zero_lbl = label_text("0  (baseline)", color=GREY_MED)
        zero_lbl.next_to(baseline_line, LEFT, buff=0.1)

        self.play(Create(baseline_line), FadeIn(zero_lbl), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(b) for b in bar_group], lag_ratio=0.1),
                  run_time=1.2)

        adv_note = label_text(
            "Green bars: reward above siblings — increase probability.  "
            "Red bars: below siblings — decrease probability.",
            color=GREY_LIGHT)
        adv_note.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(adv_note), run_time=0.5)
        self.wait(1.0)
        self.fade_all(adv_title, bar_group, baseline_line, zero_lbl, adv_note)

        # ── 8. RLOO vs GRPO vs PPO comparison table ───────────────────────
        table_title = body_text("Algorithm comparison: PPO vs GRPO vs RLOO", color=WHITE)
        table_title.to_edge(UP, buff=0.6)
        self.play(Write(table_title), run_time=0.7)

        col_headers = ["Dimension", "PPO", "GRPO", "RLOO"]
        col_colors  = [GREY_LIGHT, ORANGE_MED, YELLOW_MED, GREEN_MED]
        rows = [
            ("Baseline",       "Learned value fn",  "Full-group mean",   "Leave-one-out mean"),
            ("Baseline bias",  "Low (if trained)",  "Yes — (k-1)/k",     "Zero — unbiased"),
            ("Value model",    "Required",          "Not needed",        "Not needed"),
            ("IS + clipping",  "Yes",               "Yes",               "No"),
            ("Memory (models)","4",                 "3",                 "3"),
            ("Complexity",     "High",              "Medium",            "Low"),
        ]

        col_w = [2.9, 2.3, 2.3, 2.3]
        row_h = 0.52
        x_starts = [-6.3, -3.4, -1.1, 1.2]

        header_group = VGroup()
        for ci, (hdr, col) in enumerate(zip(col_headers, col_colors)):
            hdr_txt = label_text(hdr, color=col)
            hdr_txt.move_to([x_starts[ci] + col_w[ci] / 2, 2.5, 0])
            header_group.add(hdr_txt)

        header_bg = Rectangle(width=13.5, height=0.5,
                              fill_color=str(GREY_MED) + "33", fill_opacity=1,
                              stroke_width=0)
        header_bg.move_to([0.4, 2.5, 0])

        row_group = VGroup()
        for ri, row in enumerate(rows):
            y = 2.0 - ri * row_h
            row_bg_col = str(GREY_DARK) if ri % 2 == 0 else str(GREY_MED) + "22"
            row_bg = Rectangle(width=13.5, height=row_h,
                               fill_color=row_bg_col, fill_opacity=1,
                               stroke_width=0)
            row_bg.move_to([0.4, y, 0])
            row_group.add(row_bg)

            for ci, (cell, col) in enumerate(zip(row, col_colors)):
                cell_txt = label_text(cell, color=col if ci > 0 else GREY_LIGHT)
                cell_txt.move_to([x_starts[ci] + col_w[ci] / 2, y, 0])
                row_group.add(cell_txt)

        self.play(FadeIn(header_bg), FadeIn(header_group), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r) for r in row_group], lag_ratio=0.06),
                  run_time=1.0)
        self.wait(1.0)
        self.fade_all(table_title, header_bg, header_group, row_group)

        # ── 9. What RLOO still struggles with ─────────────────────────────
        limit_title = body_text("RLOO's remaining limitation: no length normalisation",
                                color=WHITE)
        limit_title.to_edge(UP, buff=0.6)
        self.play(Write(limit_title), run_time=0.7)

        short_box = rounded_box(5.2, 1.2,
                                fill_color=str(GREEN_MED) + "22",
                                stroke_color=GREEN_MED,
                                label="Short response (8 tokens)\nGravity pulls mass together.\nReward: 0.70",
                                label_color=GREEN_LIGHT)
        short_box.move_to(LEFT * 3.1 + DOWN * 0.1)

        long_box = rounded_box(5.2, 1.2,
                               fill_color=str(ORANGE_MED) + "22",
                               stroke_color=ORANGE_MED,
                               label="Long response (62 tokens)\nGravity is the force by which...\n(verbose, less precise)\nReward: 0.78",
                               label_color=ORANGE_MED)
        long_box.move_to(RIGHT * 3.1 + DOWN * 0.1)

        self.play(FadeIn(short_box), FadeIn(long_box), run_time=0.6)

        problem_note = label_text(
            "Longer response wins — not because it is better, but because it is longer.",
            color=RED_MED)
        problem_note.to_edge(DOWN, buff=0.55)
        fix_note = label_text(
            "Fix: Dr. GRPO normalises reward by response length — coming next.",
            color=YELLOW_MED)
        fix_note.to_edge(DOWN, buff=0.28)

        self.play(FadeIn(problem_note), run_time=0.5)
        self.play(FadeIn(fix_note), run_time=0.5)
        self.wait(1.0)
        self.fade_all(limit_title, short_box, long_box, problem_note, fix_note)

        # ── 10. Summary ───────────────────────────────────────────────────
        sum_title = self.show_title("RLOO — Summary")
        self.wait(0.4)

        summary_rows = [
            ("Leave-one-out baseline",   "Exclude your own reward — unbiased, zero shrinkage",    GREEN_MED),
            ("No value model",           "Saves memory and eliminates critic training overhead",    GREEN_MED),
            ("No importance sampling",   "On-policy: ratio = 1.0, clipping unnecessary",           GREEN_MED),
            ("KL penalty still applies", "Keeps policy close to reference, prevents reward hacking",YELLOW_MED),
            ("Matches PPO quality",      "Empirically competitive with lower resource cost",         GREEN_MED),
            ("No length normalisation",  "Dr. GRPO addresses this — divide advantage by length",    ORANGE_MED),
        ]

        sum_boxes = VGroup()
        for key, val, col in summary_rows:
            key_txt = label_text(key, color=col)
            sep = label_text("  —  ", color=GREY_MED)
            val_txt = label_text(val, color=GREY_LIGHT)
            row = VGroup(key_txt, sep, val_txt)
            row.arrange(RIGHT, buff=0.0)
            bg = SurroundingRectangle(row, color=str(col) + "33",
                                      fill_color=str(col) + "11",
                                      fill_opacity=1, buff=0.18,
                                      corner_radius=0.1)
            sum_boxes.add(VGroup(bg, row))

        sum_boxes.arrange(DOWN, buff=0.18)
        sum_boxes.scale_to_fit_width(13.0)
        sum_boxes.move_to(ORIGIN + DOWN * 0.35)

        self.play(LaggedStart(*[FadeIn(b) for b in sum_boxes], lag_ratio=0.12),
                  run_time=1.4)

        next_lbl = label_text("Up next: Dr. GRPO — length-normalised advantage estimation",
                              color=BLUE_LIGHT)
        next_lbl.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(next_lbl), run_time=0.5)
        self.wait(1.5)
        self.fade_all(sum_title, sum_boxes, next_lbl)
