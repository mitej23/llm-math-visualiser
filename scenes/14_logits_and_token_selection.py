"""
Scene 14 — Logits & Token Selection
Run: manim -pql 14_logits_and_token_selection.py LogitsTokenSelectionScene
"""

from manim import *
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


def softmax(x):
    e = np.exp(np.array(x) - max(x))
    return (e / e.sum()).tolist()


class LogitsTokenSelectionScene(LLMScene):
    def construct(self):
        title = self.show_title("Logits & Token Selection", "Picking the Next Word")
        self.wait(0.5)
        self.fade_all(title)

        # ── 1. From hidden state to logits ─────────────────────────────────────
        pipeline_title = body_text("End of decode step: vector → logits → token",
                                   color=WHITE)
        pipeline_title.to_edge(UP, buff=0.6)
        self.play(Write(pipeline_title), run_time=0.7)

        stages = [
            ("Hidden state\n4096-dim vector",   BLUE_MED),
            ("Linear projection\n4096 → 32,000", PURPLE_MED),
            ("Logits\n32,000 raw scores",         ORANGE_MED),
            ("Softmax\n→ probabilities",           GREEN_MED),
            ("Sample\none token",                  YELLOW_MED),
        ]
        stage_boxes = VGroup()
        for lbl, col in stages:
            b = rounded_box(2.2, 0.85,
                            fill_color=str(col) + "22", stroke_color=col,
                            label=lbl, label_color=col)
            stage_boxes.add(b)

        stage_boxes.arrange(RIGHT, buff=0.45)
        stage_boxes.scale_to_fit_width(13.5)
        stage_boxes.move_to(ORIGIN)

        stage_arrows = VGroup(*[
            Arrow(stage_boxes[i].get_right(), stage_boxes[i + 1].get_left(),
                  color=GREY_MED, buff=0.04, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.18)
            for i in range(len(stage_boxes) - 1)
        ])

        self.play(LaggedStart(*[FadeIn(b) for b in stage_boxes], lag_ratio=0.15),
                  run_time=1.5)
        self.play(LaggedStart(*[Create(a) for a in stage_arrows], lag_ratio=0.1),
                  run_time=0.8)

        # callout: this projection is massive!
        massive_note = label_text("4096 x 32,000 = 131 million parameters in this one layer!",
                                  color=ORANGE_MED)
        massive_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(massive_note), run_time=0.5)
        self.wait(1.2)
        self.fade_all(pipeline_title, stage_boxes, stage_arrows, massive_note)

        # ── 2. Logit values and their meaning ─────────────────────────────────
        logit_title = body_text('Logit values: "The cat sat on the ___"', color=WHITE)
        logit_title.to_edge(UP, buff=0.6)
        self.play(Write(logit_title), run_time=0.7)

        raw_tokens  = ["mat",   "floor", "chair", "roof",  "sun",   "cloud"]
        raw_logits  = [  4.2,     3.1,     2.8,    0.5,    -1.2,    -2.5  ]
        logit_colors = [GREEN_MED, BLUE_MED, BLUE_MED, GREY_MED, GREY_MED, GREY_MED]

        # Build a horizontal logit bar chart (raw values, not probabilities)
        logit_bars = VGroup()
        zero_line = Line(LEFT * 4.0, RIGHT * 4.0, color=GREY_MED, stroke_width=1.5)
        zero_line.move_to(DOWN * 0.8)
        zero_lbl = label_text("logit = 0", color=GREY_MED)
        zero_lbl.next_to(zero_line, RIGHT, buff=0.1)

        max_logit = max(abs(l) for l in raw_logits)
        scale = 1.2

        for i, (tok, logit, col) in enumerate(zip(raw_tokens, raw_logits, logit_colors)):
            h = logit * scale
            bar = Rectangle(width=0.65, height=abs(h),
                            fill_color=col, fill_opacity=0.9,
                            stroke_color=WHITE, stroke_width=1)
            if h >= 0:
                bar.align_to(zero_line, DOWN)
                bar.shift(UP * abs(h))
            else:
                bar.align_to(zero_line, UP)
                bar.shift(DOWN * abs(h))
            bar.shift(RIGHT * (i * 1.05 - 2.6))

            val_lbl = label_text(f"{logit:+.1f}", color=col)
            val_lbl.next_to(bar, UP if h >= 0 else DOWN, buff=0.1)
            tok_lbl = label_text(tok, color=GREY_LIGHT)
            tok_lbl.next_to(bar, DOWN if h >= 0 else DOWN, buff=0.45)
            logit_bars.add(VGroup(bar, val_lbl, tok_lbl))

        interp = label_text('High logit = model is confident   |   Negative logit = very unlikely',
                            color=GREY_LIGHT)
        interp.to_edge(DOWN, buff=0.35)

        self.play(Create(zero_line), FadeIn(zero_lbl), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(b) for b in logit_bars], lag_ratio=0.1), run_time=1.2)
        self.play(FadeIn(interp), run_time=0.5)
        self.wait(1.5)
        self.fade_all(logit_title, zero_line, zero_lbl, logit_bars, interp)

        # ── 3. Probability bar chart ───────────────────────────────────────────
        chart_title = body_text('After "The cat sat on the ___":  probability distribution',
                                color=WHITE)
        chart_title.to_edge(UP, buff=0.6)
        self.play(Write(chart_title), run_time=0.7)

        probs = softmax(raw_logits)
        bar_colors = [GREEN_MED, BLUE_MED, BLUE_MED, GREY_MED, GREY_MED, GREY_MED]

        bars = make_prob_bars(raw_tokens, probs,
                              max_height=3.0, bar_width=0.65,
                              colors=bar_colors)
        bars.move_to(ORIGIN + DOWN * 0.5)

        self.play(LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.1),
                  run_time=1.2)
        self.wait(0.8)

        # Greedy highlight
        greedy_rect = SurroundingRectangle(bars[0], color=YELLOW_MED,
                                           buff=0.08, stroke_width=2, corner_radius=0.1)
        greedy_lbl = label_text("Greedy: always pick the highest", color=YELLOW_MED)
        greedy_lbl.next_to(greedy_rect, UP, buff=0.25)

        self.play(Create(greedy_rect), FadeIn(greedy_lbl), run_time=0.6)
        self.wait(0.8)
        self.fade_all(chart_title, bars, greedy_rect, greedy_lbl)

        # ── 4. Softmax in detail ───────────────────────────────────────────────
        sm_title = body_text("Softmax: logits → exp() → normalise → probabilities", color=WHITE)
        sm_title.to_edge(UP, buff=0.6)
        self.play(Write(sm_title), run_time=0.7)

        example_logits = [4.2, 3.1, 0.5]
        example_tokens = ["mat", "floor", "roof"]
        exp_vals = [round(np.exp(l - max(example_logits)), 3) for l in example_logits]
        exp_sum = round(sum(exp_vals), 3)
        final_probs = [round(e / exp_sum, 3) for e in exp_vals]

        col_headers = VGroup()
        for j, hdr in enumerate(["Token", "Logit", "exp(logit)", "/ sum", "Probability"]):
            h = body_text(hdr, color=GREY_LIGHT)
            h.move_to(LEFT * 4.0 + RIGHT * j * 2.5 + UP * 1.0)
            col_headers.add(h)

        sep_line = Line(LEFT * 5.3, RIGHT * 4.8, color=GREY_MED, stroke_width=1)
        sep_line.move_to(UP * 0.6)

        table_rows = VGroup()
        for i, (tok, logit, expv, prob) in enumerate(
                zip(example_tokens, example_logits, exp_vals, final_probs)):
            row_items = VGroup()
            vals = [tok, f"{logit}", f"e^{logit} = {expv}", f"/ {exp_sum}", f"{prob:.1%}"]
            row_colors_local = [WHITE, ORANGE_MED, BLUE_MED, PURPLE_MED, GREEN_MED]
            for j, (v, rc) in enumerate(zip(vals, row_colors_local)):
                t = label_text(v, color=rc)
                t.move_to(LEFT * 4.0 + RIGHT * j * 2.5 + DOWN * (i * 0.6 - 0.1))
                row_items.add(t)
            table_rows.add(row_items)

        sum_note = label_text(f"Sum of exp values = {exp_sum}  →  divide each to get probability",
                              color=GREY_LIGHT)
        sum_note.to_edge(DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(h) for h in col_headers], lag_ratio=0.1), run_time=0.7)
        self.play(Create(sep_line), run_time=0.3)
        self.play(LaggedStart(*[FadeIn(r) for r in table_rows], lag_ratio=0.2), run_time=1.2)
        self.play(FadeIn(sum_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(sm_title, col_headers, sep_line, table_rows, sum_note)

        # ── 5. Greedy vs. sampling ─────────────────────────────────────────────
        comparison_title = body_text("Greedy vs. Sampling — pros and cons:", color=WHITE)
        comparison_title.to_edge(UP, buff=0.6)
        self.play(Write(comparison_title), run_time=0.6)

        rows_data = [
            ("Greedy",      YELLOW_MED, "Always top token",   "Consistent", "Repetitive"),
            ("Temperature", ORANGE_MED, "Reshape distribution","Creative",   "May be incoherent"),
            ("Top-k",       BLUE_MED,   "Sample from top k",  "Controlled", "Fixed cutoff"),
            ("Top-p",       GREEN_MED,  "Dynamic nucleus",    "Adaptive",   "Best for most tasks"),
        ]

        rows = VGroup()
        for name, col, how, pro, con in rows_data:
            n   = body_text(name, color=col)
            h   = label_text(how, color=WHITE)
            p   = label_text(f"+ {pro}", color=GREEN_MED)
            c   = label_text(f"- {con}", color=RED_MED)
            row = VGroup(n, h, p, c)
            row.arrange(RIGHT, buff=0.45)
            rows.add(row)

        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        rows.move_to(ORIGIN + DOWN * 0.3)
        box = SurroundingRectangle(rows, color=GREY_MED, buff=0.3, corner_radius=0.12)

        self.play(Create(box), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.2),
                  run_time=1.3)
        self.wait(1.5)
        self.fade_all(comparison_title, rows, box)

        # ── 6. Greedy decoding — show determinism ─────────────────────────────
        greedy_title = body_text("Greedy Decoding — same output every single run", color=WHITE)
        greedy_title.to_edge(UP, buff=0.6)
        self.play(Write(greedy_title), run_time=0.6)

        run_probs = softmax([4.2, 3.1, 2.8, 0.5, -1.2, -2.5])
        run_tokens = ["mat", "floor", "chair", "roof", "sun", "cloud"]

        run_bars = make_prob_bars(run_tokens, run_probs,
                                  max_height=2.5, bar_width=0.65,
                                  colors=[GREEN_MED] + [GREY_MED] * 5)
        run_bars.move_to(ORIGIN + DOWN * 0.3)

        self.play(LaggedStart(*[FadeIn(b) for b in run_bars], lag_ratio=0.1), run_time=0.9)

        greedy_arrow = Arrow(run_bars[0].get_top() + UP * 0.05,
                             run_bars[0].get_top() + UP * 0.7,
                             color=YELLOW_MED, buff=0.04, stroke_width=2,
                             max_tip_length_to_length_ratio=0.25)
        greedy_pick = label_text("Always: mat", color=YELLOW_MED)
        greedy_pick.next_to(greedy_arrow, UP, buff=0.1)

        pro_text = label_text("Pro: deterministic, reproducible", color=GREEN_LIGHT)
        pro_text.to_edge(DOWN, buff=0.55)
        con_text = label_text("Con: repetitive loops, no creativity", color=RED_MED)
        con_text.next_to(pro_text, UP, buff=0.15)

        self.play(Create(greedy_arrow), FadeIn(greedy_pick), run_time=0.5)
        self.play(FadeIn(pro_text), FadeIn(con_text), run_time=0.5)
        self.wait(1.5)
        all_greedy = VGroup(run_bars, greedy_arrow, greedy_pick, pro_text, con_text)
        self.fade_all(greedy_title, all_greedy)

        # ── 7. Beam search — exploring multiple paths ─────────────────────────
        beam_title = body_text("Beam Search — explore top 3 paths simultaneously", color=WHITE)
        beam_title.to_edge(UP, buff=0.6)
        self.play(Write(beam_title), run_time=0.6)

        # Show 3 beams as tree nodes
        beam_colors = [BLUE_MED, GREEN_MED, ORANGE_MED]
        root = rounded_box(2.0, 0.55,
                           fill_color=str(GREY_MED) + "22", stroke_color=GREY_MED,
                           label="The sky is", label_color=WHITE)
        root.move_to(UP * 1.6)

        # Step 1 — 3 beams
        step1_items = [
            ("...blue", -0.5, BLUE_MED),
            ("...dark", -0.7, GREEN_MED),
            ("...clear", -0.9, ORANGE_MED),
        ]
        step1_boxes = VGroup()
        step1_arrows = VGroup()
        for i, (lbl, score, col) in enumerate(step1_items):
            b = rounded_box(2.0, 0.55,
                            fill_color=str(col) + "22", stroke_color=col,
                            label=f"{lbl}  [{score}]", label_color=col)
            b.move_to(LEFT * 3.5 + RIGHT * i * 3.5 + UP * 0.3)
            step1_boxes.add(b)
            a = Arrow(root.get_bottom(), b.get_top(),
                      color=GREY_MED, buff=0.05, stroke_width=1.3,
                      max_tip_length_to_length_ratio=0.2)
            step1_arrows.add(a)

        # Step 2 — best 3 of 9 total candidates
        step2_items = [
            ("...blue today", -1.1, BLUE_MED),
            ("...blue and", -1.3, BLUE_MED),
            ("...dark tonight", -1.5, GREEN_MED),
        ]
        step2_boxes = VGroup()
        step2_arrows = VGroup()
        for i, (lbl, score, col) in enumerate(step2_items):
            b = rounded_box(2.2, 0.55,
                            fill_color=str(col) + "22", stroke_color=col,
                            label=f"{lbl}  [{score}]", label_color=col)
            b.move_to(LEFT * 3.5 + RIGHT * i * 3.5 + DOWN * 1.0)
            step2_boxes.add(b)
            a = Arrow(step1_boxes[i // 2 + (1 if i == 2 else 0)].get_bottom(),
                      b.get_top(),
                      color=GREY_MED, buff=0.05, stroke_width=1.3,
                      max_tip_length_to_length_ratio=0.2)
            step2_arrows.add(a)

        winner_box = SurroundingRectangle(step2_boxes[0], color=YELLOW_MED,
                                          buff=0.05, stroke_width=2, corner_radius=0.1)
        winner_lbl = label_text("Best sequence wins", color=YELLOW_MED)
        winner_lbl.to_edge(DOWN, buff=0.4)

        self.play(FadeIn(root), run_time=0.4)
        self.play(LaggedStart(*[Create(a) for a in step1_arrows], lag_ratio=0.1), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(b) for b in step1_boxes], lag_ratio=0.1), run_time=0.7)
        self.play(LaggedStart(*[Create(a) for a in step2_arrows], lag_ratio=0.1), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(b) for b in step2_boxes], lag_ratio=0.1), run_time=0.7)
        self.play(Create(winner_box), FadeIn(winner_lbl), run_time=0.5)
        self.wait(1.5)
        all_beam = VGroup(root, step1_boxes, step1_arrows, step2_boxes,
                          step2_arrows, winner_box, winner_lbl)
        self.fade_all(beam_title, all_beam)

        # ── 8. The unembedding matrix — weight tying ──────────────────────────
        tying_title = body_text("Weight Tying — embedding and unembedding share weights",
                                color=WHITE)
        tying_title.to_edge(UP, buff=0.6)
        self.play(Write(tying_title), run_time=0.7)

        # Show two matrices
        embed_box = rounded_box(3.2, 1.5,
                                fill_color=str(BLUE_MED) + "22", stroke_color=BLUE_MED)
        embed_box.move_to(LEFT * 3.2 + UP * 0.3)
        embed_heading = body_text("Embedding Matrix", color=BLUE_MED)
        embed_heading.move_to(embed_box.get_top() + DOWN * 0.45)
        embed_shape = label_text("[vocab x d_model]\n[32,000 x 4,096]", color=BLUE_LIGHT)
        embed_shape.next_to(embed_heading, DOWN, buff=0.15)
        embed_role = label_text("Token ID -> vector", color=GREY_LIGHT)
        embed_role.next_to(embed_shape, DOWN, buff=0.1)

        unembed_box = rounded_box(3.2, 1.5,
                                  fill_color=str(PURPLE_MED) + "22", stroke_color=PURPLE_MED)
        unembed_box.move_to(RIGHT * 3.2 + UP * 0.3)
        unembed_heading = body_text("Unembedding Matrix", color=PURPLE_MED)
        unembed_heading.move_to(unembed_box.get_top() + DOWN * 0.45)
        unembed_shape = label_text("[d_model x vocab]\n[4,096 x 32,000]", color=PURPLE_MED)
        unembed_shape.next_to(unembed_heading, DOWN, buff=0.15)
        unembed_role = label_text("vector -> logits", color=GREY_LIGHT)
        unembed_role.next_to(unembed_shape, DOWN, buff=0.1)

        tie_arrow1 = DashedLine(embed_box.get_right(), unembed_box.get_left(),
                                color=GREEN_MED, stroke_width=2)
        tie_lbl = body_text("= same weights (transposed)!", color=GREEN_MED)
        tie_lbl.next_to(tie_arrow1, UP, buff=0.15)

        saving = label_text("Saves 131M parameters in Llama 3 8B — with no quality loss",
                            color=YELLOW_MED)
        saving.to_edge(DOWN, buff=0.5)

        used_in = label_text(
            "Used in: GPT-2, Llama, Mistral, Falcon...  Not used in: GPT-3, some large models",
            color=GREY_LIGHT)
        used_in.next_to(saving, UP, buff=0.15)

        self.play(FadeIn(embed_box), FadeIn(unembed_box), run_time=0.6)
        self.play(Write(embed_heading), Write(unembed_heading), run_time=0.6)
        self.play(FadeIn(embed_shape), FadeIn(embed_role),
                  FadeIn(unembed_shape), FadeIn(unembed_role), run_time=0.6)
        self.play(Create(tie_arrow1), Write(tie_lbl), run_time=0.7)
        self.play(FadeIn(saving), FadeIn(used_in), run_time=0.5)
        self.wait(2.0)
        all_tying = VGroup(embed_box, embed_heading, embed_shape, embed_role,
                           unembed_box, unembed_heading, unembed_shape, unembed_role,
                           tie_arrow1, tie_lbl, saving, used_in)
        self.fade_all(tying_title, all_tying)
