"""
Scene 15 — Temperature, Top-k, Top-p
Run: manim -pql 15_temperature_sampling.py TemperatureSamplingScene
"""

from manim import *
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


def softmax(logits, T=1.0):
    x = np.array(logits) / T
    e = np.exp(x - x.max())
    return (e / e.sum()).tolist()


class TemperatureSamplingScene(LLMScene):
    def construct(self):
        title = self.show_title("Temperature · Top-k · Top-p",
                                "Controlling Creativity")
        self.wait(0.5)
        self.fade_all(title)

        # ── 1. Baseline probability distribution ──────────────────────────────
        tokens  = ["cat", "dog", "fish", "bird", "star", "moon"]
        logits  = [4.0,   3.2,   2.5,    1.1,    -0.5,   -1.8]

        section_title = body_text("Baseline distribution  (Temperature = 1.0)", color=WHITE)
        section_title.to_edge(UP, buff=0.6)
        self.play(Write(section_title), run_time=0.6)

        def draw_bars(probs, colors=None, scale=2.8):
            if colors is None:
                colors = [BLUE_MED] * len(tokens)
            bars = make_prob_bars(tokens, probs,
                                  max_height=scale, bar_width=0.65, colors=colors)
            bars.move_to(ORIGIN + DOWN * 0.3)
            return bars

        probs_base = softmax(logits, T=1.0)
        bars_base  = draw_bars(probs_base)
        self.play(LaggedStart(*[FadeIn(b) for b in bars_base], lag_ratio=0.1),
                  run_time=1.0)
        self.wait(0.8)

        # ── 2. Temperature effect ─────────────────────────────────────────────
        for T_val, T_label, T_color, desc in [
            (0.3, "T = 0.3  (cold — focused)",   BLUE_LIGHT, "Top token dominates"),
            (2.0, "T = 2.0  (hot — chaotic)",    RED_MED,    "All tokens more equal"),
            (1.0, "T = 1.0  (normal)",            GREEN_MED,  "Baseline behaviour"),
        ]:
            new_probs = softmax(logits, T=T_val)
            colors = [T_color] * len(tokens)
            bars_new = draw_bars(new_probs, colors=colors)
            new_title = body_text(T_label, color=T_color)
            new_title.to_edge(UP, buff=0.6)
            desc_lbl = label_text(desc, color=GREY_LIGHT)
            desc_lbl.next_to(new_title, DOWN, buff=0.25)

            self.play(
                Transform(section_title, new_title),
                *[Transform(bars_base[i], bars_new[i]) for i in range(len(tokens))],
                FadeIn(desc_lbl),
                run_time=0.9,
            )
            self.wait(1.0)
            self.play(FadeOut(desc_lbl), run_time=0.3)

        self.fade_all(section_title, bars_base)

        # ── 3. Temperature side-by-side comparison ────────────────────────────
        temp_compare_title = body_text("Three temperatures side by side", color=WHITE)
        temp_compare_title.to_edge(UP, buff=0.6)
        self.play(Write(temp_compare_title), run_time=0.6)

        temp_configs = [
            (0.1, "T=0.1\n(very peaked)", BLUE_LIGHT),
            (1.0, "T=1.0\n(moderate)",    GREEN_MED),
            (2.0, "T=2.0\n(flat)",        RED_MED),
        ]

        compare_groups = VGroup()
        for idx, (T_val, T_lbl, T_col) in enumerate(temp_configs):
            probs_t = softmax(logits, T=T_val)
            mini_bars = VGroup()
            for j, (tok, prob, base_col) in enumerate(
                    zip(tokens[:4], probs_t[:4], [T_col] * 4)):
                h = 1.6 * prob
                bar = Rectangle(width=0.3, height=max(h, 0.02),
                                fill_color=base_col, fill_opacity=0.9,
                                stroke_color=WHITE, stroke_width=0.8)
                bar.move_to(RIGHT * (j * 0.42 - 0.63))
                mini_bars.add(bar)

            mini_bars.move_to(ORIGIN)
            lbl = label_text(T_lbl, color=T_col)
            lbl.next_to(mini_bars, UP, buff=0.2)
            grp = VGroup(mini_bars, lbl)
            grp.move_to(LEFT * 3.8 + RIGHT * idx * 3.8 + DOWN * 0.3)
            compare_groups.add(grp)

        caption = label_text("Same logits, different temperature: distribution shape changes dramatically",
                             color=GREY_LIGHT)
        caption.to_edge(DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(g) for g in compare_groups], lag_ratio=0.2),
                  run_time=1.2)
        self.play(FadeIn(caption), run_time=0.5)
        self.wait(1.5)
        self.fade_all(temp_compare_title, compare_groups, caption)

        # ── 4. Top-k visual ───────────────────────────────────────────────────
        topk_title = body_text("Top-k Sampling — keep only the k highest tokens",
                               color=WHITE)
        topk_title.to_edge(UP, buff=0.6)
        self.play(Write(topk_title), run_time=0.7)

        probs_norm = softmax(logits, T=1.0)
        all_bars   = draw_bars(probs_norm)
        self.play(LaggedStart(*[FadeIn(b) for b in all_bars], lag_ratio=0.1),
                  run_time=0.9)

        # k=3: grey out bars 3,4,5 (0-indexed)
        k_line = DashedLine(
            all_bars[2][0].get_right() + RIGHT * 0.1 + DOWN * 3.0,
            all_bars[2][0].get_right() + RIGHT * 0.1 + UP * 0.5,
            color=YELLOW_MED, stroke_width=2,
        )
        k_lbl = label_text("k=3 cutoff", color=YELLOW_MED)
        k_lbl.next_to(k_line, UP, buff=0.1)

        # Dim bars beyond k
        dim_anims = [all_bars[i].animate.set_opacity(0.2) for i in range(3, 6)]
        self.play(Create(k_line), FadeIn(k_lbl), run_time=0.5)
        self.play(*dim_anims, run_time=0.6)

        renorm_note = label_text("After cutoff: renormalise remaining 3 to sum to 100%",
                                 color=BLUE_LIGHT)
        renorm_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(renorm_note), run_time=0.4)
        self.wait(0.8)
        self.fade_all(topk_title, all_bars, k_line, k_lbl, renorm_note)

        # ── 5. Top-p (nucleus) visual ─────────────────────────────────────────
        topp_title = body_text("Top-p (Nucleus) — keep tokens until cumulative prob >= p",
                               color=WHITE)
        topp_title.to_edge(UP, buff=0.6)
        self.play(Write(topp_title), run_time=0.7)

        probs_sorted = sorted(probs_norm, reverse=True)
        cumulative   = 0
        nucleus_size = 0
        for p in probs_sorted:
            cumulative += p
            nucleus_size += 1
            if cumulative >= 0.9:
                break

        p_bars = draw_bars(probs_norm)
        self.play(LaggedStart(*[FadeIn(b) for b in p_bars], lag_ratio=0.1),
                  run_time=0.9)

        # Show cumulative probability labels
        cum = 0
        cum_labels = VGroup()
        for i in range(nucleus_size):
            cum += probs_norm[i]
            cum_lbl = label_text(f"cum={cum*100:.0f}%", color=ORANGE_MED)
            cum_lbl.next_to(p_bars[i], DOWN, buff=0.55)
            cum_labels.add(cum_lbl)

        self.play(LaggedStart(*[FadeIn(l) for l in cum_labels], lag_ratio=0.15), run_time=0.7)

        # Highlight nucleus (first nucleus_size bars)
        nucleus_rect = SurroundingRectangle(
            VGroup(*[p_bars[i] for i in range(nucleus_size)]),
            color=GREEN_MED, buff=0.15, corner_radius=0.1, stroke_width=2,
        )
        p_lbl = label_text(f"Nucleus: top {nucleus_size} tokens cover >=90% probability",
                           color=GREEN_MED)
        p_lbl.to_edge(DOWN, buff=0.4)

        # Dim tokens outside nucleus
        outside_dims = [p_bars[i].animate.set_opacity(0.2) for i in range(nucleus_size, 6)]
        self.play(Create(nucleus_rect), run_time=0.5)
        self.play(*outside_dims, run_time=0.5)
        self.play(FadeIn(p_lbl), run_time=0.4)
        self.wait(1.2)
        self.fade_all(topp_title, p_bars, nucleus_rect, p_lbl, cum_labels)

        # ── 6. Combining all three — full pipeline ────────────────────────────
        pipeline_title = body_text("Full sampling pipeline: logits -> T -> top-k -> top-p -> sample",
                                   color=WHITE)
        pipeline_title.to_edge(UP, buff=0.6)
        self.play(Write(pipeline_title), run_time=0.7)

        pipeline_steps = [
            ("Raw logits\n50,000 scores",      GREY_LIGHT, "Start"),
            ("/ Temperature\nreshaped dist",   BLUE_MED,   "Scale"),
            ("Top-k filter\nkeep top 50",       ORANGE_MED, "Hard cut"),
            ("Top-p filter\nnucleus <=0.9",     GREEN_MED,  "Soft cut"),
            ("Sample!\none token",              YELLOW_MED, "Done"),
        ]

        pipe_boxes = VGroup()
        for step, col, stage in pipeline_steps:
            b = rounded_box(1.9, 0.9,
                            fill_color=str(col) + "22", stroke_color=col,
                            label=step, label_color=col)
            pipe_boxes.add(b)

        pipe_boxes.arrange(RIGHT, buff=0.4)
        pipe_boxes.scale_to_fit_width(13.0)
        pipe_boxes.move_to(ORIGIN + UP * 0.1)

        pipe_arrows = VGroup(*[
            Arrow(pipe_boxes[i].get_right(), pipe_boxes[i + 1].get_left(),
                  color=GREY_MED, buff=0.04, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.2)
            for i in range(len(pipe_boxes) - 1)
        ])

        pipe_note = label_text("Each filter narrows the candidate set progressively",
                               color=GREY_LIGHT)
        pipe_note.to_edge(DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(b) for b in pipe_boxes], lag_ratio=0.15), run_time=1.3)
        self.play(LaggedStart(*[Create(a) for a in pipe_arrows], lag_ratio=0.1), run_time=0.7)
        self.play(FadeIn(pipe_note), run_time=0.4)
        self.wait(1.5)
        self.fade_all(pipeline_title, pipe_boxes, pipe_arrows, pipe_note)

        # ── 7. Repetition penalty ─────────────────────────────────────────────
        rep_title = body_text("Repetition Penalty — stop the model looping", color=WHITE)
        rep_title.to_edge(UP, buff=0.6)
        self.play(Write(rep_title), run_time=0.6)

        # Show a sentence with a repeated token
        repeat_example = label_text(
            '"The cat sat on the mat. The cat sat on the mat. The cat..."',
            color=RED_MED)
        repeat_example.move_to(UP * 1.4)
        repeat_warning = label_text("Without penalty: model loops forever!", color=ORANGE_MED)
        repeat_warning.next_to(repeat_example, DOWN, buff=0.2)

        self.play(FadeIn(repeat_example), FadeIn(repeat_warning), run_time=0.6)

        # Show before/after logits for "the" and "cat"
        before_after_tokens = ["the", "cat", "sat", "on", "bird"]
        before_logits = [4.5, 3.8, 3.2, 2.1, 0.8]
        after_logits  = [4.5 / 1.3, 3.8 / 1.3, 3.2 / 1.3, 2.1, 0.8]  # penalty=1.3 on first 3

        before_label = body_text("Before penalty", color=ORANGE_MED)
        before_label.move_to(LEFT * 4.0 + DOWN * 0.3)
        after_label = body_text("After penalty (1.3)", color=GREEN_MED)
        after_label.move_to(RIGHT * 1.0 + DOWN * 0.3)

        before_row = VGroup()
        after_row = VGroup()
        for i, (tok, bl, al) in enumerate(
                zip(before_after_tokens, before_logits, after_logits)):
            bt = label_text(f"{tok}: {bl:.1f}", color=ORANGE_MED if i < 3 else GREY_LIGHT)
            bt.move_to(LEFT * 4.0 + DOWN * (i * 0.38 + 0.9))
            before_row.add(bt)

            at = label_text(f"{tok}: {al:.2f}", color=GREEN_MED if i < 3 else GREY_LIGHT)
            at.move_to(RIGHT * 1.0 + DOWN * (i * 0.38 + 0.9))
            after_row.add(at)

        penalty_note = label_text("Repeated tokens penalised: logit / 1.3  =  less likely",
                                  color=GREY_LIGHT)
        penalty_note.to_edge(DOWN, buff=0.35)

        self.play(FadeIn(before_label), FadeIn(after_label), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(b) for b in before_row], lag_ratio=0.1), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(a) for a in after_row], lag_ratio=0.1), run_time=0.7)
        self.play(FadeIn(penalty_note), run_time=0.4)
        self.wait(1.5)
        all_rep = VGroup(repeat_example, repeat_warning, before_label, after_label,
                         before_row, after_row, penalty_note)
        self.fade_all(rep_title, all_rep)

        # ── 8. Summary presets ────────────────────────────────────────────────
        preset_title = body_text("Practical guidance — settings by task:", color=WHITE)
        preset_title.to_edge(UP, buff=0.6)
        self.play(Write(preset_title), run_time=0.5)

        presets = [
            ("Creative writing",  "T=0.8,  top_p=0.9,  top_k=off",  ORANGE_MED),
            ("Coding",            "T=0.2,  top_k=10,   top_p=0.95", BLUE_MED),
            ("Factual Q&A",       "T=0,    greedy decoding",          GREEN_MED),
            ("Brainstorming",     "T=1.2,  top_p=0.99, top_k=off",  RED_MED),
            ("General chat",      "T=0.7,  top_p=0.9,  top_k=50",   PURPLE_MED),
        ]
        preset_rows = VGroup()
        for name, setting, col in presets:
            task_box = rounded_box(2.5, 0.5,
                                   fill_color=str(col) + "22",
                                   stroke_color=col,
                                   label=name, label_color=col)
            setting_lbl = label_text(setting, color=WHITE)
            setting_lbl.next_to(task_box, RIGHT, buff=0.5)
            preset_rows.add(VGroup(task_box, setting_lbl))

        preset_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        preset_rows.move_to(ORIGIN + DOWN * 0.15)
        box = SurroundingRectangle(preset_rows, color=GREY_MED,
                                   buff=0.3, corner_radius=0.12)
        self.play(Create(box), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r) for r in preset_rows], lag_ratio=0.15),
                  run_time=1.3)
        self.wait(2.0)
        self.fade_all(preset_title, preset_rows, box)
