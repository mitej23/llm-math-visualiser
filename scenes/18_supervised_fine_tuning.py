"""
Scene 18 — Supervised Fine-Tuning (SFT)
Run: manim -pql 18_supervised_fine_tuning.py SFTScene
"""

from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class SFTScene(LLMScene):
    def construct(self):
        title = self.show_title("Supervised Fine-Tuning", "From Base Model to Assistant")
        self.wait(0.5)
        self.fade_all(title)

        # ── 1. Base model vs SFT model comparison ─────────────────────────────
        compare_title = body_text("Same knowledge — very different behaviour:", color=WHITE)
        compare_title.to_edge(UP, buff=0.6)
        self.play(Write(compare_title), run_time=0.6)

        prompt = rounded_box(5.5, 0.65, stroke_color=GREY_MED,
                             label='"Write a poem about autumn."',
                             label_color=GREY_LIGHT)
        prompt.shift(UP * 1.5)
        self.play(FadeIn(prompt), run_time=0.5)

        base_box = rounded_box(4.5, 1.5,
                               fill_color=GREY_DARK, stroke_color=GREY_MED,
                               label="Base model:\n\"Autumn, Autumn — See also: Fall,\nSeason, Temperature...\"",
                               label_color=GREY_MED)
        base_box.shift(LEFT * 3 + DOWN * 0.3)
        base_lbl = label_text("Continues text like a web page", color=GREY_MED)
        base_lbl.next_to(base_box, DOWN, buff=0.2)

        sft_box = rounded_box(4.5, 1.5,
                              fill_color=GREEN_DARK, stroke_color=GREEN_MED,
                              label="SFT model:\n\"Golden leaves drift down,\nWhispering of summer gone...\"",
                              label_color=GREEN_LIGHT)
        sft_box.shift(RIGHT * 3 + DOWN * 0.3)
        sft_lbl = label_text("Responds as a helpful assistant", color=GREEN_MED)
        sft_lbl.next_to(sft_box, DOWN, buff=0.2)

        arr_base = Arrow(prompt.get_bottom(), base_box.get_top(),
                         color=GREY_MED, buff=0.05, stroke_width=1.5,
                         max_tip_length_to_length_ratio=0.18)
        arr_sft  = Arrow(prompt.get_bottom(), sft_box.get_top(),
                         color=GREEN_MED, buff=0.05, stroke_width=1.5,
                         max_tip_length_to_length_ratio=0.18)

        self.play(Create(arr_base), FadeIn(base_box), FadeIn(base_lbl), run_time=0.8)
        self.play(Create(arr_sft),  FadeIn(sft_box),  FadeIn(sft_lbl),  run_time=0.8)
        self.wait(1.2)
        self.fade_all(compare_title, prompt, arr_base, base_box, base_lbl,
                      arr_sft, sft_box, sft_lbl)

        # ── 2. The SFT dataset — demo pairs ───────────────────────────────────
        dataset_title = body_text("SFT Dataset: (prompt, ideal response) pairs",
                                  color=WHITE)
        dataset_title.to_edge(UP, buff=0.6)
        self.play(Write(dataset_title), run_time=0.6)

        examples = [
            ('"Explain black holes simply."',
             '"A black hole is a region where gravity is so strong\nthat nothing — not even light — can escape."'),
            ('"Summarise this article in 3 bullets."',
             '"• Main finding  • Supporting evidence  • Conclusion"'),
            ('"Is this email spam?"',
             '"Yes — it contains common spam indicators..."'),
        ]

        ex_group = VGroup()
        for prompt_t, resp_t in examples:
            p_box = rounded_box(5.5, 0.65, fill_color=BLUE_DARK,
                                stroke_color=BLUE_MED, label=prompt_t, label_color=BLUE_LIGHT)
            r_box = rounded_box(5.5, 0.85, fill_color=GREEN_DARK,
                                stroke_color=GREEN_MED, label=resp_t, label_color=GREEN_LIGHT)
            r_box.next_to(p_box, DOWN, buff=0.1)
            ex_group.add(VGroup(p_box, r_box))

        ex_group.arrange(DOWN, buff=0.35)
        ex_group.scale_to_fit_height(5.0)
        ex_group.move_to(ORIGIN + DOWN * 0.2)

        self.play(LaggedStart(*[FadeIn(e) for e in ex_group], lag_ratio=0.3),
                  run_time=1.5)
        self.wait(1)
        self.fade_all(dataset_title, ex_group)

        # ── 3. Three phases of RLHF — position SFT in context ─────────────────
        phase_title = body_text("SFT is Phase 1 of 3 in RLHF:", color=WHITE)
        phase_title.to_edge(UP, buff=0.6)
        self.play(Write(phase_title), run_time=0.6)

        phases = [
            ("Phase 1\nSFT",          GREEN_MED,   "Imitate demos"),
            ("Phase 2\nReward Model", ORANGE_MED,  "Learn to judge"),
            ("Phase 3\nRL + PPO",     PURPLE_MED,  "Exceed demos"),
        ]
        phase_boxes = VGroup()
        for lbl, col, sub in phases:
            b = rounded_box(2.8, 1.0, fill_color=str(col) + "22",
                            stroke_color=col, label=lbl, label_color=WHITE)
            s = label_text(sub, color=GREY_LIGHT)
            s.next_to(b, DOWN, buff=0.2)
            phase_boxes.add(VGroup(b, s))

        phase_boxes.arrange(RIGHT, buff=0.8)
        phase_boxes.move_to(ORIGIN)

        phase_arrows = VGroup(*[
            Arrow(phase_boxes[i][0].get_right(), phase_boxes[i + 1][0].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.2)
            for i in range(2)
        ])

        # Highlight phase 1
        hl = SurroundingRectangle(phase_boxes[0], color=YELLOW_MED,
                                  buff=0.12, corner_radius=0.1, stroke_width=2)

        self.play(LaggedStart(*[FadeIn(b) for b in phase_boxes], lag_ratio=0.3),
                  run_time=1.2)
        self.play(LaggedStart(*[Create(a) for a in phase_arrows], lag_ratio=0.3),
                  run_time=0.7)
        self.play(Create(hl), run_time=0.5)

        key_note = label_text(
            "Source: Ouyang et al., InstructGPT (2022)  —  1.3B SFT model beats 175B base",
            color=GREY_MED,
        )
        key_note.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(key_note), run_time=0.5)
        self.wait(2)
        self.fade_all(phase_title, phase_boxes, phase_arrows, hl, key_note)

        # ── 4. Instruction following format ───────────────────────────────────
        fmt_title = body_text("ChatML template: the model learns conversation format",
                              color=WHITE)
        fmt_title.to_edge(UP, buff=0.6)
        self.play(Write(fmt_title), run_time=0.7)

        template_lines = [
            ("<|system|>",           PURPLE_MED),
            ("You are a helpful assistant.",  GREY_LIGHT),
            ("<|user|>",             BLUE_MED),
            ("What is a black hole?",         BLUE_LIGHT),
            ("<|assistant|>",        GREEN_MED),
            ("A black hole is a region...",   GREEN_LIGHT),
            ("<|end|>",              ORANGE_MED),
        ]

        tmpl_group = VGroup()
        for line_txt, col in template_lines:
            t = code_text(line_txt, color=col)
            tmpl_group.add(t)

        tmpl_group.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        tmpl_group.move_to(ORIGIN + DOWN * 0.1)

        border = SurroundingRectangle(tmpl_group, color=GREY_MED,
                                      buff=0.3, corner_radius=0.12)

        fmt_note = label_text(
            "Special tokens tell the model who is speaking and when to stop",
            color=YELLOW_MED,
        )
        fmt_note.to_edge(DOWN, buff=0.4)

        self.play(Create(border), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(t) for t in tmpl_group], lag_ratio=0.15),
                  run_time=1.5)
        self.play(FadeIn(fmt_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(fmt_title, tmpl_group, border, fmt_note)

        # ── 5. What actually changes ───────────────────────────────────────────
        change_title = body_text("SFT changes weights very little — format, not knowledge",
                                 color=WHITE)
        change_title.to_edge(UP, buff=0.6)
        self.play(Write(change_title), run_time=0.7)

        # Weight change comparison bars
        change_data = [
            ("Pretraining (all knowledge)",   100, BLUE_MED,   "100% change from random init"),
            ("SFT weight update",              2,  GREEN_MED,  "~1-3% magnitude change"),
        ]

        bar_group = VGroup()
        for label_s, pct, col, desc_s in change_data:
            bar = Rectangle(width=pct * 0.12, height=0.6,
                            fill_color=col, fill_opacity=0.8,
                            stroke_color=col, stroke_width=1.5)
            bar.align_to(ORIGIN, LEFT)
            bar.shift(LEFT * 6)
            bar_lbl = label_text(label_s, color=col)
            bar_lbl.next_to(bar, LEFT, buff=0.2)
            desc_lbl = label_text(desc_s, color=GREY_LIGHT)
            desc_lbl.next_to(bar, RIGHT, buff=0.2)
            bar_group.add(VGroup(bar_lbl, bar, desc_lbl))

        bar_group.arrange(DOWN, aligned_edge=LEFT, buff=0.6)
        bar_group.move_to(ORIGIN + DOWN * 0.2)

        insight = label_text(
            "The knowledge was already there. SFT teaches the FORMAT.",
            color=YELLOW_MED,
        )
        insight.to_edge(DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(b) for b in bar_group], lag_ratio=0.4),
                  run_time=1.2)
        self.play(FadeIn(insight), run_time=0.5)
        self.wait(1.5)
        self.fade_all(change_title, bar_group, insight)

        # ── 6. Dataset quality vs quantity ─────────────────────────────────────
        qual_title = body_text("Quality vs Quantity: 10k clean often beats 100k noisy",
                               color=WHITE)
        qual_title.to_edge(UP, buff=0.6)
        self.play(Write(qual_title), run_time=0.7)

        # Two comparison columns
        noisy_lbl = body_text("100k noisy examples", color=ORANGE_MED)
        noisy_lbl.move_to(LEFT * 3.5 + UP * 1.5)

        clean_lbl = body_text("10k clean examples", color=GREEN_MED)
        clean_lbl.move_to(RIGHT * 2.5 + UP * 1.5)

        noisy_bar = Rectangle(width=1.5, height=3.5,
                              fill_color=str(ORANGE_MED) + "44",
                              stroke_color=ORANGE_MED, stroke_width=2)
        noisy_bar.move_to(LEFT * 3.5 + DOWN * 0.3)
        noisy_bar_lbl = label_text("Quality\nscore: 61%", color=ORANGE_MED)
        noisy_bar_lbl.next_to(noisy_bar, DOWN, buff=0.2)

        clean_bar = Rectangle(width=1.5, height=4.2,
                              fill_color=str(GREEN_MED) + "44",
                              stroke_color=GREEN_MED, stroke_width=2)
        clean_bar.move_to(RIGHT * 2.5 + DOWN * 0.1)
        clean_bar_lbl = label_text("Quality\nscore: 78%", color=GREEN_MED)
        clean_bar_lbl.next_to(clean_bar, DOWN, buff=0.2)

        divider4 = Line(UP * 2.0, DOWN * 2.5, color=GREY_MED, stroke_width=1)

        qual_note = label_text(
            "LIMA (2023): 1,000 curated examples matched 100k-scale dataset performance",
            color=YELLOW_MED,
        )
        qual_note.to_edge(DOWN, buff=0.4)

        self.play(FadeIn(noisy_lbl), FadeIn(clean_lbl), Create(divider4), run_time=0.5)
        self.play(FadeIn(noisy_bar), FadeIn(noisy_bar_lbl), run_time=0.6)
        self.play(FadeIn(clean_bar), FadeIn(clean_bar_lbl), run_time=0.6)
        self.play(FadeIn(qual_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(qual_title, noisy_lbl, clean_lbl, divider4,
                      noisy_bar, noisy_bar_lbl, clean_bar, clean_bar_lbl, qual_note)

        # ── 7. LoRA — Fine-tuning efficiently ─────────────────────────────────
        lora_title = body_text("LoRA: train tiny adapter matrices, not all 7B weights",
                               color=WHITE)
        lora_title.to_edge(UP, buff=0.6)
        self.play(Write(lora_title), run_time=0.7)

        # Full fine-tuning
        full_lbl = body_text("Full fine-tuning", color=RED_MED)
        full_lbl.move_to(LEFT * 3.5 + UP * 1.5)

        full_matrix = Rectangle(width=2.5, height=2.5,
                                fill_color=str(RED_MED) + "33",
                                stroke_color=RED_MED, stroke_width=2)
        full_matrix.move_to(LEFT * 3.5 + DOWN * 0.1)
        full_matrix_lbl = label_text("W (4096x4096)\n= 16M params updated", color=RED_MED)
        full_matrix_lbl.next_to(full_matrix, DOWN, buff=0.2)

        # LoRA
        lora_lbl = body_text("LoRA (rank=16)", color=GREEN_MED)
        lora_lbl.move_to(RIGHT * 2.5 + UP * 1.5)

        lora_a = Rectangle(width=0.4, height=2.5,
                           fill_color=str(GREEN_MED) + "55",
                           stroke_color=GREEN_MED, stroke_width=2)
        lora_b = Rectangle(width=2.5, height=0.4,
                           fill_color=str(BLUE_MED) + "55",
                           stroke_color=BLUE_MED, stroke_width=2)
        lora_a.move_to(RIGHT * 1.8 + DOWN * 0.1)
        lora_b.move_to(RIGHT * 3.4 + DOWN * 0.1)

        lora_a_lbl = label_text("A\n(4096x16)\n65k params", color=GREEN_MED)
        lora_a_lbl.next_to(lora_a, DOWN, buff=0.2)
        lora_b_lbl = label_text("B\n(16x4096)\n65k params", color=BLUE_MED)
        lora_b_lbl.next_to(lora_b, DOWN, buff=0.2)

        times_lbl = label_text("x", color=GREY_LIGHT)
        times_lbl.move_to((lora_a.get_center() + lora_b.get_center()) / 2)

        lora_saving = label_text("130k vs 16M params = 120x fewer parameters!", color=YELLOW_MED)
        lora_saving.to_edge(DOWN, buff=0.4)

        divider5 = Line(UP * 2.0, DOWN * 2.5, color=GREY_MED, stroke_width=1)

        self.play(FadeIn(full_lbl), FadeIn(lora_lbl), Create(divider5), run_time=0.5)
        self.play(FadeIn(full_matrix), FadeIn(full_matrix_lbl), run_time=0.6)
        self.play(FadeIn(lora_a), FadeIn(lora_a_lbl),
                  FadeIn(lora_b), FadeIn(lora_b_lbl),
                  FadeIn(times_lbl), run_time=0.8)
        self.play(FadeIn(lora_saving), run_time=0.5)
        self.wait(1.5)
        self.fade_all(lora_title, full_lbl, lora_lbl, divider5,
                      full_matrix, full_matrix_lbl,
                      lora_a, lora_a_lbl, lora_b, lora_b_lbl,
                      times_lbl, lora_saving)

        # ── 8. When SFT is enough ──────────────────────────────────────────────
        when_title = body_text("When SFT is enough — and when it isn't:", color=WHITE)
        when_title.to_edge(UP, buff=0.6)
        self.play(Write(when_title), run_time=0.7)

        enough_items = [
            ("Translation",          GREEN_MED),
            ("Summarization",        GREEN_MED),
            ("Classification",       GREEN_MED),
            ("Basic Q&A",            GREEN_MED),
        ]
        not_enough_items = [
            ("Preference alignment", ORANGE_MED),
            ("Safety / refusals",    ORANGE_MED),
            ("Complex reasoning",    ORANGE_MED),
            ("Novel edge cases",     ORANGE_MED),
        ]

        enough_lbl = body_text("SFT works well:", color=GREEN_MED)
        enough_lbl.move_to(LEFT * 3.5 + UP * 1.5)

        not_enough_lbl = body_text("Need RLHF too:", color=ORANGE_MED)
        not_enough_lbl.move_to(RIGHT * 2.5 + UP * 1.5)

        enough_group = VGroup()
        for txt, col in enough_items:
            t = label_text("  " + txt, color=col)
            enough_group.add(t)
        enough_group.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        enough_group.next_to(enough_lbl, DOWN, buff=0.3)

        not_group = VGroup()
        for txt, col in not_enough_items:
            t = label_text("  " + txt, color=col)
            not_group.add(t)
        not_group.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        not_group.next_to(not_enough_lbl, DOWN, buff=0.3)

        divider6 = Line(UP * 2.0, DOWN * 2.0, color=GREY_MED, stroke_width=1)

        sft_limit = label_text(
            "SFT ceiling: model can only be as good as its training demonstrations",
            color=YELLOW_MED,
        )
        sft_limit.to_edge(DOWN, buff=0.4)

        self.play(FadeIn(enough_lbl), FadeIn(not_enough_lbl), Create(divider6), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(e) for e in enough_group], lag_ratio=0.15),
                  run_time=0.8)
        self.play(LaggedStart(*[FadeIn(n) for n in not_group], lag_ratio=0.15),
                  run_time=0.8)
        self.play(FadeIn(sft_limit), run_time=0.5)
        self.wait(1.5)
        self.fade_all(when_title, enough_lbl, not_enough_lbl, divider6,
                      enough_group, not_group, sft_limit)

        # ── 9. Multi-task SFT ──────────────────────────────────────────────────
        multi_title = body_text("Multi-task SFT: mix diverse examples for robustness",
                                color=WHITE)
        multi_title.to_edge(UP, buff=0.6)
        self.play(Write(multi_title), run_time=0.7)

        task_types = [
            ("Coding\n15%",       BLUE_MED,    3),
            ("Reasoning\n10%",    GREEN_MED,   2.5),
            ("Conversation\n25%", ORANGE_MED,  4),
            ("Q&A\n25%",          PURPLE_MED,  4),
            ("Creative\n10%",     YELLOW_MED,  2.5),
            ("Instructions\n15%", RED_MED,     3),
        ]

        task_boxes = VGroup()
        for lbl, col, w in task_types:
            b = rounded_box(w * 0.45, 1.1,
                            fill_color=str(col) + "33",
                            stroke_color=col, label=lbl, label_color=WHITE)
            task_boxes.add(b)

        task_boxes.arrange(RIGHT, buff=0.3)
        task_boxes.scale_to_fit_width(13.5)
        task_boxes.move_to(ORIGIN + DOWN * 0.1)

        multi_note = label_text(
            "Mixing tasks prevents overfitting to one format — produces general assistants",
            color=GREY_LIGHT,
        )
        multi_note.to_edge(DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(b) for b in task_boxes], lag_ratio=0.15),
                  run_time=1.5)
        self.play(FadeIn(multi_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(multi_title, task_boxes, multi_note)
