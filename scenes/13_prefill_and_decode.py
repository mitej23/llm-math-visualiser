"""
Scene 13 — Prefill & Decode
Run: manim -pql 13_prefill_and_decode.py PrefillDecodeScene
"""

from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class PrefillDecodeScene(LLMScene):
    def construct(self):
        title = self.show_title("Prefill & Decode", "Two Phases of Inference")
        self.wait(0.5)
        self.fade_all(title)

        # ── 1. Timeline bar ────────────────────────────────────────────────────
        timeline_title = body_text('What happens when you send a message to an LLM:',
                                   color=WHITE)
        timeline_title.to_edge(UP, buff=0.6)
        self.play(Write(timeline_title), run_time=0.7)

        # Horizontal timeline
        line = Line(LEFT * 5.5, RIGHT * 5.5, color=GREY_MED, stroke_width=2)
        line.shift(DOWN * 0.5)

        # Prefill block
        prefill_rect = Rectangle(width=3.5, height=1.0,
                                 fill_color=BLUE_DARK, fill_opacity=0.9,
                                 stroke_color=BLUE_MED, stroke_width=2)
        prefill_rect.move_to(LEFT * 3.2 + DOWN * 0.5)
        prefill_lbl = body_text("PREFILL", color=BLUE_MED)
        prefill_lbl.move_to(prefill_rect)

        # Decode block
        decode_rect = Rectangle(width=5.5, height=1.0,
                                fill_color=GREEN_DARK, fill_opacity=0.9,
                                stroke_color=GREEN_MED, stroke_width=2)
        decode_rect.move_to(RIGHT * 1.7 + DOWN * 0.5)
        decode_lbl = body_text("DECODE", color=GREEN_MED)
        decode_lbl.move_to(decode_rect)

        ttft_line = DashedLine(
            prefill_rect.get_right() + UP * 0.8,
            prefill_rect.get_right() + DOWN * 0.8,
            color=YELLOW_MED, stroke_width=2,
        )
        ttft_lbl = label_text("TTFT\n(first token)", color=YELLOW_MED)
        ttft_lbl.next_to(ttft_line, UP, buff=0.1)

        user_arrow = Arrow(LEFT * 5.3 + DOWN * 1.4, LEFT * 5.3 + DOWN * 0.9,
                           color=GREY_LIGHT, buff=0.05, stroke_width=1.5)
        user_lbl = label_text("You press\nEnter", color=GREY_LIGHT)
        user_lbl.next_to(user_arrow, DOWN, buff=0.05)

        self.play(Create(line), GrowArrow(user_arrow), FadeIn(user_lbl), run_time=0.6)
        self.play(FadeIn(prefill_rect), Write(prefill_lbl), run_time=0.7)
        self.play(Create(ttft_line), FadeIn(ttft_lbl), run_time=0.5)
        self.play(FadeIn(decode_rect), Write(decode_lbl), run_time=0.7)
        self.wait(0.8)

        # annotations below
        prefill_note = label_text("Process ALL input tokens\nat once (parallel)",
                                  color=BLUE_LIGHT)
        prefill_note.next_to(prefill_rect, DOWN, buff=0.35)
        decode_note = label_text("Generate one token\nat a time (sequential)",
                                 color=GREEN_LIGHT)
        decode_note.next_to(decode_rect, DOWN, buff=0.35)

        self.play(FadeIn(prefill_note), FadeIn(decode_note), run_time=0.6)
        self.wait(1.2)
        self.fade_all(timeline_title, line, prefill_rect, prefill_lbl,
                      decode_rect, decode_lbl, ttft_line, ttft_lbl,
                      user_arrow, user_lbl, prefill_note, decode_note)

        # ── 2. Bottleneck comparison ───────────────────────────────────────────
        bottleneck_title = body_text("Each phase has a different bottleneck:", color=WHITE)
        bottleneck_title.to_edge(UP, buff=0.6)
        self.play(Write(bottleneck_title), run_time=0.6)

        rows_data = [
            ("Prefill", BLUE_MED, "Parallel (like prefill)", "GPU compute\n(matrix multiplications)",
             "Longer prompt = longer wait"),
            ("Decode",  GREEN_MED,"Sequential (one by one)",  "Memory bandwidth\n(read KV cache each step)",
             "Longer context = slower tokens/s"),
        ]

        comparison = VGroup()
        for phase, col, parallelism, bottleneck, effect in rows_data:
            phase_lbl = body_text(phase, color=col)
            par_lbl   = label_text(parallelism, color=WHITE)
            bot_lbl   = label_text(bottleneck, color=ORANGE_MED)
            eff_lbl   = label_text(effect, color=GREY_LIGHT)
            row = VGroup(phase_lbl, par_lbl, bot_lbl, eff_lbl)
            row.arrange(RIGHT, buff=0.5)
            comparison.add(row)

        comparison.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        comparison.move_to(ORIGIN + DOWN * 0.2)
        box = SurroundingRectangle(comparison, color=GREY_MED, buff=0.35, corner_radius=0.15)

        self.play(Create(box), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r) for r in comparison], lag_ratio=0.3),
                  run_time=1.2)
        self.wait(1.2)
        self.fade_all(bottleneck_title, comparison, box)

        # ── 3. Speculative decoding callout ───────────────────────────────────
        spec_title = body_text("Speculative Decoding — faster generation trick:", color=WHITE)
        spec_title.to_edge(UP, buff=0.6)
        self.play(Write(spec_title), run_time=0.6)

        steps = [
            ("Small\nDraft Model",    ORANGE_MED, "Quickly proposes\n5-10 tokens"),
            ("Large\nTarget Model",   BLUE_MED,   "Verifies all in\none parallel pass"),
            ("Accept /\nReject",      GREEN_MED,  "Keep correct tokens,\nregenerate rest"),
        ]

        step_boxes = VGroup()
        for lbl, col, desc in steps:
            box = rounded_box(2.8, 1.0,
                              fill_color=col + "22", stroke_color=col,
                              label=lbl, label_color=col)
            desc_txt = label_text(desc, color=GREY_LIGHT)
            desc_txt.next_to(box, DOWN, buff=0.2)
            step_boxes.add(VGroup(box, desc_txt))

        step_boxes.arrange(RIGHT, buff=0.9)
        step_boxes.move_to(ORIGIN + DOWN * 0.2)

        spec_arrows = VGroup()
        for i in range(len(step_boxes) - 1):
            arr = Arrow(step_boxes[i][0].get_right(), step_boxes[i + 1][0].get_left(),
                        color=GREY_MED, buff=0.05, stroke_width=1.5,
                        max_tip_length_to_length_ratio=0.2)
            spec_arrows.add(arr)

        self.play(LaggedStart(*[FadeIn(b) for b in step_boxes], lag_ratio=0.3),
                  run_time=1.2)
        self.play(LaggedStart(*[GrowArrow(a) for a in spec_arrows], lag_ratio=0.3),
                  run_time=0.7)

        note = label_text("Get large-model quality at small-model speed!", color=GREEN_LIGHT)
        note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(note), run_time=0.6)
        self.wait(2)
