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

        self.play(Create(line), Create(user_arrow), FadeIn(user_lbl), run_time=0.6)
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
                              fill_color=str(col) + "22", stroke_color=col,
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
        self.play(LaggedStart(*[Create(a) for a in spec_arrows], lag_ratio=0.3),
                  run_time=0.7)

        note = label_text("Get large-model quality at small-model speed!", color=GREEN_LIGHT)
        note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(note), run_time=0.6)
        self.wait(1.5)
        self.fade_all(spec_title, step_boxes, spec_arrows, note)

        # ── 4. Detailed timeline: prefill + per-token decode ──────────────────
        tl2_title = body_text("Timeline: prefill once, then decode token by token", color=WHITE)
        tl2_title.to_edge(UP, buff=0.6)
        self.play(Write(tl2_title), run_time=0.7)

        # Build segments
        segments = []
        seg_labels = []
        seg_times = []

        # Prefill segment: wide
        pf_seg = Rectangle(width=3.0, height=0.7,
                            fill_color=BLUE_DARK, fill_opacity=0.9,
                            stroke_color=BLUE_MED, stroke_width=2)
        pf_seg.move_to(LEFT * 3.8 + DOWN * 0.2)
        pf_lbl = label_text("Prefill\n200ms", color=BLUE_MED)
        pf_lbl.move_to(pf_seg)
        pf_time = label_text("compute-\nbound", color=BLUE_LIGHT)
        pf_time.next_to(pf_seg, UP, buff=0.15)
        segments.append(pf_seg)
        seg_labels.append(pf_lbl)
        seg_times.append(pf_time)

        # Decode segments: narrow, each token
        token_labels = ["T1\n20ms", "T2\n20ms", "T3\n20ms", "T4\n20ms", "..."]
        token_colors = [GREEN_MED, GREEN_MED, GREEN_MED, GREEN_MED, GREY_MED]
        x_start = -2.1
        for i, (tlbl, tcol) in enumerate(zip(token_labels, token_colors)):
            w = 0.6 if i < 4 else 0.5
            seg = Rectangle(width=w, height=0.7,
                            fill_color=GREEN_DARK if i < 4 else GREY_DARK,
                            fill_opacity=0.9,
                            stroke_color=tcol, stroke_width=2)
            seg.move_to(RIGHT * (x_start + i * 0.75) + DOWN * 0.2)
            lbl = label_text(tlbl, color=tcol)
            lbl.move_to(seg)
            segments.append(seg)
            seg_labels.append(lbl)

        bw_note = label_text("memory-bandwidth-bound", color=GREEN_LIGHT)
        bw_note.next_to(segments[-1], UP, buff=0.55)
        bw_note.shift(LEFT * 1.0)

        bw_line = Line(
            segments[1].get_top() + UP * 0.05,
            segments[-1].get_top() + UP * 0.05,
            color=GREEN_LIGHT, stroke_width=1.5
        )
        bw_line.shift(UP * 0.3)

        self.play(LaggedStart(*[FadeIn(s) for s in segments], lag_ratio=0.08), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(l) for l in seg_labels], lag_ratio=0.08), run_time=0.8)
        self.play(FadeIn(pf_time), run_time=0.4)
        self.play(Create(bw_line), FadeIn(bw_note), run_time=0.5)
        self.wait(1.5)
        all_tl2 = VGroup(*segments, *seg_labels, pf_time, bw_line, bw_note)
        self.fade_all(tl2_title, all_tl2)

        # ── 5. Why prefill is fast — parallel vs sequential ───────────────────
        parallel_title = body_text("Why prefill is fast: all tokens process simultaneously",
                                   color=WHITE)
        parallel_title.to_edge(UP, buff=0.6)
        self.play(Write(parallel_title), run_time=0.7)

        # Left: prefill parallel grid
        pf_label = body_text("PREFILL", color=BLUE_MED)
        pf_label.move_to(LEFT * 3.5 + UP * 1.4)

        pf_tokens = ["The", "cat", "sat", "on", "the", "mat"]
        pf_boxes = VGroup()
        for i, tok in enumerate(pf_tokens):
            b = rounded_box(0.85, 0.55,
                            fill_color=str(BLUE_MED) + "22",
                            stroke_color=BLUE_MED,
                            label=tok, label_color=BLUE_LIGHT)
            b.move_to(LEFT * 3.5 + RIGHT * (i * 0.95 - 2.4) + DOWN * 0.1)
            pf_boxes.add(b)

        pf_arrow = Arrow(pf_boxes.get_bottom() + DOWN * 0.1,
                         pf_boxes.get_bottom() + DOWN * 0.7,
                         color=BLUE_MED, buff=0.04, stroke_width=2,
                         max_tip_length_to_length_ratio=0.25)
        pf_arrow.shift(LEFT * 3.5)
        pf_result = label_text("All processed at once!", color=BLUE_LIGHT)
        pf_result.next_to(pf_arrow, DOWN, buff=0.1)
        pf_result.shift(LEFT * 0.1)

        # Right: decode sequential arrows
        dc_label = body_text("DECODE", color=GREEN_MED)
        dc_label.move_to(RIGHT * 2.8 + UP * 1.4)

        dc_tok_boxes = VGroup()
        dc_arrows = VGroup()
        for i in range(4):
            b = rounded_box(0.9, 0.55,
                            fill_color=str(GREEN_MED) + "22",
                            stroke_color=GREEN_MED,
                            label=f"Token {i+1}", label_color=GREEN_LIGHT)
            b.move_to(RIGHT * 2.8 + DOWN * (i * 0.85 - 0.1))
            dc_tok_boxes.add(b)

        for i in range(len(dc_tok_boxes) - 1):
            a = Arrow(dc_tok_boxes[i].get_bottom(),
                      dc_tok_boxes[i + 1].get_top(),
                      color=GREEN_MED, buff=0.04, stroke_width=1.5,
                      max_tip_length_to_length_ratio=0.2)
            dc_arrows.add(a)

        dc_result = label_text("One at a time!", color=GREEN_LIGHT)
        dc_result.next_to(dc_tok_boxes[-1], DOWN, buff=0.15)

        divider = DashedLine(UP * 1.8, DOWN * 2.2, color=GREY_MED, stroke_width=1.5)

        self.play(Write(pf_label), Write(dc_label), Create(divider), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(b) for b in pf_boxes], lag_ratio=0.08), run_time=0.8)
        self.play(Create(pf_arrow), FadeIn(pf_result), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(b) for b in dc_tok_boxes], lag_ratio=0.2), run_time=0.8)
        self.play(LaggedStart(*[Create(a) for a in dc_arrows], lag_ratio=0.2), run_time=0.6)
        self.play(FadeIn(dc_result), run_time=0.4)
        self.wait(1.5)
        all_parallel = VGroup(pf_label, pf_boxes, pf_arrow, pf_result,
                              dc_label, dc_tok_boxes, dc_arrows, dc_result, divider)
        self.fade_all(parallel_title, all_parallel)

        # ── 6. GPU utilisation — memory bandwidth bottleneck ──────────────────
        gpu_title = body_text("GPU utilisation: prefill vs decode", color=WHITE)
        gpu_title.to_edge(UP, buff=0.6)
        self.play(Write(gpu_title), run_time=0.6)

        # Prefill utilisation bar (left)
        pf_util_label = body_text("Prefill", color=BLUE_MED)
        pf_util_label.move_to(LEFT * 3.5 + UP * 1.5)

        pf_util_bg = Rectangle(width=1.5, height=3.0,
                               fill_color=GREY_DARK, fill_opacity=0.9,
                               stroke_color=GREY_MED, stroke_width=1.5)
        pf_util_bg.move_to(LEFT * 3.5)

        pf_util_bar = Rectangle(width=1.5, height=2.7,  # 90% full
                                fill_color=BLUE_MED, fill_opacity=0.85,
                                stroke_width=0)
        pf_util_bar.align_to(pf_util_bg, DOWN)
        pf_util_bar.align_to(pf_util_bg, LEFT)
        pf_util_bar.shift(RIGHT * 0.01)

        pf_pct = body_text("90%", color=WHITE)
        pf_pct.move_to(pf_util_bar.get_center())
        pf_note = label_text("High utilisation\n(big matrix multiplies)", color=BLUE_LIGHT)
        pf_note.next_to(pf_util_bg, DOWN, buff=0.2)

        # Decode utilisation bar (right)
        dc_util_label = body_text("Decode", color=GREEN_MED)
        dc_util_label.move_to(RIGHT * 2.5 + UP * 1.5)

        dc_util_bg = Rectangle(width=1.5, height=3.0,
                               fill_color=GREY_DARK, fill_opacity=0.9,
                               stroke_color=GREY_MED, stroke_width=1.5)
        dc_util_bg.move_to(RIGHT * 2.5)

        dc_util_bar = Rectangle(width=1.5, height=0.3,  # 10% full
                                fill_color=GREEN_MED, fill_opacity=0.85,
                                stroke_width=0)
        dc_util_bar.align_to(dc_util_bg, DOWN)
        dc_util_bar.align_to(dc_util_bg, LEFT)
        dc_util_bar.shift(RIGHT * 0.01)

        dc_pct = label_text("10%", color=WHITE)
        dc_pct.move_to(dc_util_bar.get_center() + UP * 0.1)
        dc_note = label_text("Low utilisation!\nMemory bandwidth\nbottleneck", color=GREEN_LIGHT)
        dc_note.next_to(dc_util_bg, DOWN, buff=0.2)

        bottleneck_box = rounded_box(5.5, 0.65,
                                     fill_color=str(ORANGE_MED) + "22",
                                     stroke_color=ORANGE_MED,
                                     label="Memory Bandwidth Bottleneck — GPU reads full KV Cache each step",
                                     label_color=ORANGE_MED)
        bottleneck_box.to_edge(DOWN, buff=0.35)

        self.play(FadeIn(pf_util_label), FadeIn(dc_util_label), run_time=0.5)
        self.play(Create(pf_util_bg), Create(dc_util_bg), run_time=0.5)
        self.play(FadeIn(pf_util_bar), FadeIn(pf_pct), run_time=0.7)
        self.play(FadeIn(dc_util_bar), FadeIn(dc_pct), run_time=0.7)
        self.play(FadeIn(pf_note), FadeIn(dc_note), run_time=0.5)
        self.play(FadeIn(bottleneck_box), run_time=0.6)
        self.wait(1.8)
        all_gpu = VGroup(pf_util_label, pf_util_bg, pf_util_bar, pf_pct, pf_note,
                         dc_util_label, dc_util_bg, dc_util_bar, dc_pct, dc_note,
                         bottleneck_box)
        self.fade_all(gpu_title, all_gpu)

        # ── 7. Speculative decoding — detailed flow ───────────────────────────
        spec2_title = body_text("Speculative Decoding — 4x faster in best case", color=WHITE)
        spec2_title.to_edge(UP, buff=0.6)
        self.play(Write(spec2_title), run_time=0.6)

        # Draft tokens row
        draft_label = label_text("Draft model generates 4 tokens quickly:", color=ORANGE_MED)
        draft_label.move_to(UP * 1.4)

        draft_tokens = ["is", "a", "large", "city"]
        draft_boxes = VGroup()
        for i, tok in enumerate(draft_tokens):
            b = rounded_box(1.2, 0.6,
                            fill_color=str(ORANGE_MED) + "22",
                            stroke_color=ORANGE_MED,
                            label=tok, label_color=ORANGE_MED)
            b.move_to(UP * 0.5 + RIGHT * (i * 1.5 - 2.25))
            draft_boxes.add(b)

        # Verify label
        verify_label = label_text("Large model verifies ALL 4 in one parallel pass:", color=BLUE_MED)
        verify_label.move_to(DOWN * 0.3)

        # Accept/reject indicators
        accept_labels = ["Accept", "Accept", "Accept", "Reject"]
        accept_colors = [GREEN_MED, GREEN_MED, GREEN_MED, RED_MED]
        accept_group = VGroup()
        for i, (albl, acol) in enumerate(zip(accept_labels, accept_colors)):
            ind = rounded_box(1.2, 0.55,
                              fill_color=str(acol) + "22",
                              stroke_color=acol,
                              label=albl, label_color=acol)
            ind.move_to(DOWN * 1.0 + RIGHT * (i * 1.5 - 2.25))
            accept_group.add(ind)

        result_note = label_text('3 tokens accepted, regenerate from "Reject" onwards',
                                 color=GREY_LIGHT)
        result_note.to_edge(DOWN, buff=0.5)

        speedup = label_text("Best case: 4 accepted = ~4x speed of normal decode!",
                             color=GREEN_LIGHT)
        speedup.next_to(result_note, UP, buff=0.2)

        self.play(FadeIn(draft_label), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(b) for b in draft_boxes], lag_ratio=0.15), run_time=0.8)
        self.play(FadeIn(verify_label), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(a) for a in accept_group], lag_ratio=0.15), run_time=0.7)
        self.play(FadeIn(result_note), FadeIn(speedup), run_time=0.5)
        self.wait(1.8)
        all_spec2 = VGroup(draft_label, draft_boxes, verify_label, accept_group,
                           result_note, speedup)
        self.fade_all(spec2_title, all_spec2)

        # ── 8. Batching decode requests ───────────────────────────────────────
        batch_title = body_text("Continuous Batching — GPU handles many users at once",
                                color=WHITE)
        batch_title.to_edge(UP, buff=0.6)
        self.play(Write(batch_title), run_time=0.7)

        # 8 user sequence rows
        user_rows = VGroup()
        row_colors = [BLUE_MED, GREEN_MED, ORANGE_MED, PURPLE_MED,
                      BLUE_LIGHT, GREEN_LIGHT, YELLOW_MED, RED_MED]
        for i in range(8):
            col = row_colors[i]
            user_lbl = label_text(f"User {i+1}", color=col)
            user_lbl.move_to(LEFT * 5.0 + DOWN * (i * 0.48 - 1.65))

            seq_boxes = VGroup()
            for j in range(6):
                sq = Rectangle(width=0.38, height=0.32,
                               fill_color=str(col) + "22",
                               fill_opacity=1.0,
                               stroke_color=col, stroke_width=1.2)
                sq.move_to(LEFT * 3.2 + RIGHT * j * 0.46 + DOWN * (i * 0.48 - 1.65))
                seq_boxes.add(sq)

            user_rows.add(VGroup(user_lbl, seq_boxes))

        step_highlight = Rectangle(width=0.38, height=8 * 0.48,
                                   fill_color=str(WHITE) + "22",
                                   fill_opacity=1.0,
                                   stroke_color=WHITE, stroke_width=2)
        step_highlight.move_to(LEFT * 3.2 + DOWN * (0 * 0.46 - 1.65 + 1.68))

        step_lbl = label_text("Current\ndecode step\n(batch of 8)", color=WHITE)
        step_lbl.next_to(step_highlight, UP, buff=0.2)

        batch_note = label_text("All 8 users decoded in one GPU pass — no wasted cycles",
                                color=GREEN_LIGHT)
        batch_note.to_edge(DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(r) for r in user_rows], lag_ratio=0.08), run_time=1.2)
        self.play(FadeIn(step_highlight), FadeIn(step_lbl), run_time=0.6)
        self.play(FadeIn(batch_note), run_time=0.4)
        self.wait(1.8)
        all_batch = VGroup(user_rows, step_highlight, step_lbl, batch_note)
        self.fade_all(batch_title, all_batch)

        # ── 9. TTFT vs Throughput tradeoff ────────────────────────────────────
        ttft_title = body_text("TTFT vs Throughput — the two key metrics", color=WHITE)
        ttft_title.to_edge(UP, buff=0.6)
        self.play(Write(ttft_title), run_time=0.6)

        # Two metric boxes side by side
        ttft_box = rounded_box(4.5, 2.2,
                               fill_color=str(BLUE_MED) + "22",
                               stroke_color=BLUE_MED)
        ttft_box.move_to(LEFT * 3.1 + DOWN * 0.2)

        ttft_heading = body_text("Time to First Token", color=BLUE_MED)
        ttft_heading.move_to(ttft_box.get_top() + DOWN * 0.4)
        ttft_detail1 = label_text("= how long prefill takes", color=GREY_LIGHT)
        ttft_detail1.next_to(ttft_heading, DOWN, buff=0.15)
        ttft_detail2 = label_text("Scales with prompt length", color=GREY_LIGHT)
        ttft_detail2.next_to(ttft_detail1, DOWN, buff=0.12)
        ttft_detail3 = label_text("100 tok: ~50ms", color=BLUE_LIGHT)
        ttft_detail3.next_to(ttft_detail2, DOWN, buff=0.12)
        ttft_detail4 = label_text("100k tok: ~20s", color=ORANGE_MED)
        ttft_detail4.next_to(ttft_detail3, DOWN, buff=0.12)

        tput_box = rounded_box(4.5, 2.2,
                               fill_color=str(GREEN_MED) + "22",
                               stroke_color=GREEN_MED)
        tput_box.move_to(RIGHT * 3.1 + DOWN * 0.2)

        tput_heading = body_text("Throughput (tokens/sec)", color=GREEN_MED)
        tput_heading.move_to(tput_box.get_top() + DOWN * 0.4)
        tput_detail1 = label_text("= decode speed once started", color=GREY_LIGHT)
        tput_detail1.next_to(tput_heading, DOWN, buff=0.15)
        tput_detail2 = label_text("Scales with batch size", color=GREY_LIGHT)
        tput_detail2.next_to(tput_detail1, DOWN, buff=0.12)
        tput_detail3 = label_text("1 user: ~50 tok/s", color=GREEN_LIGHT)
        tput_detail3.next_to(tput_detail2, DOWN, buff=0.12)
        tput_detail4 = label_text("16 users: ~800 tok/s total", color=GREEN_MED)
        tput_detail4.next_to(tput_detail3, DOWN, buff=0.12)

        tradeoff = label_text(
            "Tradeoff: longer prompts hurt TTFT but once decoding starts, throughput is high",
            color=YELLOW_MED)
        tradeoff.to_edge(DOWN, buff=0.4)

        self.play(FadeIn(ttft_box), FadeIn(tput_box), run_time=0.6)
        self.play(Write(ttft_heading), Write(tput_heading), run_time=0.6)
        self.play(LaggedStart(
            FadeIn(ttft_detail1), FadeIn(ttft_detail2),
            FadeIn(ttft_detail3), FadeIn(ttft_detail4),
            FadeIn(tput_detail1), FadeIn(tput_detail2),
            FadeIn(tput_detail3), FadeIn(tput_detail4),
            lag_ratio=0.1), run_time=1.4)
        self.play(FadeIn(tradeoff), run_time=0.5)
        self.wait(2.0)
        all_ttft = VGroup(ttft_box, ttft_heading, ttft_detail1, ttft_detail2,
                          ttft_detail3, ttft_detail4,
                          tput_box, tput_heading, tput_detail1, tput_detail2,
                          tput_detail3, tput_detail4, tradeoff)
        self.fade_all(ttft_title, all_ttft)
