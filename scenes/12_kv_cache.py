"""
Scene 12 — KV Cache
Run: manim -pql 12_kv_cache.py KVCacheScene
"""

from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class KVCacheScene(LLMScene):
    def construct(self):
        title = self.show_title("KV Cache", "Memory for Efficiency")
        self.wait(0.5)
        self.fade_all(title)

        # ── 1. Without cache — recompute everything ────────────────────────────
        no_cache_title = body_text("Without KV Cache — recompute all tokens every step:",
                                   color=RED_MED)
        no_cache_title.to_edge(UP, buff=0.6)
        self.play(Write(no_cache_title), run_time=0.7)

        # Show a sequence of tokens, each step reprocessing all
        token_labels = ["The", "cat", "sat", "on", "the", "?"]
        token_boxes = VGroup(*[
            rounded_box(0.9, 0.55,
                        fill_color=BLUE_DARK, stroke_color=BLUE_MED,
                        label=t, label_color=BLUE_MED)
            for t in token_labels
        ])
        token_boxes.arrange(RIGHT, buff=0.2)
        token_boxes.shift(UP * 0.5)

        self.play(FadeIn(token_boxes), run_time=0.7)

        # Red re-process arrows for each step
        re_arrows = VGroup()
        for i in range(len(token_labels) - 1):
            arr = CurvedArrow(
                token_boxes[-1].get_top() + UP * 0.1,
                token_boxes[0].get_top() + UP * 0.1,
                angle=-TAU / 5, color=RED_MED, stroke_width=1.8,
            )
            re_arrows.add(arr)

        cost_label = label_text("Every new token -> re-read ALL previous tokens",
                                color=RED_MED)
        cost_label.next_to(token_boxes, DOWN, buff=0.5)
        big_O = body_text("Cost grows as  N squared", color=RED_MED)
        big_O.next_to(cost_label, DOWN, buff=0.25)

        self.play(Create(re_arrows[0]), run_time=0.6)
        self.play(FadeIn(cost_label), FadeIn(big_O), run_time=0.6)
        self.wait(1)
        self.fade_all(no_cache_title, token_boxes, re_arrows, cost_label, big_O)

        # ── 2. With cache — append only ───────────────────────────────────────
        cache_title = body_text("With KV Cache — store K,V; only compute new token:",
                                color=GREEN_MED)
        cache_title.to_edge(UP, buff=0.6)
        self.play(Write(cache_title), run_time=0.7)

        # Cache box growing
        cache_box = rounded_box(5.5, 1.0,
                                fill_color=GREEN_DARK, stroke_color=GREEN_MED,
                                label="KV Cache  [K0,V0]  [K1,V1]  [K2,V2]  [K3,V3]",
                                label_color=GREEN_LIGHT)
        cache_box.shift(UP * 0.8)

        new_token_box = rounded_box(1.2, 0.6,
                                    fill_color=str(YELLOW_MED) + "33",
                                    stroke_color=YELLOW_MED,
                                    label="New Q", label_color=YELLOW_MED)
        new_token_box.shift(DOWN * 0.5 + LEFT * 3)

        attend_arrow = Arrow(new_token_box.get_top(),
                             cache_box.get_bottom() + LEFT * 1.0,
                             color=YELLOW_MED, buff=0.05, stroke_width=2)
        attend_label = label_text("Attend to cache", color=YELLOW_MED)
        attend_label.next_to(attend_arrow, RIGHT, buff=0.1)

        output_box = rounded_box(1.4, 0.6,
                                 fill_color=BLUE_DARK, stroke_color=BLUE_MED,
                                 label="Output token", label_color=BLUE_MED)
        output_box.shift(DOWN * 0.5 + RIGHT * 3)

        append_arrow = Arrow(output_box.get_top(),
                             cache_box.get_bottom() + RIGHT * 1.0,
                             color=GREEN_MED, buff=0.05, stroke_width=2)
        append_label = label_text("Append K,V to cache", color=GREEN_MED)
        append_label.next_to(append_arrow, RIGHT, buff=0.1)

        cost2 = body_text("Cost per step stays O(N)  —  huge speedup!", color=GREEN_MED)
        cost2.to_edge(DOWN, buff=0.4)

        self.play(FadeIn(cache_box), run_time=0.6)
        self.play(FadeIn(new_token_box), Create(attend_arrow),
                  FadeIn(attend_label), run_time=0.8)
        self.play(FadeIn(output_box), Create(append_arrow),
                  FadeIn(append_label), run_time=0.8)
        self.play(Write(cost2), run_time=0.6)
        self.wait(1.2)
        self.fade_all(cache_title, cache_box, new_token_box, attend_arrow,
                      attend_label, output_box, append_arrow, append_label, cost2)

        # ── 3. Cache size and GQA ──────────────────────────────────────────────
        gqa_title = body_text("Reducing cache size with GQA:", color=WHITE)
        gqa_title.to_edge(UP, buff=0.6)
        self.play(Write(gqa_title), run_time=0.6)

        rows_data = [
            ("Full MHA",   BLUE_MED,   "32 Q heads, 32 K/V heads", "1x cache"),
            ("GQA",        GREEN_MED,  "32 Q heads,  8 K/V heads", "4x smaller cache"),
            ("MQA",        ORANGE_MED, "32 Q heads,  1 K/V pair",  "32x smaller cache"),
        ]
        rows = VGroup()
        for name, col, desc, size in rows_data:
            n = body_text(name, color=col)
            d = label_text(desc, color=WHITE)
            s = label_text(size, color=col)
            row = VGroup(n, d, s)
            row.arrange(RIGHT, buff=0.5)
            rows.add(row)

        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        rows.move_to(ORIGIN + DOWN * 0.2)
        box = SurroundingRectangle(rows, color=BLUE_MED, buff=0.3, corner_radius=0.15)
        self.play(Create(box), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.25),
                  run_time=1.2)

        note = label_text("LLaMA 3 8B uses GQA: 32 query heads, 8 KV heads",
                           color=GREY_LIGHT)
        note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(2)
        self.fade_all(gqa_title, rows, box, note)

        # ── 4. Token generation without KV cache — recomputation cost ─────────
        regen_title = body_text("Without cache: generating token 100 repeats 99 computations:",
                                color=RED_MED)
        regen_title.to_edge(UP, buff=0.6)
        self.play(Write(regen_title), run_time=0.7)

        steps_viz = VGroup()
        step_labels = ["Gen t=1\n1 pass", "Gen t=2\n2 passes", "Gen t=5\n5 passes",
                       "Gen t=10\n10 passes", "Gen t=100\n100 passes"]
        step_counts = [1, 2, 5, 10, 100]
        max_count = 100

        for i, (lbl, cnt) in enumerate(zip(step_labels, step_counts)):
            bar_h = (cnt / max_count) * 3.5
            bar = Rectangle(width=1.4, height=max(bar_h, 0.05),
                            fill_color=str(RED_MED) + "66", fill_opacity=1,
                            stroke_color=RED_MED, stroke_width=1.5)
            cnt_lbl = label_text(lbl, color=RED_MED)
            cnt_lbl.next_to(bar, DOWN, buff=0.1)
            group = VGroup(bar, cnt_lbl)
            group.shift(RIGHT * i * 2.2)
            steps_viz.add(group)

        steps_viz.move_to(ORIGIN + DOWN * 0.0)

        total_note = label_text(
            "Total work = 1+2+...+N = N*(N+1)/2 — grows as N squared",
            color=RED_MED,
        )
        total_note.to_edge(DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(s) for s in steps_viz], lag_ratio=0.15),
                  run_time=1.2)
        self.play(FadeIn(total_note), run_time=0.6)
        self.wait(1.5)
        self.fade_all(regen_title, steps_viz, total_note)

        # ── 5. Cache growth over time ─────────────────────────────────────────
        growth_title = body_text("KV cache grows one entry per generated token:", color=WHITE)
        growth_title.to_edge(UP, buff=0.6)
        self.play(Write(growth_title), run_time=0.7)

        cache_steps = [1, 10, 25, 50, 100]
        max_cache = 100
        cache_bars = VGroup()

        for i, tok in enumerate(cache_steps):
            bar_h = (tok / max_cache) * 3.5
            bar = Rectangle(width=1.6, height=max(bar_h, 0.05),
                            fill_color=str(GREEN_MED) + "66", fill_opacity=1,
                            stroke_color=GREEN_MED, stroke_width=1.5)
            top_lbl = label_text(f"t={tok}\n{tok} KV pairs", color=GREEN_MED)
            top_lbl.next_to(bar, UP, buff=0.1)
            group = VGroup(bar, top_lbl)
            group.shift(RIGHT * i * 2.4)
            cache_bars.add(group)

        cache_bars.move_to(ORIGIN + DOWN * 0.1)

        linear_note = label_text(
            "Cache grows linearly — but each step only adds 1 new entry",
            color=GREEN_MED,
        )
        linear_note.to_edge(DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(b) for b in cache_bars], lag_ratio=0.15),
                  run_time=1.2)
        self.play(FadeIn(linear_note), run_time=0.6)
        self.wait(1.5)
        self.fade_all(growth_title, cache_bars, linear_note)

        # ── 6. Memory cost breakdown ───────────────────────────────────────────
        mem_title = body_text("Memory cost: layers x heads x tokens x dim:", color=WHITE)
        mem_title.to_edge(UP, buff=0.6)
        self.play(Write(mem_title), run_time=0.7)

        mem_factors = [
            ("2 (K+V)", WHITE, "Keys and Values"),
            ("x 32", BLUE_MED, "Layers"),
            ("x 8", GREEN_MED, "KV heads (GQA)"),
            ("x 4096", YELLOW_MED, "Tokens (4k context)"),
            ("x 128", ORANGE_MED, "Head dimension"),
            ("x 2 bytes", RED_MED, "fp16 precision"),
        ]

        factor_items = VGroup()
        for val, col, note in mem_factors:
            val_txt = body_text(val, color=col)
            note_txt = label_text(f"  ({note})", color=GREY_LIGHT)
            row = VGroup(val_txt, note_txt)
            row.arrange(RIGHT, buff=0.1)
            factor_items.add(row)

        factor_items.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        factor_items.shift(LEFT * 2.0 + UP * 0.5)

        result_box = rounded_box(4.5, 1.2,
                                 fill_color=str(GREEN_MED) + "22",
                                 stroke_color=GREEN_MED,
                                 label="= ~537 MB for 4k tokens", label_color=GREEN_MED)
        result_box.shift(RIGHT * 3.0 + UP * 0.5)

        scale_lbl = label_text(
            "At 128k tokens (max context): ~16 GB just for KV cache!",
            color=RED_MED,
        )
        scale_lbl.to_edge(DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(f) for f in factor_items], lag_ratio=0.12),
                  run_time=1.2)
        self.play(FadeIn(result_box), run_time=0.6)
        self.play(FadeIn(scale_lbl), run_time=0.6)
        self.wait(1.5)
        self.fade_all(mem_title, factor_items, result_box, scale_lbl)

        # ── 7. Multi-Query Attention — sharing KV ─────────────────────────────
        mqa_title = body_text("MQA: all query heads share ONE K and ONE V:", color=WHITE)
        mqa_title.to_edge(UP, buff=0.6)
        self.play(Write(mqa_title), run_time=0.7)

        # MHA side
        mha_q_heads = VGroup()
        mha_kv_heads = VGroup()
        for i in range(4):
            q = rounded_box(0.9, 0.45,
                            fill_color=str(BLUE_MED) + "44",
                            stroke_color=BLUE_MED,
                            label=f"Q{i+1}", label_color=BLUE_MED)
            mha_q_heads.add(q)
            kv = rounded_box(0.9, 0.45,
                             fill_color=str(GREEN_MED) + "44",
                             stroke_color=GREEN_MED,
                             label=f"KV{i+1}", label_color=GREEN_MED)
            mha_kv_heads.add(kv)

        mha_q_heads.arrange(DOWN, buff=0.1)
        mha_kv_heads.arrange(DOWN, buff=0.1)
        mha_q_heads.shift(LEFT * 4.5 + UP * 0.3)
        mha_kv_heads.shift(LEFT * 2.5 + UP * 0.3)

        mha_arrows = VGroup()
        for qh, kvh in zip(mha_q_heads, mha_kv_heads):
            arr = Arrow(qh.get_right(), kvh.get_left(),
                        color=GREY_MED, buff=0.05, stroke_width=1.2,
                        max_tip_length_to_length_ratio=0.2)
            mha_arrows.add(arr)

        mha_lbl = label_text("MHA: 4 heads,\n4 KV pairs", color=BLUE_MED)
        mha_lbl.next_to(mha_kv_heads, DOWN, buff=0.2)

        # MQA side
        mqa_q_heads = VGroup()
        for i in range(4):
            q = rounded_box(0.9, 0.45,
                            fill_color=str(BLUE_MED) + "44",
                            stroke_color=BLUE_MED,
                            label=f"Q{i+1}", label_color=BLUE_MED)
            mqa_q_heads.add(q)

        shared_kv = rounded_box(1.0, 1.8,
                                fill_color=str(ORANGE_MED) + "44",
                                stroke_color=ORANGE_MED,
                                label="Shared\nK, V", label_color=ORANGE_MED)

        mqa_q_heads.arrange(DOWN, buff=0.1)
        mqa_q_heads.shift(RIGHT * 1.5 + UP * 0.3)
        shared_kv.shift(RIGHT * 3.8 + UP * 0.3)

        mqa_arrows = VGroup()
        for qh in mqa_q_heads:
            arr = Arrow(qh.get_right(), shared_kv.get_left(),
                        color=ORANGE_MED, buff=0.05, stroke_width=1.2,
                        max_tip_length_to_length_ratio=0.2)
            mqa_arrows.add(arr)

        mqa_lbl = label_text("MQA: 4 heads,\n1 KV pair (4x smaller!)", color=ORANGE_MED)
        mqa_lbl.next_to(shared_kv, DOWN, buff=0.2)

        vs_lbl = body_text("vs", color=GREY_LIGHT)
        vs_lbl.move_to(ORIGIN + UP * 0.3)

        self.play(FadeIn(mha_q_heads), FadeIn(mha_kv_heads), run_time=0.5)
        self.play(LaggedStart(*[Create(a) for a in mha_arrows], lag_ratio=0.1), run_time=0.6)
        self.play(FadeIn(mha_lbl), run_time=0.4)
        self.play(FadeIn(vs_lbl), run_time=0.3)
        self.play(FadeIn(mqa_q_heads), FadeIn(shared_kv), run_time=0.5)
        self.play(LaggedStart(*[Create(a) for a in mqa_arrows], lag_ratio=0.1), run_time=0.6)
        self.play(FadeIn(mqa_lbl), run_time=0.4)
        self.wait(1.5)
        self.fade_all(mqa_title, mha_q_heads, mha_kv_heads, mha_arrows, mha_lbl,
                      vs_lbl, mqa_q_heads, shared_kv, mqa_arrows, mqa_lbl)

        # ── 8. Paged attention ────────────────────────────────────────────────
        paged_title = body_text("Paged Attention — KV cache as non-contiguous pages:", color=WHITE)
        paged_title.to_edge(UP, buff=0.6)
        self.play(Write(paged_title), run_time=0.7)

        # Show memory as a grid of pages
        total_pages = 18
        seq_a_pages = [0, 1, 2, 5, 8]
        seq_b_pages = [3, 7, 10, 11]
        seq_c_pages = [4, 6, 9]

        page_grid = VGroup()
        page_cols = 6
        for i in range(total_pages):
            row = i // page_cols
            col = i % page_cols
            if i in seq_a_pages:
                col_fill = str(BLUE_MED) + "88"
                stroke = BLUE_MED
            elif i in seq_b_pages:
                col_fill = str(GREEN_MED) + "88"
                stroke = GREEN_MED
            elif i in seq_c_pages:
                col_fill = str(ORANGE_MED) + "88"
                stroke = ORANGE_MED
            else:
                col_fill = str(GREY_MED) + "33"
                stroke = GREY_MED

            page = Square(side_length=0.75,
                          fill_color=col_fill, fill_opacity=1,
                          stroke_color=stroke, stroke_width=1.5)
            page_num = label_text(str(i), color=WHITE)
            page_num.move_to(page)
            page_cell = VGroup(page, page_num)
            page_cell.move_to([col * 0.85 - 2.1, -row * 0.85 + 0.7, 0])
            page_grid.add(page_cell)

        legend_a = rounded_box(1.6, 0.4,
                               fill_color=str(BLUE_MED) + "88",
                               stroke_color=BLUE_MED,
                               label="User A seq", label_color=WHITE)
        legend_b = rounded_box(1.6, 0.4,
                               fill_color=str(GREEN_MED) + "88",
                               stroke_color=GREEN_MED,
                               label="User B seq", label_color=WHITE)
        legend_c = rounded_box(1.6, 0.4,
                               fill_color=str(ORANGE_MED) + "88",
                               stroke_color=ORANGE_MED,
                               label="User C seq", label_color=WHITE)

        legend_a.shift(RIGHT * 4.0 + UP * 1.2)
        legend_b.shift(RIGHT * 4.0 + UP * 0.5)
        legend_c.shift(RIGHT * 4.0 + DOWN * 0.2)

        paged_note = label_text(
            "Pages are non-contiguous but efficient — no wasted memory.\n"
            "Used in vLLM for high-throughput serving.",
            color=GREY_LIGHT,
        )
        paged_note.to_edge(DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(p) for p in page_grid], lag_ratio=0.04),
                  run_time=1.2)
        self.play(FadeIn(legend_a), FadeIn(legend_b), FadeIn(legend_c), run_time=0.6)
        self.play(FadeIn(paged_note), run_time=0.6)
        self.wait(2)
        self.fade_all(paged_title, page_grid, legend_a, legend_b, legend_c, paged_note)
