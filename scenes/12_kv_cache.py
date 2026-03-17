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
            # arrow sweeping back over all previous tokens
            arr = CurvedArrow(
                token_boxes[-1].get_top() + UP * 0.1,
                token_boxes[0].get_top() + UP * 0.1,
                angle=-TAU / 5, color=RED_MED, stroke_width=1.8,
            )
            re_arrows.add(arr)

        cost_label = label_text("Every new token → re-read ALL previous tokens",
                                color=RED_MED)
        cost_label.next_to(token_boxes, DOWN, buff=0.5)
        big_O = body_text("Cost grows as  N²", color=RED_MED)
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
                                label="KV Cache  [K₀,V₀]  [K₁,V₁]  [K₂,V₂]  [K₃,V₃]",
                                label_color=GREEN_LIGHT)
        cache_box.shift(UP * 0.8)

        new_token_box = rounded_box(1.2, 0.6,
                                    fill_color=YELLOW_MED + "33",
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
        self.play(FadeIn(new_token_box), GrowArrow(attend_arrow),
                  FadeIn(attend_label), run_time=0.8)
        self.play(FadeIn(output_box), GrowArrow(append_arrow),
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
            ("Full MHA",   BLUE_MED,   "32 Q heads, 32 K/V heads", "1× cache"),
            ("GQA",        GREEN_MED,  "32 Q heads,  8 K/V heads", "4× smaller cache"),
            ("MQA",        ORANGE_MED, "32 Q heads,  1 K/V pair",  "32× smaller cache"),
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
