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

        # ── 3. Top-k visual ───────────────────────────────────────────────────
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
        self.wait(0.8)
        self.fade_all(topk_title, all_bars, k_line, k_lbl)

        # ── 4. Top-p (nucleus) visual ─────────────────────────────────────────
        topp_title = body_text("Top-p (Nucleus) — keep tokens until cumulative prob ≥ p",
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

        # Highlight nucleus (first nucleus_size bars)
        nucleus_rect = SurroundingRectangle(
            VGroup(*[p_bars[i] for i in range(nucleus_size)]),
            color=GREEN_MED, buff=0.15, corner_radius=0.1, stroke_width=2,
        )
        p_lbl = label_text(f"Nucleus: top {nucleus_size} tokens cover ≥90% probability",
                           color=GREEN_MED)
        p_lbl.to_edge(DOWN, buff=0.4)

        self.play(Create(nucleus_rect), FadeIn(p_lbl), run_time=0.7)
        self.wait(1.2)
        self.fade_all(topp_title, p_bars, nucleus_rect, p_lbl)

        # ── 5. Summary presets ────────────────────────────────────────────────
        preset_title = body_text("Common presets:", color=WHITE)
        preset_title.to_edge(UP, buff=0.6)
        self.play(Write(preset_title), run_time=0.5)

        presets = [
            ("Code generation",  "T=0.2,  top-p=0.95,  top-k=40",  BLUE_MED),
            ("General chat",     "T=0.7,  top-p=0.9,   top-k=50",  GREEN_MED),
            ("Creative writing", "T=0.9,  top-p=0.95,  top-k=off", ORANGE_MED),
            ("Brainstorming",    "T=1.2,  top-p=0.99,  top-k=off", RED_MED),
        ]
        preset_rows = VGroup()
        for name, setting, col in presets:
            n = body_text(name, color=col)
            s = label_text(setting, color=WHITE)
            s.next_to(n, RIGHT, buff=0.4)
            preset_rows.add(VGroup(n, s))

        preset_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        preset_rows.move_to(ORIGIN + DOWN * 0.2)
        box = SurroundingRectangle(preset_rows, color=GREY_MED,
                                   buff=0.3, corner_radius=0.12)
        self.play(Create(box), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r) for r in preset_rows], lag_ratio=0.2),
                  run_time=1.2)
        self.wait(2)
