"""
Scene 09 — Multi-Head Attention
Run: manim -pql 09_multi_head_attention.py MultiHeadAttentionScene
"""

from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class MultiHeadAttentionScene(LLMScene):
    def construct(self):
        title = self.show_title("Multi-Head Attention", "Multiple Perspectives")
        self.wait(0.5)
        self.fade_all(title)

        # ── 1. Show single-head vs multi-head ─────────────────────────────────
        sh_box = rounded_box(3.5, 1.2,
                             fill_color=BLUE_DARK, stroke_color=BLUE_MED,
                             label="Single-Head Attention\n(one perspective)",
                             label_color=BLUE_MED)
        sh_box.shift(LEFT * 3)

        mh_label = body_text("Multi-Head Attention\n(many perspectives)", color=PURPLE_MED)
        heads_row = VGroup(*[
            rounded_box(0.8, 0.7,
                        fill_color=col + "33", stroke_color=col,
                        label=f"H{i+1}", label_color=col)
            for i, col in enumerate([
                BLUE_MED, GREEN_MED, ORANGE_MED, RED_MED,
                PURPLE_MED, YELLOW_MED, BLUE_LIGHT, GREEN_LIGHT,
            ])
        ])
        heads_row.arrange(RIGHT, buff=0.15)
        mh_label.next_to(heads_row, UP, buff=0.25)
        mh_group = VGroup(mh_label, heads_row)
        mh_group.shift(RIGHT * 1.5)

        vs_text = body_text("vs.", color=GREY_LIGHT)
        vs_text.move_to(ORIGIN + LEFT * 0.3)

        self.play(FadeIn(sh_box), run_time=0.6)
        self.play(Write(vs_text), run_time=0.3)
        self.play(FadeIn(mh_label),
                  LaggedStart(*[FadeIn(h) for h in heads_row], lag_ratio=0.08),
                  run_time=1.0)
        self.wait(1)

        # ── 2. Expert panel analogy ────────────────────────────────────────────
        self.fade_all(sh_box, vs_text, mh_group)

        panel_title = body_text("Like an expert panel — each head specialises:", color=WHITE)
        panel_title.to_edge(UP, buff=0.6)
        self.play(Write(panel_title), run_time=0.7)

        specialisations = [
            ("Head 1", BLUE_MED,    "Subject–verb agreement"),
            ("Head 2", GREEN_MED,   "Pronoun coreference"),
            ("Head 3", ORANGE_MED,  "Syntactic structure"),
            ("Head 4", RED_MED,     "Semantic similarity"),
            ("Head 5", PURPLE_MED,  "Temporal markers"),
            ("Head 6", YELLOW_MED,  "Contrast / 'but', 'however'"),
        ]

        rows = VGroup()
        for head, col, spec in specialisations:
            h_lbl = body_text(head, color=col)
            s_lbl = label_text(spec, color=WHITE)
            s_lbl.next_to(h_lbl, RIGHT, buff=0.4)
            rows.add(VGroup(h_lbl, s_lbl))

        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        rows.move_to(ORIGIN + DOWN * 0.2)
        box = SurroundingRectangle(rows, color=GREY_MED, buff=0.3, corner_radius=0.12)

        self.play(Create(box), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.15),
                  run_time=1.4)
        self.wait(1.2)
        self.fade_all(panel_title, rows, box)

        # ── 3. Architecture: project → heads → concat → project ───────────────
        arch_title = body_text("How multi-head attention is computed:", color=WHITE)
        arch_title.to_edge(UP, buff=0.6)
        self.play(Write(arch_title), run_time=0.6)

        steps = [
            ("Input\nvector",       WHITE,      4.5),
            ("Project Q,K,V\nper head", BLUE_MED, 1.5),
            ("Run attention\nindependently", PURPLE_MED, 1.5),
            ("Concatenate\nall outputs", GREEN_MED, 1.5),
            ("Final linear\nprojection", ORANGE_MED, 1.5),
            ("Output\nvector",      WHITE,      4.5),
        ]

        step_boxes = VGroup()
        for lbl, col, w in steps:
            b = rounded_box(w * 0.35 + 0.5, 0.8,
                            fill_color=col + "22",
                            stroke_color=col,
                            label=lbl, label_color=col)
            step_boxes.add(b)

        step_boxes.arrange(RIGHT, buff=0.4)
        step_boxes.scale_to_fit_width(13)
        step_boxes.move_to(ORIGIN + DOWN * 0.2)

        step_arrows = VGroup()
        for i in range(len(step_boxes) - 1):
            arr = Arrow(step_boxes[i].get_right(), step_boxes[i + 1].get_left(),
                        color=GREY_MED, buff=0.05, stroke_width=1.5,
                        max_tip_length_to_length_ratio=0.18)
            step_arrows.add(arr)

        self.play(LaggedStart(*[FadeIn(b) for b in step_boxes], lag_ratio=0.12),
                  run_time=1.4)
        self.play(LaggedStart(*[GrowArrow(a) for a in step_arrows], lag_ratio=0.1),
                  run_time=0.9)
        self.wait(0.8)

        # ── 4. GQA callout ────────────────────────────────────────────────────
        gqa_note = body_text(
            "Modern LLMs use GQA: share K,V across groups of heads\n"
            "→ Smaller KV Cache, nearly same quality  (used in LLaMA 3)",
            color=GREY_LIGHT,
        )
        gqa_note.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(gqa_note), run_time=0.8)
        self.wait(2)
