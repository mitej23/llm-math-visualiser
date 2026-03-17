"""
Scene 08 — Attention Mechanism
Run: manim -pql 08_attention_mechanism.py AttentionMechanismScene
"""

from manim import *
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


# Simple softmax helper
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


class AttentionMechanismScene(LLMScene):
    def construct(self):
        title = self.show_title("Attention Mechanism", "Relating Words to Each Other")
        self.wait(0.5)

        # ── 1. Motivating example ─────────────────────────────────────────────
        sentence = body_text(
            '"The bank near the river is muddy."',
            color=WHITE,
        )
        sentence.next_to(title, DOWN, buff=0.5)
        self.play(Write(sentence), run_time=0.8)

        question = label_text(
            "Which 'bank'? Financial? Or riverbank?\n"
            "→ Attend to 'river' for context!",
            color=GREY_LIGHT,
        )
        question.next_to(sentence, DOWN, buff=0.35)
        self.play(FadeIn(question), run_time=0.7)
        self.wait(1)
        self.fade_all(sentence, question, title)

        # ── 2. Q / K / V illustration ─────────────────────────────────────────
        qkv_title = body_text("Each token plays three roles:", color=WHITE)
        qkv_title.to_edge(UP, buff=0.6)
        self.play(Write(qkv_title), run_time=0.6)

        roles = [
            ("Q  Query",  BLUE_MED,   '"What am I looking for?"',
             "\"bank\" asks: am I near water?"),
            ("K  Key",    GREEN_MED,  '"What do I advertise?"',
             "\"river\" says: I\'m water-related"),
            ("V  Value",  ORANGE_MED, '"What info do I carry?"',
             "\"river\" shares its full meaning"),
        ]

        role_boxes = VGroup()
        for role, col, desc, example in roles:
            header = body_text(role, color=col)
            d_txt  = label_text(desc, color=GREY_LIGHT)
            e_txt  = label_text(example, color=WHITE)
            d_txt.next_to(header, DOWN, buff=0.1)
            e_txt.next_to(d_txt, DOWN, buff=0.1)
            box = VGroup(header, d_txt, e_txt)
            bg  = SurroundingRectangle(box, color=col + "55",
                                        fill_color=col + "11",
                                        fill_opacity=1, buff=0.25,
                                        corner_radius=0.12)
            role_boxes.add(VGroup(bg, box))

        role_boxes.arrange(RIGHT, buff=0.5)
        role_boxes.shift(DOWN * 0.3)

        self.play(LaggedStart(*[FadeIn(b) for b in role_boxes], lag_ratio=0.3),
                  run_time=1.5)
        self.wait(1.2)
        self.fade_all(qkv_title, role_boxes)

        # ── 3. Attention heat-map ──────────────────────────────────────────────
        tokens = ["The", "bank", "river", "is", "muddy"]
        # Manually crafted attention scores (query=bank, row=bank)
        # Full matrix: each row = query token, each col = key token
        scores_raw = [
            [0.8, 0.3, 0.1, 0.1, 0.1],  # "The" attending
            [0.2, 0.5, 0.9, 0.1, 0.1],  # "bank" attending — strong link to "river"
            [0.1, 0.7, 0.8, 0.2, 0.3],  # "river"
            [0.1, 0.2, 0.2, 0.9, 0.3],  # "is"
            [0.1, 0.3, 0.5, 0.3, 0.9],  # "muddy"
        ]
        # normalise each row to [0,1]
        scores = [[v / max(row) for v in row] for row in scores_raw]

        attn_grid = make_attention_grid(tokens, scores, cell_size=0.72)
        attn_grid.scale_to_fit_height(4.2)
        attn_grid.move_to(ORIGIN + LEFT * 0.5)

        grid_title = label_text("Attention weight matrix\n(darker = higher attention)",
                                color=GREY_LIGHT)
        grid_title.next_to(attn_grid, RIGHT, buff=0.5)

        self.play(FadeIn(attn_grid), run_time=1.0)
        self.play(FadeIn(grid_title), run_time=0.5)
        self.wait(0.8)

        # Highlight "bank → river" cell
        # bank is row index 1, river is col index 2
        n = len(tokens)
        bank_river_cell = attn_grid[1 * n + 2]  # row 1, col 2
        highlight = SurroundingRectangle(bank_river_cell, color=YELLOW_MED,
                                          buff=0.04, stroke_width=2.5)
        hl_note = label_text('"bank" attends strongly to "river"', color=YELLOW_MED)
        hl_note.next_to(grid_title, DOWN, buff=0.4)

        self.play(Create(highlight), FadeIn(hl_note), run_time=0.7)
        self.wait(1.5)

        # ── 4. The weighted blend ──────────────────────────────────────────────
        self.fade_all(attn_grid, grid_title, highlight, hl_note)

        blend_title = body_text("Step 3 — Blend values using attention weights",
                                color=WHITE)
        blend_title.to_edge(UP, buff=0.6)
        self.play(Write(blend_title), run_time=0.7)

        blend_eq_parts = [
            ("New(bank)", WHITE),
            (" = ", GREY_LIGHT),
            ("0.62", GREEN_MED),
            (" × V(river)", GREEN_MED),
            (" + ", GREY_LIGHT),
            ("0.22", BLUE_MED),
            (" × V(bank)", BLUE_MED),
            (" + ", GREY_LIGHT),
            ("...", GREY_MED),
        ]
        blend_eq = VGroup(*[
            body_text(text, color=col) for text, col in blend_eq_parts
        ])
        blend_eq.arrange(RIGHT, buff=0.05)
        blend_eq.move_to(ORIGIN)

        self.play(LaggedStart(*[FadeIn(p) for p in blend_eq], lag_ratio=0.1),
                  run_time=1.2)

        result_note = label_text(
            '"bank" now carries information from "river"\n'
            "→ the model understands this is a riverbank",
            color=GREY_LIGHT,
        )
        result_note.next_to(blend_eq, DOWN, buff=0.5)
        self.play(FadeIn(result_note), run_time=0.7)
        self.wait(2)
