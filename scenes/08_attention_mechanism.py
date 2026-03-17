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
            bg  = SurroundingRectangle(box, color=str(col) + "55",
                                        fill_color=str(col) + "11",
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
        self.fade_all(blend_title, blend_eq, result_note)

        # ── 5. QKV Library Analogy — extended ─────────────────────────────────
        library_title = body_text("The Library Analogy", color=WHITE)
        library_title.to_edge(UP, buff=0.6)
        self.play(Write(library_title), run_time=0.6)

        query_box = rounded_box(3.8, 0.8,
                                fill_color=str(BLUE_MED) + "22",
                                stroke_color=BLUE_MED,
                                label="Query: \"Books about space exploration\"",
                                label_color=WHITE)
        query_box.shift(UP * 1.8)

        books = [
            ("Apollo 13\n[Key: Space, NASA]",   GREEN_MED,  0.9),
            ("Rocket Propulsion\n[Key: Space]",  ORANGE_MED, 0.5),
            ("French Cooking\n[Key: Food]",       RED_MED,   0.05),
        ]

        book_boxes = VGroup()
        score_lbls = VGroup()
        for name, col, score in books:
            b = rounded_box(2.6, 0.75,
                            fill_color=str(col) + "22",
                            stroke_color=col, label=name, label_color=WHITE)
            s = label_text(f"Match score: {score}", color=col)
            s.next_to(b, DOWN, buff=0.1)
            book_boxes.add(VGroup(b, s))

        book_boxes.arrange(RIGHT, buff=0.5)
        book_boxes.shift(DOWN * 0.5)

        arr_q1 = Arrow(query_box.get_bottom(), book_boxes[0].get_top(),
                       color=GREEN_MED, buff=0.06, stroke_width=1.5,
                       max_tip_length_to_length_ratio=0.2)
        arr_q2 = Arrow(query_box.get_bottom(), book_boxes[1].get_top(),
                       color=ORANGE_MED, buff=0.06, stroke_width=1.5,
                       max_tip_length_to_length_ratio=0.2)
        arr_q3 = Arrow(query_box.get_bottom(), book_boxes[2].get_top(),
                       color=RED_MED, buff=0.06, stroke_width=1.5,
                       max_tip_length_to_length_ratio=0.2)

        self.play(FadeIn(query_box), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(b) for b in book_boxes], lag_ratio=0.2),
                  run_time=0.9)
        self.play(Create(arr_q1), Create(arr_q2), Create(arr_q3), run_time=0.7)

        blend_note = label_text(
            "Value = actual book content — blended by match score (attention weight)",
            color=YELLOW_MED,
        )
        blend_note.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(blend_note), run_time=0.6)
        self.wait(2)
        self.fade_all(library_title, query_box, book_boxes,
                      arr_q1, arr_q2, arr_q3, blend_note)

        # ── 6. Computing attention scores step by step — 4-token grid ─────────
        step_title = body_text("Attention Scores — Step by Step", color=WHITE)
        step_title.to_edge(UP, buff=0.6)
        self.play(Write(step_title), run_time=0.6)

        step_tokens = ["Cats", "chase", "mice", "."]
        # Illustrative dot-product scores (Q·K^T)
        raw_scores = [
            [3.2, 1.1, 1.5, 0.2],
            [2.1, 3.8, 2.9, 0.1],
            [1.5, 0.9, 3.5, 0.3],
            [0.4, 0.3, 0.5, 3.1],
        ]

        # Build 4×4 grid of score boxes
        cell_sz = 0.9
        score_grid = VGroup()
        for i, row in enumerate(raw_scores):
            for j, val in enumerate(row):
                intensity = val / 4.0
                cell = Square(side_length=cell_sz,
                              fill_color=BLUE_MED, fill_opacity=intensity,
                              stroke_color=GREY_MED, stroke_width=0.8)
                cell.move_to([j * cell_sz, -i * cell_sz, 0])
                val_txt = label_text(f"{val:.1f}", color=WHITE)
                val_txt.move_to(cell)
                score_grid.add(VGroup(cell, val_txt))

        # Row / col labels
        for i, tok in enumerate(step_tokens):
            lbl = label_text(tok, color=GREY_LIGHT)
            lbl.next_to(score_grid[i * 4][0], LEFT, buff=0.25)
            score_grid.add(lbl)
        for j, tok in enumerate(step_tokens):
            lbl = label_text(tok, color=GREY_LIGHT)
            lbl.next_to(score_grid[j][0], UP, buff=0.25)
            score_grid.add(lbl)

        score_grid.move_to(ORIGIN + LEFT * 0.5)
        score_grid.scale_to_fit_height(4.0)

        step_note = label_text(
            "Each cell = Q[row] · K[col]   (dot product)\nDiagonal = token attends to itself",
            color=GREY_LIGHT,
        )
        step_note.next_to(score_grid, RIGHT, buff=0.5)

        self.play(FadeIn(score_grid), run_time=1.2)
        self.play(FadeIn(step_note), run_time=0.6)
        self.wait(2)
        self.fade_all(step_title, score_grid, step_note)

        # ── 7. The scaling factor — before and after ───────────────────────────
        scale_title = body_text("Why Divide by sqrt(d_k)?", color=WHITE)
        scale_title.to_edge(UP, buff=0.6)
        self.play(Write(scale_title), run_time=0.6)

        before_lbl = body_text("WITHOUT scaling", color=RED_MED)
        before_lbl.shift(LEFT * 3.2 + UP * 1.6)

        after_lbl = body_text("WITH scaling  (÷ sqrt(64))", color=GREEN_MED)
        after_lbl.shift(RIGHT * 2.2 + UP * 1.6)

        # Peaked bars (before)
        before_bars_data = [0.003, 0.008, 0.97, 0.012, 0.007]
        before_toks = ["t1", "t2", "t3", "t4", "t5"]
        before_bars = VGroup()
        for i, (tok, p) in enumerate(zip(before_toks, before_bars_data)):
            h = 3.0 * p
            bar = Rectangle(width=0.5, height=max(h, 0.02),
                            fill_color=RED_MED, fill_opacity=0.85,
                            stroke_color=WHITE, stroke_width=1)
            pct = label_text(f"{p*100:.0f}%", color=WHITE)
            pct.next_to(bar, UP, buff=0.08)
            tok_l = label_text(tok, color=GREY_LIGHT)
            tok_l.next_to(bar, DOWN, buff=0.12)
            g = VGroup(bar, pct, tok_l)
            g.shift(RIGHT * i * 0.75)
            before_bars.add(g)
        before_bars.move_to(LEFT * 3.5 + DOWN * 0.3)
        before_bars.align_to(before_bars, DOWN)

        # Smoother bars (after)
        after_bars_data = [0.08, 0.12, 0.55, 0.15, 0.10]
        after_bars = VGroup()
        for i, (tok, p) in enumerate(zip(before_toks, after_bars_data)):
            h = 3.0 * p
            bar = Rectangle(width=0.5, height=max(h, 0.02),
                            fill_color=GREEN_MED, fill_opacity=0.85,
                            stroke_color=WHITE, stroke_width=1)
            pct = label_text(f"{p*100:.0f}%", color=WHITE)
            pct.next_to(bar, UP, buff=0.08)
            tok_l = label_text(tok, color=GREY_LIGHT)
            tok_l.next_to(bar, DOWN, buff=0.12)
            g = VGroup(bar, pct, tok_l)
            g.shift(RIGHT * i * 0.75)
            after_bars.add(g)
        after_bars.move_to(RIGHT * 2.0 + DOWN * 0.3)

        vs_scale = body_text("vs.", color=GREY_MED)
        vs_scale.move_to(ORIGIN + DOWN * 0.3)

        self.play(FadeIn(before_lbl), FadeIn(after_lbl), FadeIn(vs_scale), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(b) for b in before_bars], lag_ratio=0.1),
                  run_time=0.9)
        self.play(LaggedStart(*[FadeIn(b) for b in after_bars], lag_ratio=0.1),
                  run_time=0.9)

        scale_note = label_text(
            "Large dot products → softmax spikes → one token gets 97% weight\n"
            "Dividing by sqrt(d_k) keeps scores balanced",
            color=YELLOW_MED,
        )
        scale_note.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(scale_note), run_time=0.6)
        self.wait(2)
        self.fade_all(scale_title, before_lbl, after_lbl, vs_scale,
                      before_bars, after_bars, scale_note)

        # ── 8. Softmax converts scores to weights — bar chart ─────────────────
        softmax_title = body_text("Softmax — Scores to Attention Weights", color=WHITE)
        softmax_title.to_edge(UP, buff=0.6)
        self.play(Write(softmax_title), run_time=0.6)

        raw_s = np.array([-2.0, 1.0, 4.0, 0.5])
        sm_s  = softmax(raw_s)
        tok_names = ["t1", "t2", "t3", "t4"]
        bar_colors = [BLUE_LIGHT, BLUE_MED, GREEN_MED, ORANGE_MED]

        raw_bars = VGroup()
        sm_bars  = VGroup()

        raw_lbl = body_text("Raw scores", color=GREY_LIGHT)
        raw_lbl.shift(LEFT * 3.5 + UP * 2.2)
        sm_lbl  = body_text("After softmax", color=GREY_LIGHT)
        sm_lbl.shift(RIGHT * 1.5 + UP * 2.2)

        for i, (tok, raw, sm, col) in enumerate(zip(tok_names, raw_s, sm_s, bar_colors)):
            # Raw bar (can be negative, show signed rectangle)
            raw_h = abs(raw) * 0.4
            raw_bar = Rectangle(width=0.6, height=max(raw_h, 0.08),
                                fill_color=col, fill_opacity=0.8,
                                stroke_color=WHITE, stroke_width=1)
            if raw < 0:
                raw_bar.set_fill(RED_MED, opacity=0.8)
            raw_val = label_text(f"{raw:.1f}", color=WHITE)
            raw_val.next_to(raw_bar, UP, buff=0.08)
            raw_tok = label_text(tok, color=GREY_LIGHT)
            raw_tok.next_to(raw_bar, DOWN, buff=0.12)
            rg = VGroup(raw_bar, raw_val, raw_tok)
            rg.shift(RIGHT * i * 0.95)
            raw_bars.add(rg)

            # Softmax bar
            sm_h = sm * 3.5
            sm_bar = Rectangle(width=0.6, height=max(sm_h, 0.05),
                               fill_color=col, fill_opacity=0.85,
                               stroke_color=WHITE, stroke_width=1)
            sm_val = label_text(f"{sm:.2f}", color=WHITE)
            sm_val.next_to(sm_bar, UP, buff=0.08)
            sm_tok = label_text(tok, color=GREY_LIGHT)
            sm_tok.next_to(sm_bar, DOWN, buff=0.12)
            sg = VGroup(sm_bar, sm_val, sm_tok)
            sg.shift(RIGHT * i * 0.95)
            sm_bars.add(sg)

        raw_bars.move_to(LEFT * 3.5 + DOWN * 0.4)
        sm_bars.move_to(RIGHT * 1.5 + DOWN * 0.4)

        arr_softmax = Arrow(LEFT * 0.6, RIGHT * 0.6, color=YELLOW_MED,
                            stroke_width=2.5, buff=0.0,
                            max_tip_length_to_length_ratio=0.25)
        arr_softmax.move_to(ORIGIN + DOWN * 0.4)
        softmax_fn = label_text("softmax", color=YELLOW_MED)
        softmax_fn.next_to(arr_softmax, UP, buff=0.12)

        self.play(FadeIn(raw_lbl), FadeIn(sm_lbl), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(b) for b in raw_bars], lag_ratio=0.12),
                  run_time=0.9)
        self.play(Create(arr_softmax), FadeIn(softmax_fn), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(b) for b in sm_bars], lag_ratio=0.12),
                  run_time=0.9)

        sm_note = label_text(
            "Score 4.0 dominates at 86% weight — softmax amplifies the winner",
            color=YELLOW_MED,
        )
        sm_note.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(sm_note), run_time=0.6)
        self.wait(2)
        self.fade_all(softmax_title, raw_lbl, sm_lbl, raw_bars,
                      sm_bars, arr_softmax, softmax_fn, sm_note)

        # ── 9. Causal masking in decoders ─────────────────────────────────────
        mask_title = body_text("Causal Masking — GPT Can't Peek Ahead", color=WHITE)
        mask_title.to_edge(UP, buff=0.6)
        self.play(Write(mask_title), run_time=0.6)

        mask_tokens = ["Cats", "chase", "mice", "."]
        mask_n = 4
        cell_s = 0.85

        mask_grid = VGroup()
        for i in range(mask_n):
            for j in range(mask_n):
                allowed = (j <= i)
                col = GREEN_MED if allowed else RED_MED
                op  = 0.7 if allowed else 0.3
                cell = Square(side_length=cell_s,
                              fill_color=col, fill_opacity=op,
                              stroke_color=GREY_DARK, stroke_width=0.8)
                cell.move_to([j * cell_s, -i * cell_s, 0])
                sym = label_text("✓" if allowed else "✗", color=WHITE)
                sym.move_to(cell)
                mask_grid.add(VGroup(cell, sym))

        for i, tok in enumerate(mask_tokens):
            lbl = label_text(tok, color=GREY_LIGHT)
            lbl.next_to(mask_grid[i * mask_n][0], LEFT, buff=0.3)
            mask_grid.add(lbl)
        for j, tok in enumerate(mask_tokens):
            lbl = label_text(tok, color=GREY_LIGHT)
            lbl.next_to(mask_grid[j][0], UP, buff=0.3)
            mask_grid.add(lbl)

        mask_grid.move_to(ORIGIN + LEFT * 0.8)
        mask_grid.scale(0.95)

        mask_note = label_text(
            "GPT can only look at past tokens\n"
            "Future tokens are masked to -infinity before softmax",
            color=GREY_LIGHT,
        )
        mask_note.next_to(mask_grid, RIGHT, buff=0.5)

        self.play(FadeIn(mask_grid), run_time=1.2)
        self.play(FadeIn(mask_note), run_time=0.6)
        self.wait(2)
        self.fade_all(mask_title, mask_grid, mask_note)

        # ── 10. Cross-attention — encoder and decoder ─────────────────────────
        cross_title = body_text("Cross-Attention — Encoder talks to Decoder", color=WHITE)
        cross_title.to_edge(UP, buff=0.6)
        self.play(Write(cross_title), run_time=0.6)

        enc_box = rounded_box(4.0, 1.0,
                              fill_color=str(BLUE_MED) + "22",
                              stroke_color=BLUE_MED,
                              label="Encoder Output\n(French: 'Je t'aime')",
                              label_color=WHITE)
        enc_box.shift(UP * 1.6)

        dec_box = rounded_box(4.0, 1.0,
                              fill_color=str(GREEN_MED) + "22",
                              stroke_color=GREEN_MED,
                              label="Decoder State\n(English: generating 'love')",
                              label_color=WHITE)
        dec_box.shift(DOWN * 0.8)

        kv_lbl = label_text("Keys + Values", color=BLUE_MED)
        kv_lbl.next_to(enc_box, LEFT, buff=0.4)
        q_lbl  = label_text("Query", color=GREEN_MED)
        q_lbl.next_to(dec_box, LEFT, buff=0.4)

        cross_arr = CurvedArrow(dec_box.get_top() + LEFT * 0.5,
                                enc_box.get_bottom() + LEFT * 0.5,
                                angle=TAU / 5, color=YELLOW_MED, stroke_width=2.0)
        cross_lbl = label_text("Cross-Attention:\nQuery from decoder,\nK/V from encoder",
                               color=YELLOW_MED)
        cross_lbl.next_to(cross_arr, RIGHT, buff=0.3)

        self.play(FadeIn(enc_box), FadeIn(dec_box), run_time=0.6)
        self.play(FadeIn(kv_lbl), FadeIn(q_lbl), run_time=0.4)
        self.play(Create(cross_arr), FadeIn(cross_lbl), run_time=0.8)

        cross_note = label_text(
            "Used in translation (T5, BART) and multimodal models (image → text)",
            color=GREY_LIGHT,
        )
        cross_note.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(cross_note), run_time=0.6)
        self.wait(2)
        self.fade_all(cross_title, enc_box, dec_box, kv_lbl, q_lbl,
                      cross_arr, cross_lbl, cross_note)

        # ── Closing recap ─────────────────────────────────────────────────────
        final_title = body_text("Attention — Key Ideas", color=WHITE)
        final_title.to_edge(UP, buff=0.6)
        self.play(Write(final_title), run_time=0.6)

        recap = [
            ("Q · K^T",       BLUE_MED,   "Measure relevance of every (token, token) pair"),
            ("÷ sqrt(d_k)",    ORANGE_MED, "Prevent softmax from spiking too sharply"),
            ("Softmax",        GREEN_MED,  "Convert scores to weights summing to 1"),
            ("V blend",        PURPLE_MED, "New token = weighted mix of value vectors"),
            ("Causal mask",    RED_MED,    "Decoders hide future tokens during training"),
            ("Cross-attention",YELLOW_MED, "Decoder queries encoder's K and V"),
        ]

        recap_rows = VGroup()
        for key, col, val in recap:
            k = body_text(key + ":", color=col)
            v = label_text(val, color=WHITE)
            v.next_to(k, RIGHT, buff=0.3)
            recap_rows.add(VGroup(k, v))

        recap_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        recap_rows.move_to(ORIGIN + DOWN * 0.2)

        self.play(LaggedStart(*[FadeIn(r) for r in recap_rows], lag_ratio=0.12),
                  run_time=1.8)
        self.wait(2.5)
        self.fade_all(final_title, recap_rows)
