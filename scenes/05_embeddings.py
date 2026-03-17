"""
Scene 05 — Embeddings
Run: manim -pql 05_embeddings.py EmbeddingsScene
"""

from manim import *
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class EmbeddingsScene(LLMScene):
    def construct(self):
        title = self.show_title("Embeddings", "How Numbers Carry Meaning")
        self.wait(0.5)

        # ── 1. Token ID → vector lookup ───────────────────────────────────────
        token_id_box = rounded_box(1.5, 0.65, fill_color=GREY_DARK,
                                   stroke_color=GREEN_MED,
                                   label='"king"  → 1547', label_color=GREEN_MED)
        token_id_box.shift(LEFT * 3.5 + UP * 0.5)

        arrow = Arrow(token_id_box.get_right(),
                      token_id_box.get_right() + RIGHT * 1.2,
                      color=WHITE, stroke_width=2, buff=0.1)

        lookup_label = label_text("Embedding\nLookup Table", color=GREY_LIGHT)
        lookup_label.next_to(arrow, UP, buff=0.1)

        vec_vals = [0.21, -0.48, 0.83, 0.07, -0.31, 0.60]
        vec_display = make_vector_display(vec_vals, color=BLUE_LIGHT)
        vec_display.next_to(arrow.get_end(), RIGHT, buff=0.2)
        vec_title = label_text("512-dim vector", color=BLUE_LIGHT)
        vec_title.next_to(vec_display, UP, buff=0.15)

        dots = label_text("...", color=BLUE_LIGHT)
        dots.next_to(vec_display, DOWN, buff=0.05)

        self.play(FadeIn(token_id_box), run_time=0.6)
        self.play(Create(arrow), FadeIn(lookup_label), run_time=0.6)
        self.play(FadeIn(vec_display), FadeIn(vec_title), FadeIn(dots), run_time=0.7)
        self.wait(0.8)

        # ── 2. 2D semantic space — word clusters ──────────────────────────────
        self.fade_all(token_id_box, arrow, lookup_label,
                      vec_display, vec_title, dots)

        axes = Axes(
            x_range=[-3.5, 3.5, 1], y_range=[-2.5, 2.5, 1],
            x_length=7, y_length=5,
            axis_config={"color": GREY_MED, "include_numbers": False},
            tips=False,
        )
        axes.shift(DOWN * 0.2)

        x_lbl = label_text("← meaning dimension 1 →", color=GREY_MED)
        x_lbl.next_to(axes, DOWN, buff=0.2)
        y_lbl = label_text("dim 2", color=GREY_MED)
        y_lbl.next_to(axes, LEFT, buff=0.2)

        # Word positions in 2D
        words = {
            "king":    (1.8,  1.5,  YELLOW_MED),
            "queen":   (1.5,  0.9,  YELLOW_MED),
            "prince":  (2.1,  0.3,  YELLOW_MED),
            "dog":     (-1.5, 0.8,  GREEN_MED),
            "cat":     (-1.8, 0.4,  GREEN_MED),
            "puppy":   (-1.2, -0.2, GREEN_MED),
            "Paris":   (-0.2, -1.5, ORANGE_MED),
            "London":  ( 0.6, -1.8, ORANGE_MED),
            "Berlin":  (-0.8, -2.0, ORANGE_MED),
        }

        self.play(Create(axes), FadeIn(x_lbl), FadeIn(y_lbl), run_time=0.8)

        dot_group = VGroup()
        for word, (x, y, col) in words.items():
            dot = Dot(axes.c2p(x, y), color=col, radius=0.10)
            lbl = label_text(word, color=col)
            lbl.next_to(dot, UR, buff=0.05)
            dot_group.add(VGroup(dot, lbl))

        self.play(LaggedStart(*[FadeIn(d) for d in dot_group], lag_ratio=0.1),
                  run_time=1.5)
        self.wait(0.8)

        # Draw cluster circles
        royal_ellipse = Ellipse(width=3.0, height=2.5, color=YELLOW_MED,
                                stroke_width=1.5, fill_opacity=0.06,
                                fill_color=YELLOW_MED)
        royal_ellipse.move_to(axes.c2p(1.8, 0.9))

        animal_ellipse = Ellipse(width=2.2, height=2.0, color=GREEN_MED,
                                 stroke_width=1.5, fill_opacity=0.06,
                                 fill_color=GREEN_MED)
        animal_ellipse.move_to(axes.c2p(-1.5, 0.3))

        city_ellipse = Ellipse(width=2.8, height=1.5, color=ORANGE_MED,
                               stroke_width=1.5, fill_opacity=0.06,
                               fill_color=ORANGE_MED)
        city_ellipse.move_to(axes.c2p(-0.1, -1.8))

        cluster_labels = VGroup(
            label_text("Royalty", color=YELLOW_MED).move_to(axes.c2p(2.4, 2.1)),
            label_text("Animals", color=GREEN_MED).move_to(axes.c2p(-2.4, 1.3)),
            label_text("Cities",  color=ORANGE_MED).move_to(axes.c2p(0.4, -2.3)),
        )

        self.play(Create(royal_ellipse), Create(animal_ellipse),
                  Create(city_ellipse), run_time=0.8)
        self.play(LaggedStart(*[Write(l) for l in cluster_labels], lag_ratio=0.2),
                  run_time=0.7)
        self.wait(1)

        # ── 3. The famous analogy: king - man + woman = queen ─────────────────
        analogy_txt = body_text(
            "king - man + woman  =  queen     (embedding arithmetic!)",
            color=WHITE,
        )
        analogy_txt.to_edge(DOWN, buff=0.4)
        self.play(Write(analogy_txt), run_time=1.0)

        # Draw an arrow king→queen
        king_pt  = axes.c2p(1.8, 1.5)
        queen_pt = axes.c2p(1.5, 0.9)
        analogy_arrow = Arrow(king_pt, queen_pt, color=WHITE,
                              buff=0.12, stroke_width=2)
        self.play(Create(analogy_arrow), run_time=0.5)
        self.wait(1.5)

        self.fade_all(analogy_arrow, analogy_txt, royal_ellipse,
                      animal_ellipse, city_ellipse, cluster_labels,
                      dot_group, axes, x_lbl, y_lbl, title)

        # ── 4. Summary ────────────────────────────────────────────────────────
        summary_lines = [
            "Token ID → dense vector (embedding)",
            "Similar meanings → nearby vectors",
            "Arithmetic on vectors = arithmetic on meaning",
            "Learned from context during training",
        ]
        summary = VGroup(*[body_text(f"• {l}", color=WHITE) for l in summary_lines])
        summary.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        summary.move_to(ORIGIN)
        box = SurroundingRectangle(summary, color=BLUE_MED,
                                   buff=0.35, corner_radius=0.15)
        self.play(Create(box), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(l) for l in summary], lag_ratio=0.15),
                  run_time=1.2)
        self.wait(2)

        self.fade_all(box, summary)

        # ── 5. The embedding table ─────────────────────────────────────────────
        table_title = body_text("The Embedding Table — a giant lookup dictionary", color=WHITE)
        table_title.to_edge(UP, buff=0.8)
        self.play(Write(table_title), run_time=0.7)

        # Draw a simplified table grid
        rows_count = 6
        cols_count = 8
        cell_w = 0.55
        cell_h = 0.38
        table_grid = VGroup()
        for r in range(rows_count):
            for c in range(cols_count):
                if c == 0:
                    col_color = BLUE_MED if r < 3 else GREY_MED
                    cell = Rectangle(width=cell_w * 1.8, height=cell_h,
                                     fill_color=str(col_color) + "33",
                                     fill_opacity=1,
                                     stroke_color=GREY_MED, stroke_width=0.8)
                else:
                    val = np.random.uniform(-1, 1)
                    intensity = abs(val)
                    cell = Rectangle(width=cell_w, height=cell_h,
                                     fill_color=BLUE_MED,
                                     fill_opacity=intensity * 0.7,
                                     stroke_color=GREY_DARK, stroke_width=0.5)
                cell.move_to([c * (cell_w if c > 0 else cell_w * 1.8) + (cell_w * 0.4 if c > 0 else 0),
                               -r * cell_h, 0])
                table_grid.add(cell)

        table_grid.move_to(ORIGIN + DOWN * 0.3)

        row_labels = VGroup()
        token_names = ['"king"', '"queen"', '"the"', '"...  "', '"pizza"', '"xyz"']
        for i, name in enumerate(token_names):
            lbl = label_text(name, color=BLUE_LIGHT if i < 3 else GREY_MED)
            lbl.next_to(table_grid[i * cols_count], LEFT, buff=0.15)
            row_labels.add(lbl)

        col_header = label_text("← 768 dimensions →", color=GREY_MED)
        col_header.next_to(table_grid, UP, buff=0.2)

        row_header = label_text("50,000\nrows\n(vocab)", color=GREY_MED)
        row_header.next_to(table_grid, LEFT, buff=1.0)

        self.play(Create(table_grid), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(l) for l in row_labels], lag_ratio=0.1),
                  FadeIn(col_header), FadeIn(row_header), run_time=0.8)
        self.wait(1)

        size_note = label_text("50,000 tokens × 768 dims = 38 million learned numbers!", color=YELLOW_MED)
        size_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(size_note), run_time=0.6)
        self.wait(1.5)

        self.fade_all(table_title, table_grid, row_labels, col_header, row_header, size_note)

        # ── 6. King - man + woman = queen  with 4 word dots ───────────────────
        arith_title = body_text("Semantic similarity in vector space", color=WHITE)
        arith_title.to_edge(UP, buff=0.8)
        self.play(Write(arith_title), run_time=0.7)

        axes2 = Axes(
            x_range=[-3, 3, 1], y_range=[-2.5, 2.5, 1],
            x_length=6.5, y_length=5,
            axis_config={"color": GREY_MED, "include_numbers": False},
            tips=False,
        )
        axes2.move_to(ORIGIN)

        word_positions = {
            "man":   (-1.5, -1.0, BLUE_MED),
            "woman": ( 1.5, -1.0, RED_MED),
            "king":  (-1.5,  1.5, YELLOW_MED),
            "queen": ( 1.5,  1.5, YELLOW_MED),
        }

        self.play(Create(axes2), run_time=0.7)

        dots2 = VGroup()
        for word, (x, y, col) in word_positions.items():
            d = Dot(axes2.c2p(x, y), color=col, radius=0.12)
            lbl = label_text(word, color=col)
            lbl.next_to(d, UR, buff=0.08)
            dots2.add(VGroup(d, lbl))

        self.play(LaggedStart(*[FadeIn(d) for d in dots2], lag_ratio=0.15), run_time=0.8)
        self.wait(0.5)

        # Draw parallel arrows: man→king  and  woman→queen
        man_pt   = axes2.c2p(-1.5, -1.0)
        king_pt2 = axes2.c2p(-1.5,  1.5)
        woman_pt = axes2.c2p( 1.5, -1.0)
        queen_pt2= axes2.c2p( 1.5,  1.5)

        arr_mk = Arrow(man_pt, king_pt2, color=BLUE_LIGHT, stroke_width=2, buff=0.12)
        arr_wq = Arrow(woman_pt, queen_pt2, color=RED_MED, stroke_width=2, buff=0.12)
        arr_mw = Arrow(man_pt, woman_pt, color=GREEN_MED, stroke_width=2, buff=0.12)
        arr_kq = Arrow(king_pt2, queen_pt2, color=GREEN_MED, stroke_width=2, buff=0.12)

        gender_lbl = label_text("gender direction", color=GREEN_MED)
        gender_lbl.next_to(axes2.c2p(0, -1.0), DOWN, buff=0.1)
        royalty_lbl = label_text("royalty direction", color=BLUE_LIGHT)
        royalty_lbl.next_to(axes2.c2p(-1.5, 0.25), LEFT, buff=0.1)

        self.play(Create(arr_mw), Create(arr_kq), FadeIn(gender_lbl), run_time=0.7)
        self.play(Create(arr_mk), Create(arr_wq), FadeIn(royalty_lbl), run_time=0.7)

        eq_lbl = label_text("king - man + woman = queen", color=WHITE)
        eq_lbl.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(eq_lbl), run_time=0.5)
        self.wait(1.5)

        self.fade_all(arith_title, axes2, dots2, arr_mk, arr_wq, arr_mw, arr_kq,
                      gender_lbl, royalty_lbl, eq_lbl)

        # ── 7. Embedding dimensions as a bar chart ─────────────────────────────
        dim_title = body_text("What do embedding dimensions encode?", color=WHITE)
        dim_title.to_edge(UP, buff=0.8)
        self.play(Write(dim_title), run_time=0.7)

        np.random.seed(42)
        n_bars = 20
        bar_vals = np.random.uniform(-1, 1, n_bars)
        bar_vals[3]  =  0.92   # royalty
        bar_vals[7]  = -0.85   # animate
        bar_vals[11] =  0.78   # positive sentiment
        bar_colors_dim = []
        for i, v in enumerate(bar_vals):
            if i == 3:
                bar_colors_dim.append(YELLOW_MED)
            elif i == 7:
                bar_colors_dim.append(GREEN_MED)
            elif i == 11:
                bar_colors_dim.append(BLUE_MED)
            else:
                bar_colors_dim.append(GREY_MED)

        dim_bars = VGroup()
        bar_w = 0.3
        for i, (v, col) in enumerate(zip(bar_vals, bar_colors_dim)):
            h = abs(v) * 1.5
            bar = Rectangle(width=bar_w, height=max(h, 0.02),
                            fill_color=col, fill_opacity=0.85,
                            stroke_color=WHITE, stroke_width=0.5)
            if v < 0:
                bar.next_to([i * (bar_w + 0.08) - 3.0, 0, 0], DOWN, buff=0)
            else:
                bar.next_to([i * (bar_w + 0.08) - 3.0, 0, 0], UP, buff=0)
            dim_bars.add(bar)

        dim_bars.move_to(ORIGIN + DOWN * 0.1)

        baseline = Line(LEFT * 3.5, RIGHT * 3.5, color=GREY_MED, stroke_width=1)
        baseline.move_to(ORIGIN + DOWN * 0.1)

        self.play(Create(baseline), run_time=0.3)
        self.play(LaggedStart(*[FadeIn(b) for b in dim_bars], lag_ratio=0.05), run_time=1.2)

        # Annotate notable dimensions
        ann3  = label_text("royalty?", color=YELLOW_MED)
        ann3.next_to(dim_bars[3], UP, buff=0.1)
        ann7  = label_text("animate?", color=GREEN_MED)
        ann7.next_to(dim_bars[7], DOWN, buff=0.1)
        ann11 = label_text("positive?", color=BLUE_MED)
        ann11.next_to(dim_bars[11], UP, buff=0.1)

        self.play(FadeIn(ann3), FadeIn(ann7), FadeIn(ann11), run_time=0.6)

        dim_note = label_text("Each bar = one dimension. Meaning emerges from ALL dimensions together, not any single one.", color=GREY_LIGHT)
        dim_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(dim_note), run_time=0.6)
        self.wait(1.5)

        self.fade_all(dim_title, dim_bars, baseline, ann3, ann7, ann11, dim_note)

        # ── 8. Context changes meaning — "bank" example ───────────────────────
        ctx_title = body_text("Same token, different meaning — context changes everything", color=WHITE)
        ctx_title.to_edge(UP, buff=0.8)
        self.play(Write(ctx_title), run_time=0.7)

        # Left: river bank context
        river_tokens = VGroup()
        for tok, col in [("river", BLUE_MED), ("bank", BLUE_LIGHT), ("flows", BLUE_MED)]:
            b = rounded_box(0.9, 0.5, fill_color=str(col) + "22",
                            stroke_color=col, label=tok, label_color=col)
            river_tokens.add(b)
        river_tokens.arrange(RIGHT, buff=0.15)
        river_tokens.shift(LEFT * 2.8 + UP * 1.0)

        river_label = label_text("\"river bank\"", color=BLUE_MED)
        river_label.next_to(river_tokens, UP, buff=0.2)

        # Right: finance bank context
        finance_tokens = VGroup()
        for tok, col in [("money", GREEN_MED), ("bank", GREEN_LIGHT), ("loan", GREEN_MED)]:
            b = rounded_box(0.9, 0.5, fill_color=str(col) + "22",
                            stroke_color=col, label=tok, label_color=col)
            finance_tokens.add(b)
        finance_tokens.arrange(RIGHT, buff=0.15)
        finance_tokens.shift(RIGHT * 2.8 + UP * 1.0)

        finance_label = label_text("\"finance bank\"", color=GREEN_MED)
        finance_label.next_to(finance_tokens, UP, buff=0.2)

        self.play(FadeIn(river_label), FadeIn(river_tokens),
                  FadeIn(finance_label), FadeIn(finance_tokens), run_time=0.9)
        self.wait(0.6)

        # Show diverging hidden states after layers
        div_label = label_text("After transformer layers — hidden states diverge:", color=GREY_LIGHT)
        div_label.shift(DOWN * 0.2)
        self.play(FadeIn(div_label), run_time=0.5)

        river_state = rounded_box(2.2, 0.6, fill_color=str(BLUE_MED) + "33",
                                  stroke_color=BLUE_MED,
                                  label="[water, nature, flow]", label_color=BLUE_LIGHT)
        river_state.shift(LEFT * 2.8 + DOWN * 1.0)

        finance_state = rounded_box(2.2, 0.6, fill_color=str(GREEN_MED) + "33",
                                    stroke_color=GREEN_MED,
                                    label="[money, credit, finance]", label_color=GREEN_LIGHT)
        finance_state.shift(RIGHT * 2.8 + DOWN * 1.0)

        arrow_r = Arrow(river_tokens[1].get_bottom(), river_state.get_top(),
                        color=BLUE_MED, stroke_width=1.5, buff=0.05)
        arrow_f = Arrow(finance_tokens[1].get_bottom(), finance_state.get_top(),
                        color=GREEN_MED, stroke_width=1.5, buff=0.05)

        self.play(Create(arrow_r), FadeIn(river_state),
                  Create(arrow_f), FadeIn(finance_state), run_time=0.9)

        diff_note = label_text("Same token ID — completely different hidden state after layers", color=YELLOW_MED)
        diff_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(diff_note), run_time=0.5)
        self.wait(1.5)

        self.fade_all(ctx_title, river_label, river_tokens, finance_label, finance_tokens,
                      div_label, arrow_r, river_state, arrow_f, finance_state, diff_note)

        # ── 9. Embedding vs hidden state journey ──────────────────────────────
        journey_title = body_text("The journey: token ID → embedding → hidden state", color=WHITE)
        journey_title.to_edge(UP, buff=0.8)
        self.play(Write(journey_title), run_time=0.7)

        stages = [
            ("Token ID\n1547",       GREY_MED,   "just a number\nno meaning"),
            ("Embedding\n[0.2,-0.5...]", BLUE_MED, "static lookup\nsame always"),
            ("Layer 1\nHidden State",   PURPLE_MED, "context added\nchanging"),
            ("Layer N\nHidden State",   GREEN_MED,  "rich meaning\nfully contextual"),
        ]

        stage_boxes = VGroup()
        stage_notes = VGroup()
        for label_str, col, note_str in stages:
            b = rounded_box(1.8, 0.75, fill_color=str(col) + "33",
                            stroke_color=col, label=label_str, label_color=col)
            stage_boxes.add(b)
            n = label_text(note_str, color=col)
            stage_notes.add(n)

        stage_boxes.arrange(RIGHT, buff=0.6)
        stage_boxes.move_to(ORIGIN + UP * 0.3)

        for box, note in zip(stage_boxes, stage_notes):
            note.next_to(box, DOWN, buff=0.25)

        arrows_journey = VGroup()
        for i in range(len(stage_boxes) - 1):
            a = Arrow(stage_boxes[i].get_right(), stage_boxes[i + 1].get_left(),
                      color=GREY_MED, stroke_width=1.5, buff=0.05,
                      max_tip_length_to_length_ratio=0.2)
            arrows_journey.add(a)

        self.play(LaggedStart(*[FadeIn(b) for b in stage_boxes], lag_ratio=0.2), run_time=1.0)
        self.play(LaggedStart(*[Create(a) for a in arrows_journey], lag_ratio=0.2), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(n) for n in stage_notes], lag_ratio=0.2), run_time=0.7)
        self.wait(2)

        self.fade_all(journey_title, stage_boxes, stage_notes, arrows_journey)
