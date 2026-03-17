"""
Scene 28 — DAPO: Decoupled Advantage Policy Optimization
Run: manim -pql 28_dapo.py DAPOScene
"""
from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class DAPOScene(LLMScene):
    def construct(self):

        # ── 1. Title ───────────────────────────────────────────────────────────
        title = self.show_title("DAPO", "Decoupled Advantage Policy Optimization")
        self.wait(0.8)
        self.fade_all(title)

        # ── 2. Symmetric clipping kills exploration ────────────────────────────
        prob_title = body_text("Problem: Symmetric Clipping Suppresses Exploration",
                               color=RED_MED)
        prob_title.to_edge(UP, buff=0.6)
        self.play(Write(prob_title), run_time=0.8)

        # Show the ratio window as a horizontal bar
        window_label = label_text("Policy Ratio Clip Window  [1 - e, 1 + e]",
                                  color=GREY_LIGHT)
        window_label.move_to(UP * 1.8)
        self.play(FadeIn(window_label), run_time=0.5)

        # Symmetric clip bar: lower half (red), centre point, upper half (same red)
        bar_bg = Rectangle(width=8.0, height=0.55,
                           fill_color=GREY_DARK, fill_opacity=1.0,
                           stroke_color=GREY_MED, stroke_width=1.5)
        bar_bg.move_to(UP * 1.1)

        low_bar = Rectangle(width=2.4, height=0.55,
                            fill_color=RED_MED, fill_opacity=0.75,
                            stroke_width=0)
        low_bar.move_to(bar_bg.get_left() + RIGHT * 1.2)

        high_bar = Rectangle(width=2.4, height=0.55,
                             fill_color=RED_MED, fill_opacity=0.75,
                             stroke_width=0)
        high_bar.move_to(bar_bg.get_right() + LEFT * 1.2)

        centre_line = Line(bar_bg.get_center() + UP * 0.35,
                           bar_bg.get_center() + DOWN * 0.35,
                           stroke_color=WHITE, stroke_width=2)

        lbl_low = label_text("0.8\n(-0.2)", color=RED_MED)
        lbl_low.next_to(low_bar, DOWN, buff=0.15)
        lbl_centre = label_text("1.0\n(no change)", color=WHITE)
        lbl_centre.next_to(centre_line, DOWN, buff=0.15)
        lbl_high = label_text("1.2\n(+0.2)", color=RED_MED)
        lbl_high.next_to(high_bar, DOWN, buff=0.15)

        sym_note = label_text("Both sides equal: up and down exploration equally capped",
                              color=RED_MED)
        sym_note.move_to(DOWN * 1.3)

        self.play(FadeIn(bar_bg), run_time=0.4)
        self.play(FadeIn(low_bar), FadeIn(high_bar),
                  Create(centre_line), run_time=0.5)
        self.play(FadeIn(lbl_low), FadeIn(lbl_centre), FadeIn(lbl_high),
                  run_time=0.5)
        self.play(FadeIn(sym_note), run_time=0.5)

        # Consequence explanation
        consequence = label_text(
            "Correct answer has prob 0.01  →  clip limits ratio to 1.2  →  new prob = 0.012\n"
            "Tiny step even with high reward — model can barely learn rare correct responses",
            color=ORANGE_MED,
        )
        consequence.move_to(DOWN * 2.4)
        self.play(FadeIn(consequence), run_time=0.6)
        self.wait(1.5)
        self.fade_all(prob_title, window_label, bar_bg, low_bar, high_bar,
                      centre_line, lbl_low, lbl_centre, lbl_high,
                      sym_note, consequence)

        # ── 3. Asymmetric clipping solution ───────────────────────────────────
        asym_title = body_text("Solution: Asymmetric Clipping", color=GREEN_MED)
        asym_title.to_edge(UP, buff=0.6)
        self.play(Write(asym_title), run_time=0.7)

        asym_sub = label_text(
            "clip_high = 0.28   |   clip_low = 0.20   (upper bound is wider)",
            color=GREY_LIGHT,
        )
        asym_sub.next_to(asym_title, DOWN, buff=0.3)
        self.play(FadeIn(asym_sub), run_time=0.5)

        # Draw the two bars side by side
        # GRPO bar
        grpo_lbl = label_text("GRPO  (symmetric)", color=RED_MED)
        grpo_lbl.move_to(LEFT * 3.5 + UP * 1.3)

        grpo_bg = Rectangle(width=5.2, height=0.5,
                            fill_color=GREY_DARK, fill_opacity=1.0,
                            stroke_color=GREY_MED, stroke_width=1.5)
        grpo_bg.move_to(LEFT * 3.5 + UP * 0.6)

        grpo_down = Rectangle(width=1.56, height=0.5,
                              fill_color=RED_MED, fill_opacity=0.7,
                              stroke_width=0)
        grpo_down.move_to(grpo_bg.get_left() + RIGHT * 0.78)

        grpo_up = Rectangle(width=1.56, height=0.5,
                            fill_color=RED_MED, fill_opacity=0.7,
                            stroke_width=0)
        grpo_up.move_to(grpo_bg.get_right() + LEFT * 0.78)

        grpo_cl = Line(grpo_bg.get_center() + UP * 0.3,
                       grpo_bg.get_center() + DOWN * 0.3,
                       stroke_color=WHITE, stroke_width=2)

        grpo_down_lbl = label_text("-0.20", color=RED_MED)
        grpo_down_lbl.next_to(grpo_down, DOWN, buff=0.12)
        grpo_up_lbl = label_text("+0.20", color=RED_MED)
        grpo_up_lbl.next_to(grpo_up, DOWN, buff=0.12)

        # DAPO bar
        dapo_lbl = label_text("DAPO  (asymmetric)", color=GREEN_MED)
        dapo_lbl.move_to(RIGHT * 3.5 + UP * 1.3)

        dapo_bg = Rectangle(width=5.2, height=0.5,
                            fill_color=GREY_DARK, fill_opacity=1.0,
                            stroke_color=GREY_MED, stroke_width=1.5)
        dapo_bg.move_to(RIGHT * 3.5 + UP * 0.6)

        # Down side: 0.20 → 37% of half
        dapo_down = Rectangle(width=1.3, height=0.5,
                              fill_color=RED_MED, fill_opacity=0.7,
                              stroke_width=0)
        dapo_down.move_to(dapo_bg.get_left() + RIGHT * 0.65)

        # Up side: 0.28 → wider
        dapo_up = Rectangle(width=1.82, height=0.5,
                            fill_color=GREEN_MED, fill_opacity=0.7,
                            stroke_width=0)
        dapo_up.move_to(dapo_bg.get_right() + LEFT * 0.91)

        dapo_cl = Line(dapo_bg.get_center() + UP * 0.3,
                       dapo_bg.get_center() + DOWN * 0.3,
                       stroke_color=WHITE, stroke_width=2)

        dapo_down_lbl = label_text("-0.20", color=RED_MED)
        dapo_down_lbl.next_to(dapo_down, DOWN, buff=0.12)
        dapo_up_lbl = label_text("+0.28", color=GREEN_MED)
        dapo_up_lbl.next_to(dapo_up, DOWN, buff=0.12)

        self.play(
            FadeIn(grpo_lbl), FadeIn(dapo_lbl),
            FadeIn(grpo_bg), FadeIn(dapo_bg),
            run_time=0.5,
        )
        self.play(
            FadeIn(grpo_down), FadeIn(grpo_up), Create(grpo_cl),
            FadeIn(dapo_down), FadeIn(dapo_up), Create(dapo_cl),
            run_time=0.6,
        )
        self.play(
            FadeIn(grpo_down_lbl), FadeIn(grpo_up_lbl),
            FadeIn(dapo_down_lbl), FadeIn(dapo_up_lbl),
            run_time=0.4,
        )

        insight = label_text(
            "Bigger upward room: model can make larger steps toward high-reward responses\n"
            "Downward remains conservative: known-good responses are not abandoned",
            color=GREEN_LIGHT,
        )
        insight.move_to(DOWN * 1.5)
        self.play(FadeIn(insight), run_time=0.6)
        self.wait(1.5)
        self.fade_all(asym_title, asym_sub,
                      grpo_lbl, grpo_bg, grpo_down, grpo_up, grpo_cl,
                      grpo_down_lbl, grpo_up_lbl,
                      dapo_lbl, dapo_bg, dapo_down, dapo_up, dapo_cl,
                      dapo_down_lbl, dapo_up_lbl, insight)

        # ── 4. Token-level loss vs episode-level loss ──────────────────────────
        loss_title = body_text("Token-Level Loss vs Episode-Level Loss", color=BLUE_MED)
        loss_title.to_edge(UP, buff=0.6)
        self.play(Write(loss_title), run_time=0.7)

        # Left column: GRPO (episode-level)
        grpo_header = label_text("GRPO  — episode-level", color=RED_MED)
        grpo_header.move_to(LEFT * 3.5 + UP * 1.8)

        ep_short = rounded_box(3.0, 0.55, fill_color=str(RED_MED) + "22",
                               stroke_color=RED_MED,
                               label="Short response  (80 tokens)",
                               label_color=WHITE)
        ep_short.move_to(LEFT * 3.5 + UP * 0.9)

        ep_long = rounded_box(3.0, 0.55, fill_color=str(RED_MED) + "33",
                              stroke_color=RED_MED,
                              label="Long response  (800 tokens)",
                              label_color=WHITE)
        ep_long.move_to(LEFT * 3.5 + UP * 0.1)

        ep_denom = label_text("Denominator: 2 episodes  →  long response\n"
                              "gets 10x more gradient influence",
                              color=ORANGE_MED)
        ep_denom.move_to(LEFT * 3.5 + DOWN * 0.85)

        # Right column: DAPO (token-level)
        dapo_header = label_text("DAPO  — token-level", color=GREEN_MED)
        dapo_header.move_to(RIGHT * 3.5 + UP * 1.8)

        tok_short = rounded_box(3.0, 0.55, fill_color=str(GREEN_MED) + "22",
                                stroke_color=GREEN_MED,
                                label="Short response  (80 tokens)",
                                label_color=WHITE)
        tok_short.move_to(RIGHT * 3.5 + UP * 0.9)

        tok_long = rounded_box(3.0, 0.55, fill_color=str(GREEN_MED) + "33",
                               stroke_color=GREEN_MED,
                               label="Long response  (800 tokens)",
                               label_color=WHITE)
        tok_long.move_to(RIGHT * 3.5 + UP * 0.1)

        tok_denom = label_text("Denominator: 880 tokens  →  each token\n"
                               "contributes equally regardless of length",
                               color=GREEN_LIGHT)
        tok_denom.move_to(RIGHT * 3.5 + DOWN * 0.85)

        divider = Line(UP * 2.2, DOWN * 2.0,
                       stroke_color=GREY_MED, stroke_width=1.5)
        divider.move_to(ORIGIN)

        length_bias_note = label_text(
            "Length bias removed: model has no incentive to write longer responses\n"
            "just to amplify its own gradient signal",
            color=WHITE,
        )
        length_bias_note.to_edge(DOWN, buff=0.4)

        self.play(Create(divider), run_time=0.3)
        self.play(FadeIn(grpo_header), FadeIn(dapo_header), run_time=0.4)
        self.play(FadeIn(ep_short), FadeIn(tok_short), run_time=0.4)
        self.play(FadeIn(ep_long), FadeIn(tok_long), run_time=0.4)
        self.play(FadeIn(ep_denom), FadeIn(tok_denom), run_time=0.5)
        self.play(FadeIn(length_bias_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(loss_title, divider,
                      grpo_header, ep_short, ep_long, ep_denom,
                      dapo_header, tok_short, tok_long, tok_denom,
                      length_bias_note)

        # ── 5. Entropy bonus: diversity visualization ──────────────────────────
        ent_title = body_text("Entropy Bonus: Preventing Diversity Collapse",
                              color=PURPLE_MED)
        ent_title.to_edge(UP, buff=0.6)
        self.play(Write(ent_title), run_time=0.7)

        # Narrow distribution (collapsed)
        narrow_lbl = label_text("Without entropy bonus\n(collapsed distribution)",
                                color=RED_MED)
        narrow_lbl.move_to(LEFT * 3.5 + UP * 1.6)

        narrow_bars_data = [("A", 0.88), ("B", 0.07), ("C", 0.03), ("D", 0.02)]
        narrow_bars = VGroup()
        for i, (tok, prob) in enumerate(narrow_bars_data):
            h = 2.0 * prob
            bar = Rectangle(width=0.45, height=max(h, 0.04),
                            fill_color=RED_MED, fill_opacity=0.75,
                            stroke_color=RED_MED, stroke_width=1)
            bar_top = bar.get_bottom() + UP * max(h, 0.04)
            bar.move_to(LEFT * 5.2 + RIGHT * i * 0.65 + DOWN * 0.2)
            bar.align_to(LEFT * 5.2 + RIGHT * i * 0.65 + DOWN * 0.5, DOWN)
            tok_lbl = label_text(tok, color=GREY_LIGHT)
            tok_lbl.next_to(bar, DOWN, buff=0.1)
            narrow_bars.add(VGroup(bar, tok_lbl))
        narrow_bars.move_to(LEFT * 3.5 + DOWN * 0.1)

        narrow_note = label_text("Model always picks 'A'\nNo variance in rewards\nGradient = 0",
                                 color=RED_MED)
        narrow_note.move_to(LEFT * 3.5 + DOWN * 1.9)

        # Wide distribution (healthy entropy)
        wide_lbl = label_text("With entropy bonus\n(healthy distribution)",
                              color=GREEN_MED)
        wide_lbl.move_to(RIGHT * 3.5 + UP * 1.6)

        wide_bars_data = [("A", 0.38), ("B", 0.28), ("C", 0.20), ("D", 0.14)]
        wide_bars = VGroup()
        for i, (tok, prob) in enumerate(wide_bars_data):
            h = 2.0 * prob
            bar = Rectangle(width=0.45, height=max(h, 0.04),
                            fill_color=GREEN_MED, fill_opacity=0.75,
                            stroke_color=GREEN_MED, stroke_width=1)
            bar.move_to(RIGHT * 2.1 + RIGHT * i * 0.65 + DOWN * 0.2)
            bar.align_to(RIGHT * 2.1 + RIGHT * i * 0.65 + DOWN * 0.5, DOWN)
            tok_lbl = label_text(tok, color=GREY_LIGHT)
            tok_lbl.next_to(bar, DOWN, buff=0.1)
            wide_bars.add(VGroup(bar, tok_lbl))
        wide_bars.move_to(RIGHT * 3.5 + DOWN * 0.1)

        wide_note = label_text("Diverse responses generated\nReward variance is nonzero\nGradient flows",
                               color=GREEN_MED)
        wide_note.move_to(RIGHT * 3.5 + DOWN * 1.9)

        div_line = Line(UP * 2.2, DOWN * 2.8,
                        stroke_color=GREY_MED, stroke_width=1.5)

        self.play(Create(div_line), run_time=0.3)
        self.play(FadeIn(narrow_lbl), FadeIn(wide_lbl), run_time=0.4)
        self.play(
            LaggedStart(*[FadeIn(b) for b in narrow_bars], lag_ratio=0.1),
            LaggedStart(*[FadeIn(b) for b in wide_bars], lag_ratio=0.1),
            run_time=0.8,
        )
        self.play(FadeIn(narrow_note), FadeIn(wide_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(ent_title, div_line, narrow_lbl, wide_lbl,
                      narrow_bars, wide_bars, narrow_note, wide_note)

        # ── 6. Dynamic sampling: filter non-learnable prompts ─────────────────
        samp_title = body_text("Dynamic Sampling: Filter for Learnable Prompts",
                               color=YELLOW_MED)
        samp_title.to_edge(UP, buff=0.6)
        self.play(Write(samp_title), run_time=0.7)

        # Three prompt types shown as rows
        # Too easy
        easy_box = rounded_box(3.8, 0.65,
                               fill_color=str(GREY_MED) + "33",
                               stroke_color=GREY_MED,
                               label="All 8 responses CORRECT",
                               label_color=GREY_LIGHT)
        easy_box.move_to(LEFT * 2.5 + UP * 1.5)
        easy_tag = label_text("Too easy — skip", color=GREY_MED)
        easy_tag.next_to(easy_box, RIGHT, buff=0.4)
        easy_note = label_text("advantage = 0 for all\ngradient = 0",
                               color=GREY_MED)
        easy_note.next_to(easy_box, DOWN, buff=0.2)

        # Too hard
        hard_box = rounded_box(3.8, 0.65,
                               fill_color=str(RED_MED) + "22",
                               stroke_color=RED_MED,
                               label="All 8 responses WRONG",
                               label_color=WHITE)
        hard_box.move_to(LEFT * 2.5 + DOWN * 0.2)
        hard_tag = label_text("Too hard — skip", color=RED_MED)
        hard_tag.next_to(hard_box, RIGHT, buff=0.4)
        hard_note = label_text("advantage = 0 for all\ngradient = 0",
                               color=RED_MED)
        hard_note.next_to(hard_box, DOWN, buff=0.2)

        # Just right
        right_box = rounded_box(3.8, 0.65,
                                fill_color=str(GREEN_MED) + "22",
                                stroke_color=GREEN_MED,
                                label="3 correct, 5 wrong",
                                label_color=GREEN_LIGHT)
        right_box.move_to(LEFT * 2.5 + DOWN * 1.9)
        right_tag = label_text("Learnable — KEEP", color=GREEN_MED)
        right_tag.next_to(right_box, RIGHT, buff=0.4)
        right_note = label_text("variance in rewards\ngradient flows",
                                color=GREEN_MED)
        right_note.next_to(right_box, DOWN, buff=0.2)

        filter_box = rounded_box(2.2, 1.4,
                                 fill_color=str(YELLOW_MED) + "22",
                                 stroke_color=YELLOW_MED,
                                 label="Filter\nStep",
                                 label_color=WHITE)
        filter_box.move_to(RIGHT * 4.0 + DOWN * 0.2)

        arr_easy = Arrow(easy_box.get_right(),
                         filter_box.get_left() + UP * 0.5,
                         color=GREY_MED, buff=0.05, stroke_width=1.5,
                         max_tip_length_to_length_ratio=0.2)
        arr_hard = Arrow(hard_box.get_right(),
                         filter_box.get_left(),
                         color=RED_MED, buff=0.05, stroke_width=1.5,
                         max_tip_length_to_length_ratio=0.2)
        arr_right = Arrow(right_box.get_right(),
                          filter_box.get_left() + DOWN * 0.5,
                          color=GREEN_MED, buff=0.05, stroke_width=1.5,
                          max_tip_length_to_length_ratio=0.2)

        self.play(FadeIn(easy_box), FadeIn(easy_tag), FadeIn(easy_note),
                  run_time=0.5)
        self.play(FadeIn(hard_box), FadeIn(hard_tag), FadeIn(hard_note),
                  run_time=0.5)
        self.play(FadeIn(right_box), FadeIn(right_tag), FadeIn(right_note),
                  run_time=0.5)
        self.play(FadeIn(filter_box), run_time=0.4)
        self.play(Create(arr_easy), Create(arr_hard), Create(arr_right),
                  run_time=0.6)

        oversample_note = label_text(
            "Oversample 2-3x more prompts than needed,\nthen filter to keep only learnable ones",
            color=WHITE,
        )
        oversample_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(oversample_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(samp_title, easy_box, easy_tag, easy_note,
                      hard_box, hard_tag, hard_note,
                      right_box, right_tag, right_note,
                      filter_box, arr_easy, arr_hard, arr_right,
                      oversample_note)

        # ── 7. The learnable prompt concept ───────────────────────────────────
        learn_title = body_text("The Learnable Prompt Concept", color=YELLOW_MED)
        learn_title.to_edge(UP, buff=0.6)
        self.play(Write(learn_title), run_time=0.7)

        # Show a training curve analogy with text boxes at three points
        early_box = rounded_box(3.4, 1.1,
                                fill_color=str(RED_MED) + "22",
                                stroke_color=RED_MED,
                                label="Early training\nAll hard prompts = wrong\nAll easy prompts = right",
                                label_color=WHITE)
        early_box.move_to(LEFT * 4.0 + UP * 0.5)

        mid_box = rounded_box(3.4, 1.1,
                              fill_color=str(YELLOW_MED) + "22",
                              stroke_color=YELLOW_MED,
                              label="Mid training\nMedium prompts now learnable\nHard prompts still wrong",
                              label_color=WHITE)
        mid_box.move_to(ORIGIN + UP * 0.5)

        late_box = rounded_box(3.4, 1.1,
                               fill_color=str(GREEN_MED) + "22",
                               stroke_color=GREEN_MED,
                               label="Late training\nHard prompts become learnable\nEasy ones all correct = filtered",
                               label_color=WHITE)
        late_box.move_to(RIGHT * 4.0 + UP * 0.5)

        arr_em = Arrow(early_box.get_right(), mid_box.get_left(),
                       color=WHITE, buff=0.05, stroke_width=1.5,
                       max_tip_length_to_length_ratio=0.18)
        arr_ml = Arrow(mid_box.get_right(), late_box.get_left(),
                       color=WHITE, buff=0.05, stroke_width=1.5,
                       max_tip_length_to_length_ratio=0.18)

        concept_note = label_text(
            "The learnable zone shifts automatically as the model improves.\n"
            "No manual curriculum scheduling required — the model self-calibrates.",
            color=GREEN_LIGHT,
        )
        concept_note.to_edge(DOWN, buff=0.4)

        self.play(FadeIn(early_box), run_time=0.4)
        self.play(Create(arr_em), FadeIn(mid_box), run_time=0.5)
        self.play(Create(arr_ml), FadeIn(late_box), run_time=0.5)
        self.play(FadeIn(concept_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(learn_title, early_box, mid_box, late_box,
                      arr_em, arr_ml, concept_note)

        # ── 8. DAPO full pipeline diagram ──────────────────────────────────────
        pipe_title = body_text("DAPO Full Pipeline", color=WHITE)
        pipe_title.to_edge(UP, buff=0.6)
        self.play(Write(pipe_title), run_time=0.7)

        steps = [
            ("Oversample\nPrompts", BLUE_MED),
            ("Generate G\nResponses", BLUE_MED),
            ("Compute\nRewards", GREEN_MED),
            ("Filter:\nKeep Learnable", YELLOW_MED),
            ("Compute Group\nAdvantage", ORANGE_MED),
            ("Asymmetric\nClip Loss", RED_MED),
            ("Entropy\nBonus", PURPLE_MED),
            ("Optimizer\nStep", GREEN_MED),
        ]

        boxes = VGroup()
        for lbl, col in steps:
            b = rounded_box(1.55, 0.85,
                            fill_color=str(col) + "22",
                            stroke_color=col,
                            label=lbl, label_color=WHITE)
            boxes.add(b)

        boxes.arrange(RIGHT, buff=0.28)
        boxes.scale_to_fit_width(13.0)
        boxes.move_to(ORIGIN + DOWN * 0.3)

        arrows = VGroup(*[
            Arrow(boxes[i].get_right(), boxes[i + 1].get_left(),
                  color=GREY_MED, buff=0.04, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.25)
            for i in range(len(boxes) - 1)
        ])

        back_arrow = CurvedArrow(boxes[-1].get_bottom() + DOWN * 0.05,
                                 boxes[0].get_bottom() + DOWN * 0.05,
                                 angle=-TAU / 4.5,
                                 color=GREY_MED, stroke_width=1.5,
                                 tip_length=0.18)

        token_note = label_text(
            "Token-level normalization applied at step 6  |  All four DAPO contributions active",
            color=GREY_LIGHT,
        )
        token_note.to_edge(DOWN, buff=0.35)

        self.play(LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.12),
                  run_time=1.4)
        self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.08),
                  run_time=0.8)
        self.play(Create(back_arrow), run_time=0.5)
        self.play(FadeIn(token_note), run_time=0.4)
        self.wait(1.5)
        self.fade_all(pipe_title, boxes, arrows, back_arrow, token_note)

        # ── 9. Before/after comparison: GRPO vs DAPO response quality ─────────
        ba_title = body_text("GRPO vs DAPO: Training Outcome Comparison",
                             color=WHITE)
        ba_title.to_edge(UP, buff=0.6)
        self.play(Write(ba_title), run_time=0.7)

        grpo_col_lbl = label_text("GRPO (vanilla)", color=RED_MED)
        grpo_col_lbl.move_to(LEFT * 3.5 + UP * 1.9)
        dapo_col_lbl = label_text("DAPO", color=GREEN_MED)
        dapo_col_lbl.move_to(RIGHT * 3.5 + UP * 1.9)

        comparison_data = [
            ("Exploration",        "Low — symmetric clip blocks\nupward moves on rare responses",
             "High — asymmetric clip allows\nlarger upward steps"),
            ("Response Length",    "Biased long — episode normalization\nrewarded longer outputs",
             "Fair — token normalization\neliminate length bias"),
            ("Diversity",          "Collapses over time — no\nentropy regularization",
             "Maintained — entropy bonus\nprevents template collapse"),
            ("Compute Efficiency", "Wastes budget on all-correct\nand all-wrong prompts",
             "Every gradient step is\non a learnable prompt"),
        ]

        rows = VGroup()
        for dimension, grpo_txt, dapo_txt in comparison_data:
            dim_lbl = label_text(dimension, color=WHITE)
            grpo_cell = label_text(grpo_txt, color=RED_MED)
            dapo_cell = label_text(dapo_txt, color=GREEN_MED)
            row = VGroup(dim_lbl, grpo_cell, dapo_cell)
            rows.add(row)

        row_y_positions = [1.2, 0.35, -0.5, -1.35]
        for row, y in zip(rows, row_y_positions):
            dim_lbl, grpo_cell, dapo_cell = row
            dim_lbl.move_to(LEFT * 6.0 + UP * y)
            dim_lbl.align_to(LEFT * 6.0, LEFT)
            grpo_cell.move_to(LEFT * 2.8 + UP * y)
            grpo_cell.align_to(LEFT * 2.8, LEFT)
            dapo_cell.move_to(RIGHT * 1.0 + UP * y)
            dapo_cell.align_to(RIGHT * 1.0, LEFT)

        h_line = Line(LEFT * 6.5, RIGHT * 6.5,
                      stroke_color=GREY_MED, stroke_width=1)
        h_line.move_to(UP * 1.65)
        v_line1 = Line(UP * 1.7, DOWN * 2.0,
                       stroke_color=GREY_MED, stroke_width=1)
        v_line1.move_to(LEFT * 1.0)
        v_line2 = Line(UP * 1.7, DOWN * 2.0,
                       stroke_color=GREY_MED, stroke_width=1)
        v_line2.move_to(RIGHT * 3.3)

        self.play(Create(h_line), Create(v_line1), Create(v_line2),
                  FadeIn(grpo_col_lbl), FadeIn(dapo_col_lbl),
                  run_time=0.5)
        self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.2),
                  run_time=1.2)
        self.wait(1.5)
        self.fade_all(ba_title, h_line, v_line1, v_line2,
                      grpo_col_lbl, dapo_col_lbl, rows)

        # ── 10. DAPO vs GRPO vs Dr.GRPO comparison table ──────────────────────
        cmp_title = body_text("Algorithm Comparison: GRPO / Dr.GRPO / DAPO",
                              color=WHITE)
        cmp_title.to_edge(UP, buff=0.6)
        self.play(Write(cmp_title), run_time=0.7)

        # Column headers
        feat_hdr = label_text("Feature", color=WHITE)
        grpo_hdr = label_text("GRPO", color=RED_MED)
        drgrpo_hdr = label_text("Dr.GRPO", color=ORANGE_MED)
        dapo_hdr = label_text("DAPO", color=GREEN_MED)

        feat_hdr.move_to(LEFT * 5.2 + UP * 1.8)
        grpo_hdr.move_to(LEFT * 1.8 + UP * 1.8)
        drgrpo_hdr.move_to(RIGHT * 1.4 + UP * 1.8)
        dapo_hdr.move_to(RIGHT * 4.6 + UP * 1.8)

        table_data = [
            ("Clip bounds",    "Symmetric",      "Symmetric",      "Asymmetric"),
            ("Loss normalize", "Episode-level",  "Unbiased (math)", "Token-level"),
            ("Entropy reg.",   "None",           "None",           "Explicit bonus"),
            ("Sampling",       "Uniform",        "Uniform",        "Dynamic filter"),
            ("Value model",    "None",           "None",           "None"),
            ("Primary goal",   "Simplify PPO",   "Remove bias",    "Scale perf."),
        ]

        row_colors = [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE]
        col_colors = [GREY_LIGHT, RED_MED, ORANGE_MED, GREEN_MED]

        table_rows = VGroup()
        y_start = 1.15
        for i, (feat, grpo_v, drgrpo_v, dapo_v) in enumerate(table_data):
            y = y_start - i * 0.6
            feat_cell = label_text(feat, color=GREY_LIGHT)
            grpo_cell = label_text(grpo_v, color=RED_MED)
            drgrpo_cell = label_text(drgrpo_v, color=ORANGE_MED)
            dapo_cell = label_text(dapo_v, color=GREEN_MED)

            feat_cell.move_to(LEFT * 5.2 + UP * y)
            feat_cell.align_to(LEFT * 5.2, LEFT)
            grpo_cell.move_to(LEFT * 1.8 + UP * y)
            drgrpo_cell.move_to(RIGHT * 1.4 + UP * y)
            dapo_cell.move_to(RIGHT * 4.6 + UP * y)

            table_rows.add(VGroup(feat_cell, grpo_cell, drgrpo_cell, dapo_cell))

        h_header = Line(LEFT * 6.5, RIGHT * 6.5,
                        stroke_color=GREY_MED, stroke_width=1)
        h_header.move_to(UP * 1.55)

        tv1 = Line(UP * 2.0, DOWN * 2.5, stroke_color=GREY_DARK, stroke_width=1)
        tv1.move_to(LEFT * 3.4)
        tv2 = Line(UP * 2.0, DOWN * 2.5, stroke_color=GREY_DARK, stroke_width=1)
        tv2.move_to(RIGHT * 0.0)
        tv3 = Line(UP * 2.0, DOWN * 2.5, stroke_color=GREY_DARK, stroke_width=1)
        tv3.move_to(RIGHT * 3.2)

        self.play(Create(h_header), Create(tv1), Create(tv2), Create(tv3),
                  FadeIn(feat_hdr), FadeIn(grpo_hdr),
                  FadeIn(drgrpo_hdr), FadeIn(dapo_hdr),
                  run_time=0.6)
        self.play(LaggedStart(*[FadeIn(r) for r in table_rows], lag_ratio=0.15),
                  run_time=1.2)

        table_note = label_text(
            "DAPO is the most comprehensive: addresses exploration, normalization, diversity, and data efficiency",
            color=GREEN_LIGHT,
        )
        table_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(table_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(cmp_title, h_header, tv1, tv2, tv3,
                      feat_hdr, grpo_hdr, drgrpo_hdr, dapo_hdr,
                      table_rows, table_note)

        # ── 11. Summary ────────────────────────────────────────────────────────
        sum_title = body_text("DAPO — Summary", color=WHITE)
        sum_title.to_edge(UP, buff=0.6)
        self.play(Write(sum_title), run_time=0.7)

        summary_items = [
            ("Asymmetric Clipping",    GREEN_MED,    "clip_high > clip_low  →  more room to explore better responses"),
            ("Token-Level Loss",       BLUE_MED,     "divide by token count, not episodes  →  removes length bias"),
            ("Entropy Bonus",          PURPLE_MED,   "small regularization  →  prevents diversity collapse"),
            ("Dynamic Sampling",       YELLOW_MED,   "oversample, filter  →  train only on learnable prompts"),
        ]

        sum_rows = VGroup()
        for name, col, desc in summary_items:
            name_box = rounded_box(2.8, 0.55,
                                   fill_color=str(col) + "22",
                                   stroke_color=col,
                                   label=name, label_color=WHITE)
            desc_txt = label_text(desc, color=WHITE)
            desc_txt.next_to(name_box, RIGHT, buff=0.4)
            sum_rows.add(VGroup(name_box, desc_txt))

        sum_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        sum_rows.scale_to_fit_width(12.8)
        sum_rows.move_to(ORIGIN + DOWN * 0.3)

        next_note = label_text(
            "Up next:  CISPO — adaptive clip bounds that self-tune during training",
            color=GREY_LIGHT,
        )
        next_note.to_edge(DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(r) for r in sum_rows], lag_ratio=0.2),
                  run_time=1.2)
        self.play(FadeIn(next_note), run_time=0.5)
        self.wait(2.0)
        self.fade_all(sum_title, sum_rows, next_note)
        self.wait(0.5)
