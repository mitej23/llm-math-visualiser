"""
Scene 04 — Tokenization
Run: manim -pql 04_tokenization.py TokenizationScene
"""

from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class TokenizationScene(LLMScene):
    def construct(self):
        title = self.show_title("Tokenization", "How Text Becomes Numbers")
        self.wait(0.5)

        # ── 1. Write a sentence, then split into tokens ───────────────────────
        sentence = "The cat sat on the mat."
        tokens_text = ["The", " cat", " sat", " on", " the", " mat", "."]
        token_ids   = [464,   3797,  3332,  319,  262,   2603,  13]
        token_colors = [GREEN_MED, BLUE_MED, PURPLE_MED, ORANGE_MED,
                        GREEN_MED, RED_MED, YELLOW_MED]

        sent_txt = body_text(f'"{sentence}"', color=WHITE)
        sent_txt.shift(UP * 1.5)
        self.play(Write(sent_txt), run_time=0.9)
        self.wait(0.5)

        # Step label
        step1 = label_text("Step 1 — Split into tokens", color=GREY_LIGHT)
        step1.next_to(sent_txt, DOWN, buff=0.4)
        self.play(FadeIn(step1), run_time=0.5)

        # Build token boxes
        token_boxes = VGroup()
        for tok, col in zip(tokens_text, token_colors):
            box = rounded_box(
                width=max(len(tok) * 0.18 + 0.5, 0.7),
                height=0.55,
                fill_color=col + "33",  # 20% opacity
                stroke_color=col,
                label=tok,
                label_color=col,
            )
            token_boxes.add(box)

        token_boxes.arrange(RIGHT, buff=0.18)
        token_boxes.next_to(step1, DOWN, buff=0.45)
        self.play(LaggedStart(*[FadeIn(b) for b in token_boxes], lag_ratio=0.1),
                  run_time=1.2)
        self.wait(0.8)

        # ── 2. Map tokens to IDs ──────────────────────────────────────────────
        step2 = label_text("Step 2 — Assign numeric IDs from the vocabulary",
                           color=GREY_LIGHT)
        step2.next_to(token_boxes, DOWN, buff=0.5)
        self.play(FadeIn(step2), run_time=0.5)

        id_boxes = VGroup()
        for tid, col in zip(token_ids, token_colors):
            box = rounded_box(
                width=0.85, height=0.55,
                fill_color=GREY_DARK,
                stroke_color=col,
                label=str(tid),
                label_color=col,
            )
            id_boxes.add(box)

        id_boxes.arrange(RIGHT, buff=0.18)
        id_boxes.next_to(step2, DOWN, buff=0.35)

        # Draw arrows from token to ID
        arrows = VGroup()
        for t_box, i_box in zip(token_boxes, id_boxes):
            arr = Arrow(t_box.get_bottom(), i_box.get_top(),
                        color=GREY_MED, buff=0.05,
                        stroke_width=1.5, max_tip_length_to_length_ratio=0.2)
            arrows.add(arr)

        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.08),
                  run_time=1.0)
        self.play(LaggedStart(*[FadeIn(b) for b in id_boxes], lag_ratio=0.08),
                  run_time=0.9)
        self.wait(1)

        # ── 3. Show subword tokenization for a rare word ──────────────────────
        self.fade_all(sent_txt, step1, token_boxes, arrows, id_boxes, step2)

        rare_label = body_text("Rare word → broken into subword tokens:", color=WHITE)
        rare_label.shift(UP * 2.0)
        self.play(Write(rare_label), run_time=0.7)

        rare_word = body_text('"Schwarzenegger"', color=YELLOW_MED)
        rare_word.next_to(rare_label, DOWN, buff=0.4)
        self.play(Write(rare_word), run_time=0.6)

        sub_tokens = ["Sch", "war", "zen", "egg", "er"]
        sub_colors  = [BLUE_MED, GREEN_MED, ORANGE_MED, PURPLE_MED, RED_MED]
        sub_boxes = VGroup()
        for tok, col in zip(sub_tokens, sub_colors):
            b = rounded_box(0.9, 0.55, fill_color=col + "33",
                            stroke_color=col, label=tok, label_color=col)
            sub_boxes.add(b)

        sub_boxes.arrange(RIGHT, buff=0.15)
        sub_boxes.next_to(rare_word, DOWN, buff=0.4)
        arrow_down = Arrow(rare_word.get_bottom(), sub_boxes.get_top(),
                           color=WHITE, buff=0.05, stroke_width=2)

        self.play(GrowArrow(arrow_down), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(b) for b in sub_boxes], lag_ratio=0.15),
                  run_time=0.9)

        count_label = label_text("1 word  →  5 tokens", color=GREY_LIGHT)
        count_label.next_to(sub_boxes, DOWN, buff=0.3)
        self.play(FadeIn(count_label), run_time=0.5)
        self.wait(1.5)

        # ── 4. Context window callout ──────────────────────────────────────────
        self.fade_all(rare_label, rare_word, arrow_down, sub_boxes, count_label, title)

        ctx_box = rounded_box(7.0, 1.5, fill_color=BLUE_DARK, stroke_color=BLUE_MED,
                              label="Context Window  =  max tokens the model can see at once")
        ctx_box.move_to(ORIGIN)

        examples_txt = VGroup(
            label_text("128k tokens  ≈  100,000 words  ≈  200 pages", color=GREEN_LIGHT),
            label_text("1 token  ≈  0.75 English words on average", color=GREY_LIGHT),
        )
        examples_txt.arrange(DOWN, buff=0.25)
        examples_txt.next_to(ctx_box, DOWN, buff=0.45)

        self.play(FadeIn(ctx_box), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(e) for e in examples_txt], lag_ratio=0.3),
                  run_time=0.9)
        self.wait(2)
