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
                fill_color=str(col) + "33",  # 20% opacity
                stroke_color=col,
                label=tok,
                label_color=WHITE,
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
                label_color=WHITE,
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

        self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.08),
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
            b = rounded_box(0.9, 0.55, fill_color=str(col) + "33",
                            stroke_color=col, label=tok, label_color=WHITE)
            sub_boxes.add(b)

        sub_boxes.arrange(RIGHT, buff=0.15)
        sub_boxes.next_to(rare_word, DOWN, buff=0.4)
        arrow_down = Arrow(rare_word.get_bottom(), sub_boxes.get_top(),
                           color=WHITE, buff=0.05, stroke_width=2)

        self.play(Create(arrow_down), run_time=0.4)
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

        # ── 5. Why not character-by-character? ────────────────────────────────
        self.fade_all(ctx_box, examples_txt)

        compare_title = body_text("Three approaches to splitting text", color=WHITE)
        compare_title.to_edge(UP, buff=0.8)
        self.play(Write(compare_title), run_time=0.7)

        # Row 1 — character level
        char_label = label_text("Character:", color=RED_MED)
        char_label.shift(LEFT * 4.5 + UP * 1.2)
        char_tokens = VGroup()
        for ch, col in zip(["h","e","l","l","o"], [RED_MED]*5):
            b = rounded_box(0.45, 0.5, fill_color=str(RED_MED) + "22",
                            stroke_color=RED_MED, label=ch, label_color=WHITE)
            char_tokens.add(b)
        char_tokens.arrange(RIGHT, buff=0.1)
        char_tokens.next_to(char_label, RIGHT, buff=0.3)
        char_note = label_text("5 tokens — too slow, loses word structure", color=RED_MED)
        char_note.next_to(char_tokens, RIGHT, buff=0.3)

        # Row 2 — word level
        word_label = label_text("Word:", color=ORANGE_MED)
        word_label.shift(LEFT * 4.5 + UP * 0.0)
        word_box = rounded_box(1.2, 0.5, fill_color=str(ORANGE_MED) + "22",
                               stroke_color=ORANGE_MED, label="hello", label_color=WHITE)
        word_box.next_to(word_label, RIGHT, buff=0.3)
        word_note = label_text("1 token — but 170,000+ word vocab is huge!", color=ORANGE_MED)
        word_note.next_to(word_box, RIGHT, buff=0.3)

        # Row 3 — BPE
        bpe_label = label_text("BPE:", color=GREEN_MED)
        bpe_label.shift(LEFT * 4.5 + DOWN * 1.2)
        bpe_box = rounded_box(1.2, 0.5, fill_color=str(GREEN_MED) + "22",
                              stroke_color=GREEN_MED, label="hello", label_color=WHITE)
        bpe_box.next_to(bpe_label, RIGHT, buff=0.3)
        bpe_note = label_text("1 token — vocab ~50k, goldilocks zone!", color=GREEN_MED)
        bpe_note.next_to(bpe_box, RIGHT, buff=0.3)

        self.play(FadeIn(char_label), LaggedStart(*[FadeIn(b) for b in char_tokens], lag_ratio=0.1),
                  run_time=0.8)
        self.play(FadeIn(char_note), run_time=0.5)
        self.wait(0.5)

        self.play(FadeIn(word_label), FadeIn(word_box), run_time=0.6)
        self.play(FadeIn(word_note), run_time=0.5)
        self.wait(0.5)

        self.play(FadeIn(bpe_label), FadeIn(bpe_box), run_time=0.6)
        self.play(FadeIn(bpe_note), run_time=0.5)
        self.wait(1.5)

        self.fade_all(compare_title, char_label, char_tokens, char_note,
                      word_label, word_box, word_note,
                      bpe_label, bpe_box, bpe_note)

        # ── 6. BPE step by step ────────────────────────────────────────────────
        bpe_title = body_text("Byte Pair Encoding — 3 merge steps", color=WHITE)
        bpe_title.to_edge(UP, buff=0.8)
        self.play(Write(bpe_title), run_time=0.7)

        step_colors = [BLUE_MED, GREEN_MED, PURPLE_MED]

        # Initial characters
        init_label = label_text("Start: characters of  \"lower\"", color=GREY_LIGHT)
        init_label.shift(UP * 1.8)
        self.play(FadeIn(init_label), run_time=0.5)

        chars = ["l", "o", "w", "e", "r"]
        char_boxes = VGroup()
        for ch in chars:
            b = rounded_box(0.5, 0.5, fill_color=GREY_DARK,
                            stroke_color=GREY_MED, label=ch, label_color=WHITE)
            char_boxes.add(b)
        char_boxes.arrange(RIGHT, buff=0.15)
        char_boxes.next_to(init_label, DOWN, buff=0.4)
        self.play(LaggedStart(*[FadeIn(b) for b in char_boxes], lag_ratio=0.12), run_time=0.8)
        self.wait(0.5)

        # Merge 1: l + o → lo
        m1_label = label_text('Merge 1: "l" + "o" → "lo"  (most frequent pair)', color=BLUE_MED)
        m1_label.next_to(char_boxes, DOWN, buff=0.5)
        self.play(FadeIn(m1_label), run_time=0.5)

        merge1_boxes = VGroup()
        for tok in ["lo", "w", "e", "r"]:
            b = rounded_box(0.6, 0.5, fill_color=str(BLUE_MED) + "22",
                            stroke_color=BLUE_MED, label=tok, label_color=WHITE)
            merge1_boxes.add(b)
        merge1_boxes.arrange(RIGHT, buff=0.15)
        merge1_boxes.next_to(m1_label, DOWN, buff=0.35)
        self.play(LaggedStart(*[FadeIn(b) for b in merge1_boxes], lag_ratio=0.12), run_time=0.7)
        self.wait(0.5)

        # Merge 2: lo + w → low
        m2_label = label_text('Merge 2: "lo" + "w" → "low"', color=GREEN_MED)
        m2_label.next_to(merge1_boxes, DOWN, buff=0.4)
        self.play(FadeIn(m2_label), run_time=0.5)

        merge2_boxes = VGroup()
        for tok in ["low", "e", "r"]:
            b = rounded_box(0.7, 0.5, fill_color=str(GREEN_MED) + "22",
                            stroke_color=GREEN_MED, label=tok, label_color=WHITE)
            merge2_boxes.add(b)
        merge2_boxes.arrange(RIGHT, buff=0.15)
        merge2_boxes.next_to(m2_label, DOWN, buff=0.3)
        self.play(LaggedStart(*[FadeIn(b) for b in merge2_boxes], lag_ratio=0.15), run_time=0.6)
        self.wait(0.5)

        # Merge 3: low + er → lower
        m3_label = label_text('Merge 3: "low" + "er" → "lower"  (one token!)', color=PURPLE_MED)
        m3_label.next_to(merge2_boxes, DOWN, buff=0.4)
        self.play(FadeIn(m3_label), run_time=0.5)

        final_box = rounded_box(1.2, 0.55, fill_color=str(PURPLE_MED) + "22",
                                stroke_color=PURPLE_MED, label="lower", label_color=WHITE)
        final_box.next_to(m3_label, DOWN, buff=0.3)
        self.play(FadeIn(final_box), run_time=0.6)
        self.wait(1.5)

        self.fade_all(bpe_title, init_label, char_boxes,
                      m1_label, merge1_boxes, m2_label, merge2_boxes,
                      m3_label, final_box)

        # ── 7. Vocabulary size comparison ─────────────────────────────────────
        vocab_title = body_text("Vocabulary sizes — why BPE hits the sweet spot", color=WHITE)
        vocab_title.to_edge(UP, buff=0.8)
        self.play(Write(vocab_title), run_time=0.7)

        bar_data = [
            ("Chars\n~256",  0.003, RED_MED),
            ("BPE\n~50k",    0.5,   GREEN_MED),
            ("Words\n~100k", 1.0,   ORANGE_MED),
        ]
        max_h = 3.0
        bars_group = VGroup()
        for label_str, ratio, col in bar_data:
            h = max(max_h * ratio, 0.05)
            bar = Rectangle(width=1.2, height=h,
                            fill_color=col, fill_opacity=0.85,
                            stroke_color=WHITE, stroke_width=1)
            bar_label = label_text(label_str, color=col)
            bar_label.next_to(bar, DOWN, buff=0.15)
            group = VGroup(bar, bar_label)
            bars_group.add(group)

        bars_group.arrange(RIGHT, buff=1.2)
        bars_group.shift(DOWN * 0.3)
        # align bars to bottom
        for grp in bars_group:
            grp[0].align_to(bars_group[0][0], DOWN)

        self.play(LaggedStart(*[FadeIn(g) for g in bars_group], lag_ratio=0.25), run_time=1.0)

        sweet_spot = label_text("BPE is the goldilocks zone — small enough to be efficient,\nlarge enough to cover most language", color=GREEN_LIGHT)
        sweet_spot.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(sweet_spot), run_time=0.6)
        self.wait(1.5)

        self.fade_all(vocab_title, bars_group, sweet_spot)

        # ── 8. Special tokens ──────────────────────────────────────────────────
        spec_title = body_text("Special Tokens — structural signals for the model", color=WHITE)
        spec_title.to_edge(UP, buff=0.8)
        self.play(Write(spec_title), run_time=0.7)

        special_tokens = [
            ("[BOS]", "Start of sequence",  BLUE_MED),
            ("[EOS]", "End of sequence",    RED_MED),
            ("[PAD]", "Padding (ignored)",  GREY_MED),
            ("[SEP]", "Separator",          ORANGE_MED),
            ("[UNK]", "Unknown token",      PURPLE_MED),
        ]

        spec_rows = VGroup()
        for tok, meaning, col in special_tokens:
            tok_box = rounded_box(1.3, 0.52, fill_color=str(col) + "33",
                                  stroke_color=col, label=tok, label_color=WHITE)
            arrow = Arrow(tok_box.get_right(),
                          tok_box.get_right() + RIGHT * 0.6,
                          color=GREY_MED, stroke_width=1.5, buff=0.05,
                          max_tip_length_to_length_ratio=0.25)
            desc = label_text(meaning, color=WHITE)
            desc.next_to(arrow.get_end(), RIGHT, buff=0.15)
            row = VGroup(tok_box, arrow, desc)
            spec_rows.add(row)

        spec_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        spec_rows.move_to(ORIGIN)

        self.play(LaggedStart(*[FadeIn(r) for r in spec_rows], lag_ratio=0.15),
                  run_time=1.5)
        self.wait(1.5)

        self.fade_all(spec_title, spec_rows)

        # ── 9. Tokenization surprises ──────────────────────────────────────────
        surp_title = body_text("Tokenization Surprises — not always what you expect!", color=WHITE)
        surp_title.to_edge(UP, buff=0.8)
        self.play(Write(surp_title), run_time=0.7)

        surprises = [
            ('"ChatGPT"',      ["Chat", "G", "PT"],          [BLUE_MED, GREEN_MED, ORANGE_MED]),
            ('"tokenization"', ["token", "ization"],          [PURPLE_MED, RED_MED]),
            ('"99999"',        ["999", "99"],                 [YELLOW_MED, ORANGE_MED]),
        ]

        all_rows = VGroup()
        for word_str, parts, cols in surprises:
            word_lbl = label_text(word_str + "  →", color=WHITE)
            part_boxes = VGroup()
            for part, col in zip(parts, cols):
                b = rounded_box(max(len(part) * 0.2 + 0.3, 0.7), 0.5,
                                fill_color=str(col) + "22",
                                stroke_color=col, label=part, label_color=WHITE)
                part_boxes.add(b)
            part_boxes.arrange(RIGHT, buff=0.1)
            row = VGroup(word_lbl, part_boxes)
            row.arrange(RIGHT, buff=0.25)
            all_rows.add(row)

        all_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        all_rows.move_to(ORIGIN)

        self.play(LaggedStart(*[FadeIn(r) for r in all_rows], lag_ratio=0.2),
                  run_time=1.2)

        note = label_text("Words split at unexpected boundaries — affects spelling, arithmetic, multilingual tasks", color=GREY_LIGHT)
        note.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(note), run_time=0.6)
        self.wait(1.5)

        self.fade_all(surp_title, all_rows, note)

        # ── 10. Token IDs — the final output ──────────────────────────────────
        ids_title = body_text("Token IDs — the final numeric output", color=WHITE)
        ids_title.to_edge(UP, buff=0.8)
        self.play(Write(ids_title), run_time=0.7)

        phrase = body_text('"hello world"', color=WHITE)
        phrase.shift(UP * 1.5)
        self.play(Write(phrase), run_time=0.6)

        arrow_down2 = Arrow(phrase.get_bottom(),
                            phrase.get_bottom() + DOWN * 0.8,
                            color=GREY_MED, stroke_width=2, buff=0.05)
        self.play(Create(arrow_down2), run_time=0.4)

        id_row = VGroup()
        for tok_str, tok_id, col in [("hello", "15496", GREEN_MED), (" world", "995", BLUE_MED)]:
            tok_b = rounded_box(1.1, 0.52, fill_color=str(col) + "22",
                                stroke_color=col, label=tok_str, label_color=WHITE)
            id_b = rounded_box(0.9, 0.52, fill_color=GREY_DARK,
                               stroke_color=col, label=tok_id, label_color=WHITE)
            pair = VGroup(tok_b, id_b)
            pair.arrange(DOWN, buff=0.12)
            id_row.add(pair)
        id_row.arrange(RIGHT, buff=0.5)
        id_row.next_to(arrow_down2.get_end(), DOWN, buff=0.25)

        self.play(LaggedStart(*[FadeIn(p) for p in id_row], lag_ratio=0.2), run_time=0.8)

        result_lbl = label_text("Text: \"hello world\"   →   IDs: [15496, 995]", color=YELLOW_MED)
        result_lbl.next_to(id_row, DOWN, buff=0.5)
        self.play(FadeIn(result_lbl), run_time=0.6)

        pipeline_lbl = label_text("These IDs are fed directly into the embedding table — the model never sees raw text", color=GREY_LIGHT)
        pipeline_lbl.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(pipeline_lbl), run_time=0.6)
        self.wait(2)

        self.fade_all(ids_title, phrase, arrow_down2, id_row, result_lbl, pipeline_lbl)
