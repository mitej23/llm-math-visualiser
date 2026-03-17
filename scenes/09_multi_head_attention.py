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
                        fill_color=str(col) + "33", stroke_color=col,
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
        self.fade_all(sh_box, vs_text, mh_group)

        # ── 2. Expert panel analogy ────────────────────────────────────────────
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
                            fill_color=str(col) + "22",
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
        self.play(LaggedStart(*[Create(a) for a in step_arrows], lag_ratio=0.1),
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
        self.fade_all(arch_title, step_boxes, step_arrows, gqa_note)

        # ── 5. The problem with single-head attention ─────────────────────────
        prob_title = body_text("The Problem with Single-Head Attention", color=WHITE)
        prob_title.to_edge(UP, buff=0.6)
        self.play(Write(prob_title), run_time=0.6)

        # Show a single attention pattern — only one relationship visible
        sentence_txt = body_text(
            '"The lawyer defended her client because she believed in justice."',
            color=WHITE,
        )
        sentence_txt.shift(UP * 2.0)
        sentence_txt.scale(0.75)
        self.play(Write(sentence_txt), run_time=0.8)

        # Relationships that single head must juggle
        relations = [
            ("she → lawyer",             BLUE_MED,  "coreference"),
            ("defended → client",         GREEN_MED, "verb-object"),
            ("lawyer → defended",         ORANGE_MED,"subject-verb"),
            ("believed → justice",        PURPLE_MED,"semantic"),
        ]

        rel_boxes = VGroup()
        for rel, col, rtype in relations:
            b = rounded_box(2.8, 0.65,
                            fill_color=str(col) + "22",
                            stroke_color=col, label=rel, label_color=col)
            t = label_text(rtype, color=GREY_MED)
            t.next_to(b, RIGHT, buff=0.3)
            rel_boxes.add(VGroup(b, t))

        rel_boxes.arrange(DOWN, buff=0.22)
        rel_boxes.shift(DOWN * 0.3 + LEFT * 1.2)

        single_note = label_text(
            "Single head must average\nall four — captures none well",
            color=RED_MED,
        )
        single_note.next_to(rel_boxes, RIGHT, buff=0.6)

        self.play(LaggedStart(*[FadeIn(r) for r in rel_boxes], lag_ratio=0.15),
                  run_time=1.2)
        self.play(FadeIn(single_note), run_time=0.6)
        self.wait(1.5)
        self.fade_all(prob_title, sentence_txt, rel_boxes, single_note)

        # ── 6. Multiple heads, multiple perspectives ───────────────────────────
        multi_title = body_text("4 Heads — 4 Different Perspectives", color=WHITE)
        multi_title.to_edge(UP, buff=0.6)
        self.play(Write(multi_title), run_time=0.6)

        head_patterns = [
            ("Head 1", BLUE_MED,    "Subject-Verb",          "lawyer ↔ defended"),
            ("Head 2", GREEN_MED,   "Pronoun-Antecedent",     "she ↔ lawyer"),
            ("Head 3", ORANGE_MED,  "Modifier-Noun",          "her ↔ client"),
            ("Head 4", PURPLE_MED,  "Positional (local)",     "adjacent tokens"),
        ]

        head_cols_grp = VGroup()
        for hname, col, pattern, example in head_patterns:
            title_t = body_text(hname, color=col)
            pat_t   = label_text(pattern, color=WHITE)
            ex_t    = label_text(example, color=GREY_LIGHT)
            pat_t.next_to(title_t, DOWN, buff=0.12)
            ex_t.next_to(pat_t, DOWN, buff=0.1)
            content = VGroup(title_t, pat_t, ex_t)
            bg = SurroundingRectangle(content, color=str(col) + "55",
                                      fill_color=str(col) + "11",
                                      fill_opacity=1, buff=0.25,
                                      corner_radius=0.14)
            head_cols_grp.add(VGroup(bg, content))

        head_cols_grp.arrange(RIGHT, buff=0.4)
        head_cols_grp.move_to(ORIGIN + DOWN * 0.2)

        self.play(LaggedStart(*[FadeIn(h) for h in head_cols_grp], lag_ratio=0.2),
                  run_time=1.5)

        multi_note = label_text(
            "Each head specialises — together they capture every important relationship",
            color=YELLOW_MED,
        )
        multi_note.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(multi_note), run_time=0.6)
        self.wait(2)
        self.fade_all(multi_title, head_cols_grp, multi_note)

        # ── 7. How heads are sized — dimension split ───────────────────────────
        dim_title = body_text("How Heads Are Sized — Splitting d_model", color=WHITE)
        dim_title.to_edge(UP, buff=0.6)
        self.play(Write(dim_title), run_time=0.6)

        # d_model = 512 bar
        dmodel_box = rounded_box(10.0, 0.8,
                                  fill_color=str(BLUE_MED) + "33",
                                  stroke_color=BLUE_MED,
                                  label="d_model = 512 dimensions",
                                  label_color=BLUE_MED)
        dmodel_box.shift(UP * 2.0)
        self.play(FadeIn(dmodel_box), run_time=0.5)

        # Split into 8 heads of 64 each
        head_colors_dim = [BLUE_MED, GREEN_MED, ORANGE_MED, RED_MED,
                           PURPLE_MED, YELLOW_MED, BLUE_LIGHT, GREEN_LIGHT]
        head_boxes_dim = VGroup()
        for i, col in enumerate(head_colors_dim):
            b = rounded_box(1.1, 0.75,
                            fill_color=str(col) + "33",
                            stroke_color=col,
                            label=f"H{i+1}\n64d", label_color=col)
            head_boxes_dim.add(b)

        head_boxes_dim.arrange(RIGHT, buff=0.12)
        head_boxes_dim.move_to(ORIGIN + DOWN * 0.1)

        split_lbl = label_text("512 ÷ 8 heads = 64 dimensions each", color=GREY_LIGHT)
        split_lbl.shift(DOWN * 0.95)

        # Arrows from dmodel to heads
        split_arrows = VGroup()
        for b in head_boxes_dim:
            a = Arrow(dmodel_box.get_bottom(), b.get_top(),
                      color=GREY_MED, buff=0.06, stroke_width=1.0,
                      max_tip_length_to_length_ratio=0.25)
            split_arrows.add(a)

        self.play(LaggedStart(*[Create(a) for a in split_arrows], lag_ratio=0.05),
                  run_time=0.9)
        self.play(LaggedStart(*[FadeIn(b) for b in head_boxes_dim], lag_ratio=0.06),
                  run_time=0.9)
        self.play(FadeIn(split_lbl), run_time=0.5)

        cost_note = label_text(
            "Same total compute as one 512-dim head — but 8 different learned perspectives",
            color=YELLOW_MED,
        )
        cost_note.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(cost_note), run_time=0.6)
        self.wait(2)
        self.fade_all(dim_title, dmodel_box, split_arrows, head_boxes_dim,
                      split_lbl, cost_note)

        # ── 8. Concatenation and projection ───────────────────────────────────
        concat_title = body_text("Concatenate Outputs, Then Project", color=WHITE)
        concat_title.to_edge(UP, buff=0.6)
        self.play(Write(concat_title), run_time=0.6)

        # 8 head output boxes → concat → projection
        h_out_colors = [BLUE_MED, GREEN_MED, ORANGE_MED, RED_MED,
                        PURPLE_MED, YELLOW_MED, BLUE_LIGHT, GREEN_LIGHT]
        h_out_boxes = VGroup()
        for i, col in enumerate(h_out_colors):
            b = rounded_box(0.9, 0.65,
                            fill_color=str(col) + "33",
                            stroke_color=col,
                            label=f"out{i+1}", label_color=col)
            h_out_boxes.add(b)

        h_out_boxes.arrange(RIGHT, buff=0.1)
        h_out_boxes.shift(UP * 1.4)

        concat_box = rounded_box(8.5, 0.7,
                                  fill_color=str(PURPLE_MED) + "22",
                                  stroke_color=PURPLE_MED,
                                  label="Concatenated: 512 dimensions",
                                  label_color=PURPLE_MED)
        concat_box.shift(DOWN * 0.1)

        proj_box = rounded_box(4.0, 0.7,
                               fill_color=str(ORANGE_MED) + "22",
                               stroke_color=ORANGE_MED,
                               label="Linear Projection (W_O): 512 → 512",
                               label_color=ORANGE_MED)
        proj_box.shift(DOWN * 1.3)

        concat_arrows = VGroup()
        for b in h_out_boxes:
            a = Arrow(b.get_bottom(), concat_box.get_top(),
                      color=GREY_MED, buff=0.05, stroke_width=1.0,
                      max_tip_length_to_length_ratio=0.3)
            concat_arrows.add(a)

        proj_arrow = Arrow(concat_box.get_bottom(), proj_box.get_top(),
                           color=ORANGE_MED, buff=0.05, stroke_width=1.8,
                           max_tip_length_to_length_ratio=0.2)

        self.play(LaggedStart(*[FadeIn(b) for b in h_out_boxes], lag_ratio=0.06),
                  run_time=0.8)
        self.play(LaggedStart(*[Create(a) for a in concat_arrows], lag_ratio=0.04),
                  run_time=0.7)
        self.play(FadeIn(concat_box), run_time=0.5)
        self.play(Create(proj_arrow), FadeIn(proj_box), run_time=0.5)

        proj_note = label_text(
            "W_O learns how to best combine the 8 perspectives into one output",
            color=GREY_LIGHT,
        )
        proj_note.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(proj_note), run_time=0.6)
        self.wait(2)
        self.fade_all(concat_title, h_out_boxes, concat_arrows, concat_box,
                      proj_arrow, proj_box, proj_note)

        # ── 9. Ablation study results ──────────────────────────────────────────
        ablate_title = body_text("What Happens When You Remove Heads?", color=WHITE)
        ablate_title.to_edge(UP, buff=0.6)
        self.play(Write(ablate_title), run_time=0.6)

        # Bar chart: performance drop when each head is removed
        head_importance = [0.8, 4.2, 1.1, 0.3, 3.9, 0.5, 0.2, 2.1]
        ablate_colors = [BLUE_MED, GREEN_MED, ORANGE_MED, RED_MED,
                         PURPLE_MED, YELLOW_MED, BLUE_LIGHT, GREEN_LIGHT]
        ablate_bars = VGroup()

        for i, (drop, col) in enumerate(zip(head_importance, ablate_colors)):
            h = drop * 0.55
            bar = Rectangle(width=0.75, height=max(h, 0.05),
                            fill_color=col, fill_opacity=0.85,
                            stroke_color=WHITE, stroke_width=1)
            drop_lbl = label_text(f"-{drop}%", color=WHITE)
            drop_lbl.next_to(bar, UP, buff=0.08)
            h_lbl = label_text(f"H{i+1}", color=GREY_LIGHT)
            h_lbl.next_to(bar, DOWN, buff=0.12)
            grp = VGroup(bar, drop_lbl, h_lbl)
            grp.shift(RIGHT * i * 1.05)
            ablate_bars.add(grp)

        ablate_bars.move_to(ORIGIN + DOWN * 0.3)

        # Highlight critical heads (H2, H5)
        critical_hl = SurroundingRectangle(
            VGroup(ablate_bars[1], ablate_bars[4]),
            color=YELLOW_MED, buff=0.12, corner_radius=0.1, stroke_width=2
        )
        critical_lbl = label_text("Critical heads —\nremove them and quality drops sharply",
                                  color=YELLOW_MED)
        critical_lbl.to_edge(RIGHT, buff=0.4)

        self.play(LaggedStart(*[FadeIn(b) for b in ablate_bars], lag_ratio=0.08),
                  run_time=1.2)
        self.play(Create(critical_hl), FadeIn(critical_lbl), run_time=0.7)

        ablate_note = label_text(
            "Most heads are redundant — a few 'induction heads' do critical work",
            color=GREY_LIGHT,
        )
        ablate_note.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(ablate_note), run_time=0.6)
        self.wait(2)
        self.fade_all(ablate_title, ablate_bars, critical_hl, critical_lbl, ablate_note)

        # ── 10. Grouped Query Attention (GQA) ─────────────────────────────────
        gqa_title = body_text("Grouped Query Attention (GQA)", color=WHITE)
        gqa_title.to_edge(UP, buff=0.6)
        self.play(Write(gqa_title), run_time=0.6)

        # Full MHA (left)
        mha_lbl = body_text("Full MHA", color=BLUE_MED)
        mha_lbl.shift(LEFT * 4.0 + UP * 2.0)

        mha_q_boxes = VGroup(*[
            rounded_box(0.55, 0.5,
                        fill_color=str(BLUE_MED) + "33", stroke_color=BLUE_MED,
                        label=f"Q{i+1}", label_color=BLUE_MED)
            for i in range(4)
        ])
        mha_k_boxes = VGroup(*[
            rounded_box(0.55, 0.5,
                        fill_color=str(GREEN_MED) + "33", stroke_color=GREEN_MED,
                        label=f"K{i+1}", label_color=GREEN_MED)
            for i in range(4)
        ])
        mha_v_boxes = VGroup(*[
            rounded_box(0.55, 0.5,
                        fill_color=str(ORANGE_MED) + "33", stroke_color=ORANGE_MED,
                        label=f"V{i+1}", label_color=ORANGE_MED)
            for i in range(4)
        ])

        mha_q_boxes.arrange(RIGHT, buff=0.08)
        mha_k_boxes.arrange(RIGHT, buff=0.08)
        mha_v_boxes.arrange(RIGHT, buff=0.08)
        mha_q_row = VGroup(label_text("Q:", color=BLUE_MED), mha_q_boxes)
        mha_k_row = VGroup(label_text("K:", color=GREEN_MED), mha_k_boxes)
        mha_v_row = VGroup(label_text("V:", color=ORANGE_MED), mha_v_boxes)

        for row in [mha_q_row, mha_k_row, mha_v_row]:
            row.arrange(RIGHT, buff=0.2)

        mha_stack = VGroup(mha_q_row, mha_k_row, mha_v_row)
        mha_stack.arrange(DOWN, buff=0.3)
        mha_stack.shift(LEFT * 3.8 + DOWN * 0.3)

        # GQA (right) — 4 Q heads, 2 KV heads
        gqa_lbl = body_text("GQA (LLaMA 3 style)", color=GREEN_MED)
        gqa_lbl.shift(RIGHT * 2.2 + UP * 2.0)

        gqa_q_boxes = VGroup(*[
            rounded_box(0.55, 0.5,
                        fill_color=str(BLUE_MED) + "33", stroke_color=BLUE_MED,
                        label=f"Q{i+1}", label_color=BLUE_MED)
            for i in range(4)
        ])
        gqa_k_boxes = VGroup(*[
            rounded_box(0.55, 0.5,
                        fill_color=str(GREEN_MED) + "33", stroke_color=GREEN_MED,
                        label=f"K{i+1}", label_color=GREEN_MED)
            for i in range(2)
        ])
        gqa_v_boxes = VGroup(*[
            rounded_box(0.55, 0.5,
                        fill_color=str(ORANGE_MED) + "33", stroke_color=ORANGE_MED,
                        label=f"V{i+1}", label_color=ORANGE_MED)
            for i in range(2)
        ])

        gqa_q_boxes.arrange(RIGHT, buff=0.08)
        gqa_k_boxes.arrange(RIGHT, buff=0.08)
        gqa_v_boxes.arrange(RIGHT, buff=0.08)
        gqa_q_row = VGroup(label_text("Q:", color=BLUE_MED), gqa_q_boxes)
        gqa_k_row = VGroup(label_text("K:", color=GREEN_MED), gqa_k_boxes)
        gqa_v_row = VGroup(label_text("V:", color=ORANGE_MED), gqa_v_boxes)

        for row in [gqa_q_row, gqa_k_row, gqa_v_row]:
            row.arrange(RIGHT, buff=0.2)

        gqa_stack = VGroup(gqa_q_row, gqa_k_row, gqa_v_row)
        gqa_stack.arrange(DOWN, buff=0.3)
        gqa_stack.shift(RIGHT * 2.2 + DOWN * 0.3)

        vs_gqa = body_text("vs.", color=GREY_MED)
        vs_gqa.move_to(ORIGIN + DOWN * 0.3)

        self.play(FadeIn(mha_lbl), FadeIn(gqa_lbl), FadeIn(vs_gqa), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(r) for r in mha_stack], lag_ratio=0.2),
                  LaggedStart(*[FadeIn(r) for r in gqa_stack], lag_ratio=0.2),
                  run_time=1.2)

        gqa_benefit = label_text(
            "GQA: 4 Query heads share 2 KV heads → 2× smaller KV cache, near-identical quality\n"
            "LLaMA 3 70B: 64 Q heads, 8 KV heads → 8× KV cache reduction",
            color=YELLOW_MED,
        )
        gqa_benefit.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(gqa_benefit), run_time=0.7)
        self.wait(2)
        self.fade_all(gqa_title, mha_lbl, gqa_lbl, vs_gqa,
                      mha_stack, gqa_stack, gqa_benefit)

        # ── Closing recap ─────────────────────────────────────────────────────
        recap_title = body_text("Multi-Head Attention — Key Ideas", color=WHITE)
        recap_title.to_edge(UP, buff=0.6)
        self.play(Write(recap_title), run_time=0.6)

        recap = [
            ("Multiple heads",  BLUE_MED,   "Each captures a different relationship type"),
            ("Head size",       GREEN_MED,  "d_head = d_model / n_heads  (usually 64–128)"),
            ("Concat + project",ORANGE_MED, "Combine all head outputs via W_O"),
            ("Specialisation",  PURPLE_MED, "Subject-verb, coreference, semantics, position..."),
            ("Induction heads", RED_MED,    "Enable in-context few-shot learning"),
            ("GQA",             YELLOW_MED, "Share K/V across Q groups → saves GPU memory"),
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
        self.fade_all(recap_title, recap_rows)
