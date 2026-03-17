"""
Scene 21 — RLHF Overview
Run: manim -pql 21_rlhf_overview.py RLHFOverviewScene
"""

from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class RLHFOverviewScene(LLMScene):
    def construct(self):
        title = self.show_title("RLHF Overview", "The Full Alignment Pipeline")
        self.wait(0.5)
        self.fade_all(title)

        # ── 1. The alignment problem — before/after RLHF ─────────────────────
        align_title = body_text("The Alignment Problem: base models are powerful but unsafe",
                                color=WHITE)
        align_title.to_edge(UP, buff=0.6)
        self.play(Write(align_title), run_time=0.7)

        before_box = rounded_box(5.5, 1.5, fill_color=str(RED_MED) + "22",
                                 stroke_color=RED_MED,
                                 label="Base GPT-3 (175B)\nCompletes harmful prompts\nHallucinates confidently\nIgnores actual user intent",
                                 label_color=WHITE)
        before_box.shift(LEFT * 3.2 + DOWN * 0.2)

        after_box = rounded_box(5.5, 1.5, fill_color=GREEN_DARK,
                                stroke_color=GREEN_MED,
                                label="InstructGPT (1.3B)\nFollows instructions\nRefuses harmful requests\nAdmits uncertainty",
                                label_color=GREEN_LIGHT)
        after_box.shift(RIGHT * 3.2 + DOWN * 0.2)

        rlhf_arrow = Arrow(before_box.get_right(), after_box.get_left(),
                           color=YELLOW_MED, buff=0.05, stroke_width=2.5,
                           max_tip_length_to_length_ratio=0.18)
        rlhf_lbl = label_text("RLHF", color=YELLOW_MED)
        rlhf_lbl.next_to(rlhf_arrow, UP, buff=0.15)

        self.play(FadeIn(before_box), run_time=0.6)
        self.play(Create(rlhf_arrow), FadeIn(rlhf_lbl), run_time=0.5)
        self.play(FadeIn(after_box), run_time=0.6)

        align_note = label_text(
            "1.3B aligned model preferred over 175B base — alignment > scale (Ouyang et al., 2022)",
            color=GREY_LIGHT,
        )
        align_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(align_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(align_title, before_box, rlhf_arrow, rlhf_lbl, after_box, align_note)

        # ── 2. The three phases ────────────────────────────────────────────────
        phases_title = body_text("Three phases: SFT → Reward Model → RL Fine-Tuning",
                                 color=WHITE)
        phases_title.to_edge(UP, buff=0.6)
        self.play(Write(phases_title), run_time=0.7)

        phase_data = [
            ("Phase 1\nSFT",
             GREEN_MED,
             "Collect human demos\nFine-tune base model\nResult: SFT model"),
            ("Phase 2\nReward Model",
             ORANGE_MED,
             "Generate responses\nHumans rank them\nTrain reward network"),
            ("Phase 3\nRL + PPO",
             PURPLE_MED,
             "Sample prompts\nScore with RM\nUpdate via PPO"),
        ]

        phase_boxes = VGroup()
        for lbl, col, details in phase_data:
            header = body_text(lbl, color=col)
            detail_txt = label_text(details, color=WHITE)
            detail_txt.next_to(header, DOWN, buff=0.2)
            content = VGroup(header, detail_txt)
            bg = SurroundingRectangle(content, color=col, fill_color=str(col) + "11",
                                      fill_opacity=1, buff=0.3, corner_radius=0.12)
            phase_boxes.add(VGroup(bg, content))

        phase_boxes.arrange(RIGHT, buff=0.6)
        phase_boxes.scale_to_fit_width(12.5)
        phase_boxes.move_to(ORIGIN + DOWN * 0.2)

        phase_arrows = VGroup(*[
            Arrow(phase_boxes[i][0].get_right(), phase_boxes[i + 1][0].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.18)
            for i in range(2)
        ])

        self.play(LaggedStart(*[FadeIn(p) for p in phase_boxes], lag_ratio=0.3),
                  run_time=1.5)
        self.play(LaggedStart(*[Create(a) for a in phase_arrows], lag_ratio=0.3),
                  run_time=0.6)
        self.wait(1.2)
        self.fade_all(phases_title, phase_boxes, phase_arrows)

        # ── 3. PPO algorithm overview ─────────────────────────────────────────
        ppo_title = body_text("PPO: sample → score → compute advantage → clip → update",
                              color=WHITE)
        ppo_title.to_edge(UP, buff=0.6)
        self.play(Write(ppo_title), run_time=0.7)

        ppo_steps = [
            ("Sample\nprompts",   GREY_LIGHT,  "Batch from\nprompt dataset"),
            ("Generate\nresponse", BLUE_MED,    "RL policy runs\nat temperature 1"),
            ("Score with\nRM",     ORANGE_MED,  "Reward model\noutputs scalar r"),
            ("Compute\nadvantage", YELLOW_MED,  "A = r - V(state)\nbetter than avg?"),
            ("Clip +\nUpdate",     PURPLE_MED,  "ratio bounded to\n[1-e, 1+e]"),
        ]

        ppo_boxes = VGroup()
        for lbl, col, note in ppo_steps:
            b = rounded_box(2.2, 0.85, fill_color=str(col) + "22",
                            stroke_color=col, label=lbl, label_color=WHITE)
            n = label_text(note, color=GREY_LIGHT)
            n.next_to(b, DOWN, buff=0.18)
            ppo_boxes.add(VGroup(b, n))

        ppo_boxes.arrange(RIGHT, buff=0.45)
        ppo_boxes.scale_to_fit_width(13)
        ppo_boxes.move_to(ORIGIN + UP * 0.3)

        ppo_arrows = VGroup(*[
            Arrow(ppo_boxes[i][0].get_right(), ppo_boxes[i + 1][0].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.2)
            for i in range(4)
        ])

        loop_back = CurvedArrow(
            ppo_boxes[4][0].get_bottom(),
            ppo_boxes[0][0].get_bottom(),
            angle=-TAU / 6, color=PURPLE_MED, stroke_width=1.5,
        )
        loop_lbl = label_text("next batch", color=PURPLE_MED)
        loop_lbl.next_to(loop_back, DOWN, buff=0.1)

        self.play(LaggedStart(*[FadeIn(b) for b in ppo_boxes], lag_ratio=0.15),
                  run_time=1.3)
        self.play(LaggedStart(*[Create(a) for a in ppo_arrows], lag_ratio=0.1),
                  run_time=0.6)
        self.play(Create(loop_back), FadeIn(loop_lbl), run_time=0.6)
        self.wait(1.5)
        self.fade_all(ppo_title, ppo_boxes, ppo_arrows, loop_back, loop_lbl)

        # ── 4. PPO clipping ───────────────────────────────────────────────────
        clip_title = body_text("PPO clipping: bound policy updates to prevent instability",
                               color=WHITE)
        clip_title.to_edge(UP, buff=0.6)
        self.play(Write(clip_title), run_time=0.7)

        # Show the ratio concept
        ratio_boxes = VGroup(
            rounded_box(3.5, 0.75, fill_color=str(RED_MED) + "22",
                        stroke_color=RED_MED,
                        label="Without clipping\nratio can go to 5x, 10x...\nUnstable huge updates!",
                        label_color=WHITE),
            rounded_box(3.5, 0.75, fill_color=GREEN_DARK,
                        stroke_color=GREEN_MED,
                        label="With clipping (eps=0.2)\nratio bounded to [0.8, 1.2]\nStable conservative steps",
                        label_color=GREEN_LIGHT),
        )
        ratio_boxes.arrange(RIGHT, buff=1.5)
        ratio_boxes.move_to(ORIGIN + UP * 0.6)

        vs_lbl = body_text("vs", color=GREY_MED)
        vs_lbl.move_to(ORIGIN + UP * 0.6)

        self.play(FadeIn(ratio_boxes[0]), run_time=0.5)
        self.play(FadeIn(vs_lbl), run_time=0.3)
        self.play(FadeIn(ratio_boxes[1]), run_time=0.5)

        # Visualise the clipping range
        clip_range = rounded_box(5.0, 0.65, fill_color=str(BLUE_MED) + "22",
                                 stroke_color=BLUE_MED,
                                 label="ratio r(theta)  clipped to  [ 1 - epsilon,  1 + epsilon ]",
                                 label_color=BLUE_LIGHT)
        clip_range.move_to(ORIGIN + DOWN * 0.5)

        formula_note = label_text(
            "L_CLIP = min( r * A,  clip(r, 1-e, 1+e) * A )   — take the more conservative bound",
            color=ORANGE_MED,
        )
        formula_note.to_edge(DOWN, buff=0.55)

        llm_note = label_text(
            "For LLMs: 32k token vocab means huge action space — clipping essential for stability",
            color=GREY_MED,
        )
        llm_note.to_edge(DOWN, buff=0.25)

        self.play(FadeIn(clip_range), run_time=0.5)
        self.play(FadeIn(formula_note), FadeIn(llm_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(clip_title, ratio_boxes, vs_lbl, clip_range, formula_note, llm_note)

        # ── 5. The RL loop diagram ─────────────────────────────────────────────
        loop_title = body_text("The Phase 3 RL Training Loop:", color=WHITE)
        loop_title.to_edge(UP, buff=0.6)
        self.play(Write(loop_title), run_time=0.6)

        nodes = [
            ("Prompt\nbatch",         GREY_LIGHT,  [-5.0,  0.5, 0]),
            ("RL Policy\ngenerates",  BLUE_MED,    [-2.0,  0.5, 0]),
            ("Reward\nModel scores",  ORANGE_MED,  [ 1.0,  0.5, 0]),
            ("KL\nPenalty",           RED_MED,     [ 1.0, -1.2, 0]),
            ("Total\nReward",         YELLOW_MED,  [ 4.0,  0.5, 0]),
            ("PPO\nUpdate",           PURPLE_MED,  [ 4.0, -1.2, 0]),
        ]

        node_boxes = {}
        node_group = VGroup()
        for name, col, pos in nodes:
            b = rounded_box(1.6, 0.75, fill_color=str(col) + "22",
                            stroke_color=col, label=name, label_color=WHITE)
            b.move_to(pos)
            node_boxes[name] = b
            node_group.add(b)

        # Reference policy box (frozen)
        ref_box = rounded_box(1.8, 0.65, fill_color=GREY_DARK,
                              stroke_color=GREY_MED,
                              label="Reference\nPolicy (frozen)", label_color=GREY_MED)
        ref_box.move_to([-2.0, -1.2, 0])
        node_group.add(ref_box)

        node_group.scale_to_fit_width(13.5)
        node_group.move_to(ORIGIN + DOWN * 0.2)

        self.play(LaggedStart(*[FadeIn(b) for b in node_group], lag_ratio=0.1),
                  run_time=1.5)

        # Draw edges
        def quick_arrow(a, b, col=GREY_MED):
            return Arrow(a.get_right(), b.get_left(), color=col, buff=0.05,
                         stroke_width=1.5, max_tip_length_to_length_ratio=0.15)

        e1 = quick_arrow(node_boxes["Prompt\nbatch"],        node_boxes["RL Policy\ngenerates"])
        e2 = quick_arrow(node_boxes["RL Policy\ngenerates"], node_boxes["Reward\nModel scores"])
        e3 = quick_arrow(node_boxes["Reward\nModel scores"], node_boxes["Total\nReward"])
        e4 = Arrow(node_boxes["KL\nPenalty"].get_right(),
                   node_boxes["Total\nReward"].get_bottom(),
                   color=RED_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        e5 = Arrow(node_boxes["Total\nReward"].get_bottom(),
                   node_boxes["PPO\nUpdate"].get_top(),
                   color=YELLOW_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        # PPO updates RL policy
        e6 = CurvedArrow(
            node_boxes["PPO\nUpdate"].get_left(),
            node_boxes["RL Policy\ngenerates"].get_bottom(),
            angle=TAU / 5, color=PURPLE_MED, stroke_width=1.5,
        )
        # Ref policy → KL
        e7 = Arrow(ref_box.get_right(),
                   node_boxes["KL\nPenalty"].get_left(),
                   color=GREY_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)
        # RL policy → KL
        e8 = Arrow(node_boxes["RL Policy\ngenerates"].get_bottom(),
                   node_boxes["KL\nPenalty"].get_top(),
                   color=GREY_MED, buff=0.05, stroke_width=1.5,
                   max_tip_length_to_length_ratio=0.15)

        edges = VGroup(e1, e2, e3, e4, e5, e6, e7, e8)
        self.play(LaggedStart(*[Create(e) for e in edges], lag_ratio=0.08),
                  run_time=1.5)
        self.wait(1.2)
        self.fade_all(loop_title, node_group, edges)

        # ── 6. DPO as an alternative to RLHF ─────────────────────────────────
        dpo_title = body_text("DPO: same goal, one less stage, no RL algorithm needed",
                              color=WHITE)
        dpo_title.to_edge(UP, buff=0.6)
        self.play(Write(dpo_title), run_time=0.7)

        rlhf_chain = VGroup(
            rounded_box(2.4, 0.75, fill_color=str(BLUE_MED) + "22",
                        stroke_color=BLUE_MED, label="SFT\nmodel", label_color=BLUE_LIGHT),
            rounded_box(2.4, 0.75, fill_color=str(ORANGE_MED) + "22",
                        stroke_color=ORANGE_MED, label="Train\nReward Model", label_color=WHITE),
            rounded_box(2.4, 0.75, fill_color=str(PURPLE_MED) + "22",
                        stroke_color=PURPLE_MED, label="PPO +\nKL penalty", label_color=WHITE),
            rounded_box(2.4, 0.75, fill_color=GREEN_DARK,
                        stroke_color=GREEN_MED, label="Aligned\nLLM", label_color=GREEN_LIGHT),
        )
        rlhf_chain.arrange(RIGHT, buff=0.4)
        rlhf_chain.move_to(ORIGIN + UP * 1.2)

        rlhf_arrows = VGroup(*[
            Arrow(rlhf_chain[i].get_right(), rlhf_chain[i + 1].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.18)
            for i in range(3)
        ])
        rlhf_tag = label_text("RLHF: 3 stages", color=GREY_MED)
        rlhf_tag.next_to(rlhf_chain, LEFT, buff=0.3)

        self.play(LaggedStart(*[FadeIn(b) for b in rlhf_chain], lag_ratio=0.15),
                  FadeIn(rlhf_tag), run_time=0.9)
        self.play(LaggedStart(*[Create(a) for a in rlhf_arrows], lag_ratio=0.15),
                  run_time=0.5)

        dpo_chain = VGroup(
            rounded_box(2.4, 0.75, fill_color=str(BLUE_MED) + "22",
                        stroke_color=BLUE_MED, label="SFT\nmodel", label_color=BLUE_LIGHT),
            rounded_box(3.2, 0.75, fill_color=str(GREEN_MED) + "22",
                        stroke_color=GREEN_MED,
                        label="DPO fine-tuning\n(KL implicit in loss)", label_color=GREEN_LIGHT),
            rounded_box(2.4, 0.75, fill_color=GREEN_DARK,
                        stroke_color=GREEN_MED, label="Aligned\nLLM", label_color=GREEN_LIGHT),
        )
        dpo_chain.arrange(RIGHT, buff=0.4)
        dpo_chain.move_to(ORIGIN + DOWN * 0.9)

        dpo_arrows = VGroup(*[
            Arrow(dpo_chain[i].get_right(), dpo_chain[i + 1].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.18)
            for i in range(2)
        ])
        dpo_tag = label_text("DPO: 2 stages", color=GREEN_MED)
        dpo_tag.next_to(dpo_chain, LEFT, buff=0.3)

        self.play(LaggedStart(*[FadeIn(b) for b in dpo_chain], lag_ratio=0.15),
                  FadeIn(dpo_tag), run_time=0.9)
        self.play(LaggedStart(*[Create(a) for a in dpo_arrows], lag_ratio=0.15),
                  run_time=0.4)

        dpo_note = label_text(
            "DPO = simpler, more stable, comparable quality — default for most open-source models",
            color=GREY_MED,
        )
        dpo_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(dpo_note), run_time=0.5)
        self.wait(1.5)
        self.fade_all(dpo_title, rlhf_chain, rlhf_arrows, rlhf_tag,
                      dpo_chain, dpo_arrows, dpo_tag, dpo_note)

        # ── 7. Key InstructGPT result ─────────────────────────────────────────
        result_title = body_text("Key Result from InstructGPT (Ouyang et al., 2022):",
                                 color=WHITE)
        result_title.to_edge(UP, buff=0.6)
        self.play(Write(result_title), run_time=0.7)

        comparison = VGroup()
        models = [
            ("GPT-3 175B\n(base model)",   GREY_MED,  1.3, "100x bigger"),
            ("InstructGPT 1.3B\n(RLHF)",   GREEN_MED, 3.2, "Human preferred ✅"),
        ]
        for name, col, preference_scale, note in models:
            bar = Rectangle(width=preference_scale * 1.5, height=0.7,
                            fill_color=col, fill_opacity=0.8,
                            stroke_color=col, stroke_width=1)
            lbl = label_text(name, color=col)
            lbl.next_to(bar, LEFT, buff=0.2)
            n_lbl = label_text(note, color=col)
            n_lbl.next_to(bar, RIGHT, buff=0.2)
            comparison.add(VGroup(bar, lbl, n_lbl))

        comparison.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        comparison.move_to(ORIGIN + UP * 0.2)

        # Add task breakdown note
        task_note = label_text(
            "Improved across: creative writing  |  open QA  |  summarisation  |  instruction following",
            color=GREY_LIGHT,
        )
        task_note.next_to(comparison, DOWN, buff=0.35)

        headline = body_text(
            "Alignment > Scale: 1.3B aligned beats 175B base",
            color=YELLOW_MED,
        )
        headline.next_to(task_note, DOWN, buff=0.3)

        source = label_text(
            "Source: Ouyang et al., arXiv 2203.02155 (2022)",
            color=GREY_MED,
        )
        source.to_edge(DOWN, buff=0.35)

        self.play(LaggedStart(*[FadeIn(c) for c in comparison], lag_ratio=0.4),
                  run_time=1.0)
        self.play(FadeIn(task_note), run_time=0.5)
        self.play(Write(headline), run_time=0.8)
        self.play(FadeIn(source), run_time=0.5)
        self.wait(1.5)
        self.fade_all(result_title, comparison, task_note, headline, source)

        # ── 8. Modern alignment — beyond InstructGPT ─────────────────────────
        modern_title = body_text("Modern alignment: the field has evolved rapidly since 2022",
                                 color=WHITE)
        modern_title.to_edge(UP, buff=0.6)
        self.play(Write(modern_title), run_time=0.7)

        evol_data = [
            ("InstructGPT\n2022", BLUE_MED,   "RLHF + PPO\nBaseline approach"),
            ("DPO\n2023",         GREEN_MED,  "No RL needed\nSimpler pipeline"),
            ("CAI / RLAIF\n2023", ORANGE_MED, "AI-generated\npreference labels"),
            ("GRPO / DAPO\n2024", PURPLE_MED, "Group-based RL\nNo value model"),
            ("Verifiable\nRewards 2025", YELLOW_MED, "Maths/code:\ncorrect = 1, wrong = 0"),
        ]
        evol_boxes = VGroup()
        for lbl, col, note in evol_data:
            b = rounded_box(2.2, 0.85, fill_color=str(col) + "22",
                            stroke_color=col, label=lbl, label_color=WHITE)
            n = label_text(note, color=GREY_LIGHT)
            n.next_to(b, DOWN, buff=0.18)
            evol_boxes.add(VGroup(b, n))

        evol_boxes.arrange(RIGHT, buff=0.45)
        evol_boxes.scale_to_fit_width(13)
        evol_boxes.move_to(ORIGIN + UP * 0.2)

        evol_arrows = VGroup(*[
            Arrow(evol_boxes[i][0].get_right(), evol_boxes[i + 1][0].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=1.5,
                  max_tip_length_to_length_ratio=0.2)
            for i in range(4)
        ])

        self.play(LaggedStart(*[FadeIn(b) for b in evol_boxes], lag_ratio=0.2),
                  run_time=1.3)
        self.play(LaggedStart(*[Create(a) for a in evol_arrows], lag_ratio=0.15),
                  run_time=0.6)

        core_note = label_text(
            "Core idea constant across all: SFT → preference learning → KL-constrained improvement",
            color=GREY_MED,
        )
        core_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(core_note), run_time=0.5)
        self.wait(2)
        self.fade_all(modern_title, evol_boxes, evol_arrows, core_note)
