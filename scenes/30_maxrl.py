"""
Scene 30 — MaxRL: Maximum Likelihood RL
Run: manim -pql 30_maxrl.py MaxRLScene
"""
from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


class MaxRLScene(LLMScene):
    def construct(self):

        # ── 1. Title ──────────────────────────────────────────────────────────
        title = self.show_title("MaxRL", "Maximum Likelihood Reinforcement Learning")
        self.wait(1.0)
        self.fade_all(title)

        # ── 2. The filter-and-train pipeline ─────────────────────────────────
        pipeline_title = body_text(
            "Core pipeline: generate many — verify — keep correct — train",
            color=WHITE,
        )
        pipeline_title.to_edge(UP, buff=0.6)
        self.play(Write(pipeline_title), run_time=0.8)

        step_data = [
            ("Generate k\nCandidates",   BLUE_MED,    "Model samples\nk responses"),
            ("Verify\nCorrectness",       YELLOW_MED,  "Checker: correct\nor incorrect?"),
            ("Keep Only\nCorrect Ones",   GREEN_MED,   "Discard all\nwrong answers"),
            ("Supervised\nFine-Tuning",   ORANGE_MED,  "Cross-entropy loss\non correct set"),
        ]

        step_boxes = VGroup()
        for lbl, col, note in step_data:
            b = rounded_box(2.4, 0.9, fill_color=str(col) + "22",
                            stroke_color=col, label=lbl, label_color=WHITE)
            n = label_text(note, color=GREY_LIGHT)
            n.next_to(b, DOWN, buff=0.2)
            step_boxes.add(VGroup(b, n))

        step_boxes.arrange(RIGHT, buff=0.55)
        step_boxes.scale_to_fit_width(13.0)
        step_boxes.move_to(ORIGIN + UP * 0.3)

        step_arrows = VGroup(*[
            Arrow(step_boxes[i][0].get_right(), step_boxes[i + 1][0].get_left(),
                  color=GREY_MED, buff=0.05, stroke_width=2,
                  max_tip_length_to_length_ratio=0.18)
            for i in range(3)
        ])

        self.play(LaggedStart(*[FadeIn(b) for b in step_boxes], lag_ratio=0.25),
                  run_time=1.4)
        self.play(LaggedStart(*[Create(a) for a in step_arrows], lag_ratio=0.2),
                  run_time=0.7)

        loop_back = CurvedArrow(
            step_boxes[3][0].get_bottom(),
            step_boxes[0][0].get_bottom(),
            angle=-TAU / 6, color=BLUE_MED, stroke_width=1.5,
        )
        loop_lbl = label_text("next iteration — model improved", color=BLUE_MED)
        loop_lbl.next_to(loop_back, DOWN, buff=0.1)

        self.play(Create(loop_back), FadeIn(loop_lbl), run_time=0.6)

        key_note = label_text(
            "No reward model needed — just a verifier that checks correct / incorrect",
            color=GREY_MED,
        )
        key_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(key_note), run_time=0.5)
        self.wait(1.0)
        self.fade_all(pipeline_title, step_boxes, step_arrows, loop_back, loop_lbl, key_note)

        # ── 3. pass@k visualisation: 8 responses, 3 green, 5 red ─────────────
        passk_title = body_text(
            "pass@k: generate k responses — at least one must be correct",
            color=WHITE,
        )
        passk_title.to_edge(UP, buff=0.6)
        self.play(Write(passk_title), run_time=0.8)

        prompt_box = rounded_box(5.0, 0.65, stroke_color=GREY_MED,
                                 label='Problem: "Solve 3x + 7 = 22.  What is x?"',
                                 label_color=GREY_LIGHT)
        prompt_box.move_to(UP * 2.6)
        self.play(FadeIn(prompt_box), run_time=0.5)

        # 8 response boxes: indices 1,3,6 are correct (green), rest red
        response_states = [
            ("R1\nx = 5", GREEN_MED, True),
            ("R2\nx = 4.5", RED_MED, False),
            ("R3\nx = 5", GREEN_MED, True),
            ("R4\nx = 3", RED_MED, False),
            ("R5\nx = 7", RED_MED, False),
            ("R6\nx = 5", GREEN_MED, True),
            ("R7\nx = 15", RED_MED, False),
            ("R8\nx = 2", RED_MED, False),
        ]

        resp_boxes = VGroup()
        for lbl, col, correct in response_states:
            fill = GREEN_DARK if correct else str(RED_MED) + "22"
            b = rounded_box(1.45, 0.75, fill_color=fill,
                            stroke_color=col, label=lbl, label_color=WHITE)
            resp_boxes.add(b)

        resp_boxes.arrange(RIGHT, buff=0.22)
        resp_boxes.scale_to_fit_width(13.0)
        resp_boxes.move_to(ORIGIN + UP * 0.9)

        self.play(LaggedStart(*[FadeIn(b) for b in resp_boxes], lag_ratio=0.1),
                  run_time=1.2)

        # Label the keep/discard split
        keep_lbl = body_text("Keep  (3 correct)", color=GREEN_MED)
        discard_lbl = body_text("Discard  (5 wrong)", color=RED_MED)
        keep_lbl.move_to(LEFT * 3.5 + DOWN * 0.4)
        discard_lbl.move_to(RIGHT * 3.0 + DOWN * 0.4)
        self.play(FadeIn(keep_lbl), FadeIn(discard_lbl), run_time=0.6)

        passk_note = label_text(
            "pass@8 satisfied — at least 1 of 8 correct.  Those 3 become training data.",
            color=GREY_LIGHT,
        )
        passk_note.move_to(DOWN * 1.4)
        self.play(FadeIn(passk_note), run_time=0.5)

        k_values = label_text(
            "Typical k: 8 (easy tasks)   16-32 (medium)   64 (hard reasoning)",
            color=GREY_MED,
        )
        k_values.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(k_values), run_time=0.5)
        self.wait(1.0)
        self.fade_all(passk_title, prompt_box, resp_boxes, keep_lbl, discard_lbl,
                      passk_note, k_values)

        # ── 4. Why temperature matters for diversity ───────────────────────────
        temp_title = body_text(
            "Temperature controls diversity — diversity enables pass@k",
            color=WHITE,
        )
        temp_title.to_edge(UP, buff=0.6)
        self.play(Write(temp_title), run_time=0.8)

        temp_data = [
            ("Temperature\n= 0.0", GREY_MED,   "Greedy — identical\nresponses every time",
             "pass@k = pass@1\nUseless for MaxRL"),
            ("Temperature\n= 1.0", BLUE_MED,   "Standard diversity\nDifferent paths explored",
             "Most correct samples\nfound per compute"),
            ("Temperature\n= 1.5", ORANGE_MED, "High diversity\nUnusual approaches",
             "May explore better\nbut noisier outputs"),
        ]

        temp_boxes = VGroup()
        for lbl, col, top_note, bot_note in temp_data:
            header = rounded_box(3.2, 0.8, fill_color=str(col) + "22",
                                 stroke_color=col, label=lbl, label_color=WHITE)
            top = label_text(top_note, color=GREY_LIGHT)
            top.next_to(header, DOWN, buff=0.18)
            bot = label_text(bot_note, color=col)
            bot.next_to(top, DOWN, buff=0.15)
            temp_boxes.add(VGroup(header, top, bot))

        temp_boxes.arrange(RIGHT, buff=0.7)
        temp_boxes.scale_to_fit_width(13.0)
        temp_boxes.move_to(ORIGIN + DOWN * 0.1)

        self.play(LaggedStart(*[FadeIn(b) for b in temp_boxes], lag_ratio=0.25),
                  run_time=1.3)

        sweet_spot = label_text(
            "Sweet spot: 0.8 to 1.2 — diverse enough to find correct solutions, "
            "coherent enough for reasoning",
            color=YELLOW_MED,
        )
        sweet_spot.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(sweet_spot), run_time=0.5)
        self.wait(1.0)
        self.fade_all(temp_title, temp_boxes, sweet_spot)

        # ── 5. Compute indexing: correct samples found vs compute spent ────────
        compute_title = body_text(
            "Compute indexing: how many correct samples does your budget buy?",
            color=WHITE,
        )
        compute_title.to_edge(UP, buff=0.6)
        self.play(Write(compute_title), run_time=0.8)

        # Axes frame
        ax_origin = LEFT * 4.5 + DOWN * 2.0
        ax_w = 8.5
        ax_h = 3.8

        x_axis = Arrow(ax_origin, ax_origin + RIGHT * ax_w, color=GREY_LIGHT,
                       buff=0, stroke_width=2, max_tip_length_to_length_ratio=0.04)
        y_axis = Arrow(ax_origin, ax_origin + UP * ax_h, color=GREY_LIGHT,
                       buff=0, stroke_width=2, max_tip_length_to_length_ratio=0.04)

        x_lbl = label_text("Compute Spent (inference FLOPs)", color=GREY_LIGHT)
        x_lbl.next_to(x_axis, DOWN, buff=0.25)
        y_lbl = label_text("Correct Samples Found", color=GREY_LIGHT)
        y_lbl.next_to(y_axis, LEFT, buff=0.15)
        y_lbl.rotate(PI / 2)

        self.play(Create(x_axis), Create(y_axis), FadeIn(x_lbl), FadeIn(y_lbl),
                  run_time=0.7)

        # Three curves representing different model accuracy levels
        # Strong model (high p): steeper rise
        # Weak model (low p): shallower rise
        # Plot as a series of line segments

        def make_curve(p_rate, color, label_str, x_offset=0):
            """p_rate: probability each response is correct."""
            pts = []
            k_vals = [1, 2, 4, 8, 16, 32, 64, 128]
            for k in k_vals:
                pass_k = 1 - (1 - p_rate) ** k
                x = ax_origin[0] + (k / 128) * ax_w + x_offset
                y = ax_origin[1] + pass_k * ax_h
                pts.append([x, y, 0])
            curve = VMobject(color=color, stroke_width=2.5)
            curve.set_points_as_corners(pts)
            lbl = label_text(label_str, color=color)
            lbl.move_to([pts[-1][0] + 0.7, pts[-1][1], 0])
            return curve, lbl

        curve_strong, lbl_strong = make_curve(0.40, GREEN_MED,  "Strong model\n(p=40%)")
        curve_medium, lbl_medium = make_curve(0.15, BLUE_MED,   "Medium model\n(p=15%)")
        curve_weak,   lbl_weak   = make_curve(0.04, ORANGE_MED, "Weak model\n(p=4%)")

        self.play(Create(curve_strong), FadeIn(lbl_strong), run_time=0.7)
        self.play(Create(curve_medium), FadeIn(lbl_medium), run_time=0.7)
        self.play(Create(curve_weak),   FadeIn(lbl_weak),   run_time=0.7)

        insight = label_text(
            "More compute always finds more correct samples — MaxRL makes this trade-off explicit",
            color=GREY_MED,
        )
        insight.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(insight), run_time=0.5)
        self.wait(1.0)
        self.fade_all(compute_title, x_axis, y_axis, x_lbl, y_lbl,
                      curve_strong, lbl_strong, curve_medium, lbl_medium,
                      curve_weak, lbl_weak, insight)

        # ── 6. Comparison: MaxRL needs no reward model, just a verifier ────────
        no_rm_title = body_text(
            "MaxRL replaces the reward model with a simple verifier",
            color=WHITE,
        )
        no_rm_title.to_edge(UP, buff=0.6)
        self.play(Write(no_rm_title), run_time=0.8)

        rlhf_pipeline = VGroup(
            rounded_box(2.1, 0.8, fill_color=str(BLUE_MED) + "22",
                        stroke_color=BLUE_MED, label="Policy\nModel", label_color=WHITE),
            rounded_box(2.1, 0.8, fill_color=str(ORANGE_MED) + "22",
                        stroke_color=ORANGE_MED, label="Reward\nModel", label_color=WHITE),
            rounded_box(2.1, 0.8, fill_color=str(RED_MED) + "22",
                        stroke_color=RED_MED, label="Value\nModel", label_color=WHITE),
            rounded_box(2.1, 0.8, fill_color=str(GREY_MED) + "22",
                        stroke_color=GREY_MED, label="Reference\nPolicy", label_color=GREY_LIGHT),
        )
        rlhf_pipeline.arrange(RIGHT, buff=0.35)
        rlhf_pipeline.move_to(UP * 1.6)

        rlhf_tag = label_text("PPO / RLHF  (4 models in memory)", color=RED_MED)
        rlhf_tag.next_to(rlhf_pipeline, LEFT, buff=0.25)

        maxrl_pipeline = VGroup(
            rounded_box(2.5, 0.8, fill_color=str(BLUE_MED) + "22",
                        stroke_color=BLUE_MED, label="Policy\nModel", label_color=WHITE),
            rounded_box(2.5, 0.8, fill_color=str(GREEN_MED) + "22",
                        stroke_color=GREEN_MED, label="External\nVerifier", label_color=WHITE),
        )
        maxrl_pipeline.arrange(RIGHT, buff=0.5)
        maxrl_pipeline.move_to(DOWN * 0.2)

        maxrl_tag = label_text("MaxRL  (1 model + external verifier)", color=GREEN_MED)
        maxrl_tag.next_to(maxrl_pipeline, LEFT, buff=0.25)

        self.play(LaggedStart(*[FadeIn(b) for b in rlhf_pipeline], lag_ratio=0.15),
                  FadeIn(rlhf_tag), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(b) for b in maxrl_pipeline], lag_ratio=0.2),
                  FadeIn(maxrl_tag), run_time=0.9)

        verifier_note = label_text(
            "Verifier: runs code against test cases / checks math answer — no training needed",
            color=GREY_LIGHT,
        )
        verifier_note.move_to(DOWN * 1.5)
        self.play(FadeIn(verifier_note), run_time=0.5)

        memory_note = label_text(
            "Memory: PPO needs ~4x model size in GPU RAM.  MaxRL needs ~1x.",
            color=GREY_MED,
        )
        memory_note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(memory_note), run_time=0.5)
        self.wait(1.0)
        self.fade_all(no_rm_title, rlhf_pipeline, rlhf_tag, maxrl_pipeline, maxrl_tag,
                      verifier_note, memory_note)

        # ── 7. Why MaxRL works for math/code but not creative tasks ───────────
        domain_title = body_text(
            "MaxRL needs a verifier — it works where correctness is checkable",
            color=WHITE,
        )
        domain_title.to_edge(UP, buff=0.6)
        self.play(Write(domain_title), run_time=0.8)

        works_box = rounded_box(5.5, 2.4, fill_color=GREEN_DARK,
                                stroke_color=GREEN_MED,
                                label="Works well\nMath — compare answer vs ground truth\n"
                                      "Code — run test cases, pass = correct\n"
                                      "Formal proofs — proof checker (Lean / Coq)\n"
                                      "Structured output — schema validation",
                                label_color=GREEN_LIGHT)
        works_box.shift(LEFT * 3.3 + DOWN * 0.2)

        fails_box = rounded_box(5.5, 2.4, fill_color=str(RED_MED) + "22",
                                stroke_color=RED_MED,
                                label="Does not work\nCreative writing — no correct answer\n"
                                      "Open-ended Q&A — no binary verifier\n"
                                      "Helpfulness judgments — subjective\n"
                                      "Nuanced reasoning — no checker",
                                label_color=WHITE)
        fails_box.shift(RIGHT * 3.3 + DOWN * 0.2)

        vs_text = body_text("vs", color=GREY_MED)
        vs_text.move_to(ORIGIN + DOWN * 0.2)

        self.play(FadeIn(works_box), FadeIn(vs_text), FadeIn(fails_box), run_time=0.9)

        boundary = label_text(
            "The boundary of MaxRL = the boundary of automatic verifiability",
            color=YELLOW_MED,
        )
        boundary.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(boundary), run_time=0.5)
        self.wait(1.0)
        self.fade_all(domain_title, works_box, vs_text, fails_box, boundary)

        # ── 8. MaxRL vs GRPO vs PPO comparison table ──────────────────────────
        table_title = body_text(
            "Algorithm comparison: PPO vs GRPO vs MaxRL",
            color=WHITE,
        )
        table_title.to_edge(UP, buff=0.6)
        self.play(Write(table_title), run_time=0.8)

        # Header row
        headers = ["Property", "PPO", "GRPO", "MaxRL"]
        header_colors = [WHITE, BLUE_MED, ORANGE_MED, GREEN_MED]

        rows_data = [
            ("Models needed",    "4",               "3",             "1 + verifier"),
            ("Reward model",     "Required",        "Optional",      "Not needed"),
            ("Value model",      "Required",        "Not needed",    "Not needed"),
            ("Loss function",    "Clipped PG",      "Group PG",      "Cross-entropy"),
            ("Learns from wrong","Yes",             "Yes",           "No (discarded)"),
            ("Stability",        "Moderate",        "High",          "Very high"),
            ("Memory footprint", "Highest",         "High",          "Lowest"),
        ]

        col_w = [3.1, 2.3, 2.3, 2.3]
        row_h = 0.48
        table_origin = LEFT * 5.0 + UP * 1.5

        # Header
        header_group = VGroup()
        x_cursor = 0
        for i, (hdr, col) in enumerate(zip(headers, header_colors)):
            txt = label_text(hdr, color=col)
            txt.move_to(table_origin + RIGHT * (x_cursor + col_w[i] / 2))
            header_group.add(txt)
            x_cursor += col_w[i]
        self.play(FadeIn(header_group), run_time=0.5)

        # Separator line
        sep = Line(table_origin + DOWN * 0.28 + LEFT * 0.1,
                   table_origin + DOWN * 0.28 + RIGHT * (sum(col_w) + 0.1),
                   color=GREY_MED, stroke_width=1)
        self.play(Create(sep), run_time=0.3)

        row_groups = VGroup()
        for row_i, (prop, ppo_val, grpo_val, maxrl_val) in enumerate(rows_data):
            vals = [prop, ppo_val, grpo_val, maxrl_val]
            val_colors = [GREY_LIGHT, BLUE_MED, ORANGE_MED, GREEN_MED]
            y = table_origin[1] - (row_i + 1) * row_h - 0.1
            x_cursor = 0
            row_g = VGroup()
            for i, (val, col) in enumerate(zip(vals, val_colors)):
                txt = label_text(val, color=col)
                txt.move_to([table_origin[0] + x_cursor + col_w[i] / 2, y, 0])
                row_g.add(txt)
                x_cursor += col_w[i]
            row_groups.add(row_g)

        self.play(LaggedStart(*[FadeIn(r) for r in row_groups], lag_ratio=0.12),
                  run_time=1.5)
        self.wait(1.0)
        self.fade_all(table_title, header_group, sep, row_groups)

        # ── 9. Real numbers: k values and compute efficiency ──────────────────
        numbers_title = body_text(
            "Real numbers: how k and accuracy interact",
            color=WHITE,
        )
        numbers_title.to_edge(UP, buff=0.6)
        self.play(Write(numbers_title), run_time=0.8)

        # Show a table of pass@k for different p values
        p_header = label_text("Model accuracy (p)", color=WHITE)
        p_header.move_to(UP * 2.0 + LEFT * 3.5)
        self.play(FadeIn(p_header), run_time=0.4)

        p_values = [0.05, 0.10, 0.20, 0.40]
        k_show   = [1, 8, 16, 32, 64]

        col_spacing = 2.2
        row_spacing = 0.5

        # Column headers (k values)
        k_header_lbl = label_text("pass@k for k =", color=GREY_LIGHT)
        k_header_lbl.move_to(UP * 2.0 + LEFT * 0.2)
        self.play(FadeIn(k_header_lbl), run_time=0.3)

        k_headers_group = VGroup()
        for j, k in enumerate(k_show):
            kh = label_text(str(k), color=YELLOW_MED)
            kh.move_to(UP * 1.5 + LEFT * 0.2 + RIGHT * (j * col_spacing))
            k_headers_group.add(kh)
        self.play(FadeIn(k_headers_group), run_time=0.4)

        # Row data
        num_rows = VGroup()
        for i, p in enumerate(p_values):
            p_lbl = label_text(f"p = {int(p*100)}%", color=BLUE_MED)
            p_lbl.move_to(UP * 1.0 + LEFT * 3.5 + DOWN * (i * row_spacing))
            row_g = VGroup(p_lbl)
            for j, k in enumerate(k_show):
                pass_k = 1 - (1 - p) ** k
                val_str = f"{pass_k * 100:.0f}%"
                col = GREEN_MED if pass_k > 0.80 else (YELLOW_MED if pass_k > 0.40 else GREY_LIGHT)
                val_lbl = label_text(val_str, color=col)
                val_lbl.move_to(UP * 1.0 + LEFT * 0.2 + RIGHT * (j * col_spacing)
                                + DOWN * (i * row_spacing))
                row_g.add(val_lbl)
            num_rows.add(row_g)

        self.play(LaggedStart(*[FadeIn(r) for r in num_rows], lag_ratio=0.2),
                  run_time=1.3)

        takeaway = label_text(
            "k=64 recovers most correct samples even from weak models (p=5% -> 96% pass@64)",
            color=ORANGE_MED,
        )
        takeaway.to_edge(DOWN, buff=0.55)
        cost_note = label_text(
            "Cost: 64x more inference compute per problem — parallelizable across GPUs",
            color=GREY_MED,
        )
        cost_note.to_edge(DOWN, buff=0.28)
        self.play(FadeIn(takeaway), FadeIn(cost_note), run_time=0.5)
        self.wait(1.0)
        self.fade_all(numbers_title, p_header, k_header_lbl, k_headers_group,
                      num_rows, takeaway, cost_note)

        # ── 10. Limitations: bad verifier & all-wrong edge cases ──────────────
        limits_title = body_text(
            "Limitations: what can go wrong with MaxRL?",
            color=WHITE,
        )
        limits_title.to_edge(UP, buff=0.6)
        self.play(Write(limits_title), run_time=0.8)

        limit_data = [
            ("Verifier errors",    RED_MED,
             "Incorrect test cases accept wrong code.  Model learns bad solutions.",
             "Use trusted test suites, formal checkers where possible"),
            ("All-wrong batches",  ORANGE_MED,
             "Model generates k=64 responses, none correct.  Zero training signal.",
             "Curriculum: start easy, increase difficulty as model improves"),
            ("All-correct batches", YELLOW_MED,
             "Model already solves these problems.  Training signal wasted on known tasks.",
             "Adaptive sampling: weight harder problems more heavily"),
            ("Catastrophic forgetting", BLUE_MED,
             "Training only on math may degrade general language capabilities.",
             "Mix general SFT data into the training loop"),
        ]

        limit_boxes = VGroup()
        for name, col, problem, solution in limit_data:
            name_txt = label_text(name, color=col)
            prob_txt = label_text(problem, color=GREY_LIGHT)
            sol_txt  = label_text("Fix: " + solution, color=col)
            prob_txt.next_to(name_txt, RIGHT, buff=0.35)
            sol_txt.next_to(prob_txt, DOWN, buff=0.12)
            sol_txt.align_to(prob_txt, LEFT)
            row = VGroup(name_txt, prob_txt, sol_txt)
            limit_boxes.add(row)

        limit_boxes.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        limit_boxes.scale_to_fit_width(12.8)
        limit_boxes.move_to(ORIGIN + DOWN * 0.3)

        self.play(LaggedStart(*[FadeIn(r) for r in limit_boxes], lag_ratio=0.2),
                  run_time=1.5)
        self.wait(1.0)
        self.fade_all(limits_title, limit_boxes)

        # ── 11. Summary ───────────────────────────────────────────────────────
        summary_title = self.show_title("MaxRL — Summary", "The simplest RL that works")
        self.wait(0.6)
        self.fade_all(summary_title)

        summary_points = [
            ("Generate k responses",     BLUE_MED,   "Sample with high temperature for diversity"),
            ("Verify — binary filter",   YELLOW_MED, "Correct stays, wrong is discarded"),
            ("Supervised fine-tuning",   GREEN_MED,  "Cross-entropy on correct responses only"),
            ("No reward model",          ORANGE_MED, "Just an external verifier — simpler pipeline"),
            ("Best for math and code",   GREEN_MED,  "Wherever correctness can be checked"),
            ("pass@k drives the loop",   BLUE_MED,   "Larger k = more correct samples found"),
            ("Limitation: needs verifier", RED_MED,  "Cannot apply to subjective or open-ended tasks"),
        ]

        summary_rows = VGroup()
        for point, col, detail in summary_points:
            pt = label_text(point, color=col)
            dt = label_text(detail, color=GREY_LIGHT)
            dt.next_to(pt, RIGHT, buff=0.4)
            summary_rows.add(VGroup(pt, dt))

        summary_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        summary_rows.scale_to_fit_width(12.8)
        summary_rows.move_to(ORIGIN + DOWN * 0.2)

        self.play(LaggedStart(*[FadeIn(r) for r in summary_rows], lag_ratio=0.15),
                  run_time=1.8)

        next_lbl = label_text(
            "Up next: Trust Regions and Open Problems in RL for LLMs",
            color=GREY_MED,
        )
        next_lbl.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(next_lbl), run_time=0.5)
        self.wait(1.0)
        self.fade_all(summary_rows, next_lbl)
