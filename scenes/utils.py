"""
Shared utilities for LLM Math Visualiser Manim scenes.
Style inspired by 3blue1brown (3b1b).
"""

from manim import *

# ─── Colour palette ────────────────────────────────────────────────────────────
BLUE_DARK   = "#1C3A5E"
BLUE_MED    = "#2196F3"
BLUE_LIGHT  = "#90CAF9"
GREEN_DARK  = "#1B5E20"
GREEN_MED   = "#4CAF50"
GREEN_LIGHT = "#A5D6A7"
ORANGE_MED  = "#FF9800"
ORANGE_DARK = "#E65100"
RED_MED     = "#F44336"
PURPLE_MED  = "#9C27B0"
YELLOW_MED  = "#FFEB3B"
GREY_DARK   = "#212121"
GREY_MED    = "#616161"
GREY_LIGHT  = "#EEEEEE"

# ─── Typography helpers ────────────────────────────────────────────────────────
def title_text(text: str, color=WHITE) -> Text:
    return Text(text, color=color, font_size=42, weight=BOLD)

def subtitle_text(text: str, color=GREY_LIGHT) -> Text:
    return Text(text, color=color, font_size=28)

def body_text(text: str, color=WHITE) -> Text:
    return Text(text, color=color, font_size=22)

def label_text(text: str, color=GREY_LIGHT) -> Text:
    return Text(text, color=color, font_size=18)

def code_text(text: str, color=GREEN_LIGHT) -> Text:
    return Text(text, color=color, font_size=18, font="Courier New")


# ─── Node / neuron shapes ──────────────────────────────────────────────────────
def make_node(color=BLUE_MED, radius=0.3, label: str = "") -> VGroup:
    """A circle node with optional label."""
    circle = Circle(radius=radius, color=color, fill_opacity=0.8, fill_color=color)
    group = VGroup(circle)
    if label:
        txt = Text(label, font_size=16, color=WHITE)
        txt.move_to(circle)
        group.add(txt)
    return group


def make_layer(n: int, color=BLUE_MED, label: str = "", spacing: float = 0.9) -> VGroup:
    """A vertical column of n nodes."""
    nodes = VGroup(*[make_node(color=color) for _ in range(n)])
    nodes.arrange(DOWN, buff=spacing - 0.6)
    if label:
        lbl = label_text(label, color=color)
        lbl.next_to(nodes, DOWN, buff=0.3)
        return VGroup(nodes, lbl)
    return nodes


def connect_layers(layer_a: VGroup, layer_b: VGroup,
                   color=GREY_MED, stroke_width=1.2) -> VGroup:
    """Draw edges from every node in layer_a to every node in layer_b."""
    # layer_a and layer_b may be VGroups containing a VGroup of circles (plus label)
    def get_circles(layer):
        # If first child is a VGroup, it's the node sub-group
        first = layer[0]
        if isinstance(first, VGroup):
            return first
        return layer

    circles_a = get_circles(layer_a)
    circles_b = get_circles(layer_b)
    edges = VGroup()
    for a in circles_a:
        for b in circles_b:
            edge = Line(a.get_center(), b.get_center(),
                        stroke_width=stroke_width, color=color)
            edges.add(edge)
    return edges


# ─── Arrow helpers ─────────────────────────────────────────────────────────────
def right_arrow(start: np.ndarray, end: np.ndarray, color=WHITE) -> Arrow:
    return Arrow(start, end, color=color, buff=0.1, stroke_width=2)


# ─── Box helpers ───────────────────────────────────────────────────────────────
def rounded_box(width: float, height: float,
                fill_color=BLUE_DARK, stroke_color=BLUE_MED,
                label: str = "", label_color=WHITE) -> VGroup:
    rect = RoundedRectangle(width=width, height=height,
                            corner_radius=0.15,
                            fill_color=fill_color, fill_opacity=0.9,
                            stroke_color=stroke_color, stroke_width=2)
    group = VGroup(rect)
    if label:
        txt = body_text(label, color=label_color)
        txt.move_to(rect)
        group.add(txt)
    return group


# ─── Number vector display ─────────────────────────────────────────────────────
def make_vector_display(values: list, color=BLUE_LIGHT,
                        decimal_places: int = 2) -> VGroup:
    """Render a short list of floats as a bracketed column vector."""
    entries = VGroup(*[
        Text(f"{v:.{decimal_places}f}", color=color, font_size=20)
        for v in values
    ])
    entries.arrange(DOWN, buff=0.15)
    bracket_l = Text("[", color=WHITE, font_size=36)
    bracket_r = Text("]", color=WHITE, font_size=36)
    bracket_l.next_to(entries, LEFT, buff=0.1)
    bracket_r.next_to(entries, RIGHT, buff=0.1)
    return VGroup(bracket_l, entries, bracket_r)


# ─── Probability bar chart ─────────────────────────────────────────────────────
def make_prob_bars(labels: list, probs: list,
                  max_height: float = 2.5,
                  bar_width: float = 0.5,
                  colors: list = None) -> VGroup:
    """Simple bar chart for token probabilities."""
    if colors is None:
        colors = [BLUE_MED] * len(labels)
    bars = VGroup()
    for i, (lbl, prob, col) in enumerate(zip(labels, probs, colors)):
        h = max_height * prob
        bar = Rectangle(width=bar_width, height=max(h, 0.02),
                        fill_color=col, fill_opacity=0.9,
                        stroke_color=WHITE, stroke_width=1)
        pct_lbl = label_text(f"{prob*100:.0f}%", color=WHITE)
        pct_lbl.next_to(bar, UP, buff=0.1)
        word_lbl = label_text(lbl, color=GREY_LIGHT)
        word_lbl.next_to(bar, DOWN, buff=0.15)
        group = VGroup(bar, pct_lbl, word_lbl)
        group.shift(RIGHT * i * (bar_width + 0.3))
        bars.add(group)
    bars.move_to(ORIGIN)
    return bars


# ─── Attention heat-map grid ───────────────────────────────────────────────────
def make_attention_grid(tokens: list, scores: list,
                        cell_size: float = 0.55) -> VGroup:
    """
    Create an attention matrix where scores[i][j] is the attention
    from token i to token j.  Darker = higher attention.
    """
    n = len(tokens)
    grid = VGroup()
    for i in range(n):
        for j in range(n):
            val = scores[i][j]
            alpha = val  # 0→transparent, 1→opaque
            cell = Square(side_length=cell_size,
                          fill_color=BLUE_MED, fill_opacity=alpha,
                          stroke_color=GREY_DARK, stroke_width=0.5)
            cell.move_to([j * cell_size, -i * cell_size, 0])
            grid.add(cell)

    # Row labels (query)
    for i, tok in enumerate(tokens):
        lbl = label_text(tok, color=GREY_LIGHT)
        lbl.next_to(grid[i * n], LEFT, buff=0.15)
        grid.add(lbl)

    # Column labels (key)
    for j, tok in enumerate(tokens):
        lbl = label_text(tok, color=GREY_LIGHT)
        lbl.next_to(grid[j], UP, buff=0.15)
        grid.add(lbl)

    return grid


# ─── Scene base class ──────────────────────────────────────────────────────────
class LLMScene(Scene):
    """
    Base class with common camera config and helper methods.
    Subclass this for every topic scene.
    """
    def setup(self):
        self.camera.background_color = "#0D0D0D"

    def show_title(self, text: str, subtitle: str = "") -> VGroup:
        t = title_text(text)
        group = VGroup(t)
        if subtitle:
            s = subtitle_text(subtitle)
            s.next_to(t, DOWN, buff=0.3)
            group.add(s)
        group.to_edge(UP, buff=0.5)
        self.play(Write(t), run_time=1)
        if subtitle:
            self.play(FadeIn(group[1]), run_time=0.7)
        return group

    def fade_all(self, *mobjects, run_time: float = 0.5):
        self.play(*[FadeOut(m) for m in mobjects], run_time=run_time)
