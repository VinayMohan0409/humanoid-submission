"""Draw the architecture diagram in two styles, matte fills and a paper style with dark outlines and pale fills: training the world model, training the inverse dynamics model, and planning at test time."""
import matplotlib
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

from src.config import FIGURE_DIR
from src.plotting import INK, MUTED, SURFACE, capitalise, new_figure

matplotlib.rcParams["svg.fonttype"] = "none"
ROLES = ("data", "encoder", "model", "loss", "plan")
STYLES = {
    "matte": {"fill": dict(zip(ROLES, ("#C9CCD1", "#8FA3BF", "#A9C0AE", "#E0CB9A", "#D4A98F"))), "edge": "none", "lw": 0, "round": 0.12, "font": "DejaVu Sans", "ink": INK, "line": MUTED, "arrow_lw": 1.3},
    "paper": {"fill": dict(zip(ROLES, ("#F2F2F2", "#E3EAF4", "#E6F0E8", "#F6EFD9", "#F4E6DD"))), "edge": dict(zip(ROLES, ("#4A4A4A", "#2F4A72", "#2E5E3E", "#7A6420", "#8A4B2D"))), "lw": 1.2, "round": 0.04, "font": "STIXGeneral", "ink": "#111111", "line": "#1f1f1f", "arrow_lw": 1.0},
}
CAPTION = (r"$z_1 \dots z_5$ Is The Latent Action Sequence: One Latent Action (2 Tokens × 8 Numbers) For Each Of The Five 0.2 s Steps Between $F_0 \dots F_5$," "\n"
           r"Which Are DINOv3 Patch-Token Features Of Six Frames. $\hat{F}$ Marks A Prediction.")
MATH_FONTS = {"matte": "dejavusans", "paper": "stix"}


class Drawer:
    """Draws boxes, trapeziums, arrows and labels on one axes in one of the styles."""

    def __init__(self, ax, style):
        self.ax, self.st = ax, STYLES[style]

    def text(self, x, y, text, size=10, weight="normal", colour=None, ha="center"):
        """Place text in the style's font."""
        self.ax.text(x, y, text, ha=ha, va="center", fontsize=size, fontweight=weight, color=colour or self.st["ink"], family=self.st["font"])

    def label(self, x, y, title, detail, role, shape="box", w=2.2, h=1.3, size=10):
        """Draw a filled shape centred on (x, y) with a bold Title Case title and a detail line; return its left, right, top and bottom anchors."""
        edge = self.st["edge"]
        fill, edge, lw = self.st["fill"][role], edge if isinstance(edge, str) else edge[role], self.st["lw"]
        if shape == "trapezium":  # wide where frames enter, narrow where the latent action leaves
            self.ax.add_patch(Polygon([(x - w / 2, y + h / 2), (x + w / 2, y + h * 0.28), (x + w / 2, y - h * 0.28), (x - w / 2, y - h / 2)], closed=True, fc=fill, ec=edge, lw=lw, joinstyle="round"))
            x -= 0.3
        else:
            self.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={self.st['round']}", fc=fill, ec=edge, lw=lw))
        self.text(x, y + (0.17 if detail else 0), title, 11.5, "bold")
        if detail:
            self.text(x, y - 0.24, detail, size)
        x0 = x + (0.3 if shape == "trapezium" else 0)
        return {"l": (x0 - w / 2, y), "r": (x0 + w / 2, y), "t": (x0, y + h / 2), "b": (x0, y - h / 2)}

    def path(self, points, dashed=False):
        """Draw a right-angled path through the points with an arrowhead on its last segment."""
        style = dict(color=self.st["line"], lw=self.st["arrow_lw"], ls="--" if dashed else "-")
        for (x0, y0), (x1, y1) in zip(points[:-2], points[1:-1]):
            self.ax.plot([x0, x1], [y0, y1], solid_capstyle="butt", **style)
        self.ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=12, shrinkA=0, shrinkB=0, **style))

    def link(self, a, b, dy_a=0.0, dy_b=0.0, **kwargs):
        """Join the right side of `a` to the left side of `b` with a straight arrow or one right-angled step."""
        (x0, y0), (x1, y1) = a["r"], b["l"]
        y0, y1 = y0 + dy_a, y1 + dy_b
        mid = (x0 + x1) / 2
        self.path([(x0, y0), (x1, y1)] if y0 == y1 else [(x0, y0), (mid, y0), (mid, y1), (x1, y1)], **kwargs)


def draw(style):
    """Draw all three rows in one style and save architecture_<style>.png and .svg."""
    matplotlib.rcParams["mathtext.fontset"] = MATH_FONTS[style]
    fig, axes = new_figure(size=(17, 11.8))
    ax = axes[0, 0]
    ax.set(xlim=(0, 17), ylim=(-2.1, 9.4), aspect="equal")
    ax.axis("off")
    d = Drawer(ax, style)

    d.text(0.1, 9.0, capitalise("(a) Training The World Model: Human Video + Panda Demos, No Action Labels"), 12.5, "bold", ha="left")
    human = d.label(1.5, 7.9, "Human Video", "64 Clips", "data", w=2.3, h=0.8)
    panda = d.label(1.5, 6.7, "Panda Demos", "30 Episodes", "data", w=2.3, h=0.8)
    dino = d.label(4.05, 7.3, "Frozen DINOv3", "Frames → Patch Tokens", "encoder", w=2.4)
    feats = d.label(6.6, 7.3, r"Features $F_0 \dots F_5$", "6 Frames, 0.2 s Apart", "data")
    enc = d.label(9.5, 7.3, "Latent Action Encoder", r"$(F_t, F_{t+1}) \rightarrow z$", "encoder", "trapezium", w=3.0, h=1.2)
    wm = d.label(13.3, 7.3, "World Model", r"$F_0 + z_1 \dots z_5 \rightarrow \hat{F}_1 \dots \hat{F}_5$", "model", w=2.7)
    loss = d.label(16.1, 7.3, "Feature Loss", r"MSE Against $F_1 \dots F_5$", "loss", w=1.8, size=9.5)
    d.link(human, dino, dy_b=0.35)
    d.link(panda, dino, dy_b=-0.35)
    d.link(dino, feats)
    d.link(feats, enc)
    d.link(enc, wm)
    d.text((enc["r"][0] + wm["l"][0]) / 2, 6.9, r"$z_1 \dots z_5$", 14)
    d.path([feats["b"], (feats["b"][0], 5.95), (wm["b"][0], 5.95), wm["b"]])
    d.text((feats["b"][0] + wm["b"][0]) / 2, 6.25, r"$F_0$", 15)
    d.link(wm, loss)

    d.text(0.1, 5.15, capitalise("(b) Training The Inverse Dynamics Model: Separately, On Labelled Play Data"), 12.5, "bold", ha="left")
    play = d.label(1.9, 3.9, "Play Episodes", "180 Panda Or 180 Robotiq", "data", w=3.0, h=1.0)
    dino2 = d.label(5.3, 3.9, "Frozen DINOv3", "Frame Pairs → Features", "encoder", w=2.4, h=1.0)
    idm = d.label(8.9, 3.9, "Inverse Dynamics Model", r"$(F_t, F_{t+1}) \rightarrow$ Actions", "model", w=3.2, h=1.0)
    act = d.label(12.6, 3.9, "Position Commands", "x, y, z For 4 Control Steps", "loss", w=3.0, h=1.0)
    d.link(play, dino2)
    d.link(dino2, idm)
    d.link(idm, act)

    d.text(0.1, 2.35, capitalise("(c) Planning At Test Time: Goal Image In, Position Commands Out"), 12.5, "bold", ha="left")
    cur = d.label(1.7, 1.15, "Current Frame", "From The Simulator", "data", w=2.4, h=0.9)
    goal = d.label(1.7, 0.1, "Goal Image", "Scripted Arm At The Goal", "data", w=2.4, h=0.9)
    dino3 = d.label(4.6, 0.62, "Frozen DINOv3", "→ Features", "encoder", w=1.9)
    cem = d.label(7.9, 0.62, "CEM Planner", "Sample z, Predict, Score Against Goal", "plan", w=3.6)
    wm2 = d.label(11.2, 0.62, "World Model", r"$z^* \rightarrow \hat{F}_1$", "model", w=2.0)
    idm2 = d.label(13.7, 0.62, "IDM", r"$(F_0, \hat{F}_1) \rightarrow$ Commands", "model", w=2.3)
    sim = d.label(16.0, 0.62, "Simulator", None, "data", w=1.8)
    d.link(cur, dino3, dy_b=0.3)
    d.link(goal, dino3, dy_b=-0.3)
    for a, b in ((dino3, cem), (cem, wm2), (wm2, idm2), (idm2, sim)):
        d.link(a, b)
    low, left = -0.95, 0.25  # the replanning loop runs below the row and up the left side
    d.path([sim["b"], (sim["b"][0], low), (left, low), (left, cur["l"][1]), cur["l"]], dashed=True)
    d.text(9.0, low + 0.22, capitalise("Replan Every 4 Control Steps With The New Frame"), 10)
    d.text(8.5, -1.65, CAPTION, 10.5)

    for extension in ("png", "svg"):
        fig.savefig(FIGURE_DIR / f"architecture_{style}.{extension}", dpi=300, facecolor=SURFACE, bbox_inches="tight")
    print(FIGURE_DIR / f"architecture_{style}.png")


def main():
    """Draw both styles."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for style in STYLES:
        draw(style)


if __name__ == "__main__":
    main()
