"""Rebuilds Figure 1 (inclusion/exclusion flow chart) with the corrected two-batch numbers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import ConnectionStyle

OUT = "docs/manuscript_revision/Figure1_flow_chart.png"

fig, ax = plt.subplots(figsize=(9, 8.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")

BOX_FC = "#ececec"
BOX_EC = "#666666"
FONT = dict(fontsize=9.5, ha="center", va="center", family="DejaVu Sans")


def box(x, y, w, h, text, fc=BOX_FC):
    b = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        fc=fc, ec=BOX_EC, lw=1.1, zorder=2,
    )
    ax.add_patch(b)
    ax.text(x, y, text, wrap=True, zorder=3, **FONT)
    return (x, y, w, h)


def arrow(p1, p2, dashed=False, label=None):
    style = "-" if not dashed else (0, (4, 3))
    a = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=14,
        color="#3a5a78", lw=1.3, linestyle=style, zorder=1,
        connectionstyle=ConnectionStyle("Arc3", rad=0.0),
    )
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx + 0.15, my, label, fontsize=8, ha="left", va="center", color="#3a5a78")


ax.text(5, 11.7, "Figure 1: Inclusion and Exclusion Flow Chart", fontsize=12,
        fontweight="bold", ha="center")

# Row 1: initial retrieval
b1 = box(5, 10.4, 5.4, 0.9, "773 full-text PDFs initially retrieved")

# duplicates removed (side box)
b2 = box(1.7, 9.0, 3.0, 0.9, "5 duplicate articles removed")
arrow((5, 9.95), (1.7, 9.45), dashed=True)

# unique articles
b3 = box(5, 8.0, 4.6, 0.8, "768 unique articles")
arrow((5, 9.95), (5, 8.4))

# split: excluded for size vs retained
b4 = box(7.9, 6.5, 3.6, 1.1, "129 articles excluded — file size\nexceeded 1.5 MB, preventing\nautomated processing")
arrow((5.6, 7.75), (7.2, 7.0), dashed=True)

b5 = box(2.3, 6.5, 3.6, 1.1, "639 full-text articles met\nall criteria — retained\nfor Batch-1 processing")
arrow((4.4, 7.75), (2.9, 7.0))

# Batch-1 late addition
b6 = box(2.3, 4.9, 3.6, 1.1, "+1 article identified\nand added later\n→ 640 articles: Batch-1 final")
arrow((2.3, 5.95), (2.3, 5.45))

# Batch-2 recovery
b7 = box(7.9, 4.9, 3.6, 1.3,
         "Reprocessed via PDF compression;\nunrecoverable articles replaced with\nequivalent newly-identified\nopen-access articles")
arrow((7.9, 5.95), (7.9, 5.55))

b8 = box(7.9, 3.3, 3.6, 0.9, "129 articles: Batch-2 final")
arrow((7.9, 4.25), (7.9, 3.75))

# Final total
b9 = box(5, 1.6, 6.2, 1.1, "640 (Batch-1) + 129 (Batch-2) = 769 total articles\nprocessed, 1,012 tool-level CSVs generated",
         fc="#dfe9f0")
arrow((2.3, 4.35), (4.3, 2.1))
arrow((7.9, 2.85), (5.7, 2.1))

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved {OUT}")
