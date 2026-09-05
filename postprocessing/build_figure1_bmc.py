"""Rebuilds the RAG pipeline diagram for the BMC manuscript, labeled as Figure 1 since the
Frontiers version's Figure 1 (inclusion/exclusion flow chart) does not exist in this
manuscript -- this pipeline diagram is the only figure. Same drawing as
build_figure2.py (kept untouched for the historical Frontiers submission), just relabeled
and saved to a separate path so neither manuscript's figure numbering is disturbed."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
from matplotlib.patches import ConnectionStyle

OUT = "docs/BMC_Submission/Figure1_pipeline_diagram.png"

fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_xlim(0, 15)
ax.set_ylim(0, 7)
ax.axis("off")

ax.text(7.5, 6.6, "Figure 1: Retrieval-Augmented Generation (RAG) Pipeline", fontsize=12.5,
        fontweight="bold", ha="center")


def rbox(x, y, w, h, text, fc, ec="#444444", fontsize=9, fontcolor="black", weight="normal"):
    b = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.1",
        fc=fc, ec=ec, lw=1.2, zorder=2,
    )
    ax.add_patch(b)
    ax.text(x, y, text, fontsize=fontsize, ha="center", va="center", color=fontcolor,
            weight=weight, zorder=3)


def arrow(p1, p2, color="#333333", lw=1.4):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, color=color, lw=lw,
                         connectionstyle=ConnectionStyle("Arc3", rad=0.0), zorder=1)
    ax.add_patch(a)


# --- Input PDF ellipse + count ---
e1 = Ellipse((1.1, 4.4), 1.9, 0.55, fc="none", ec="#c0622a", lw=2.2)
ax.add_patch(e1)
ax.annotate("", xy=(1.1, 4.4), xytext=(1.1, 4.4),
            arrowprops=dict(arrowstyle="->", color="#c0622a", lw=1.6,
                             connectionstyle="arc3,rad=1.3"))
rbox(1.1, 3.55, 2.1, 0.8, "Input PDF", fc="#3f6fb5", fontcolor="white", fontsize=10.5, weight="bold")
ax.text(1.1, 4.4, "769 PDFs", fontsize=9.5, ha="center", va="center", color="#c0622a", weight="bold")

# --- Preprocessing box (PDF -> Text/Tables/Images) ---
outer = FancyBboxPatch((2.9, 2.0), 4.2, 3.6, boxstyle="round,pad=0.05,rounding_size=0.08",
                        fc="none", ec="#333333", lw=1.4, zorder=1)
ax.add_patch(outer)
ax.text(3.15, 5.35, "PDF", fontsize=8, ha="left", va="center")

for i, (label, fc) in enumerate([("Text", "#f4d9a0"), ("Tables", "#f4d9a0"), ("Images", "#f4d9a0")]):
    y = 4.6 - i * 1.0
    rbox(3.7, y, 1.1, 0.6, label, fc=fc, fontsize=8.5)
    # chunk icons "C"
    for j in range(2):
        rbox(4.9 + j * 0.55, y + 0.15 - j * 0.05, 0.42, 0.42, "C", fc="#e3e3e3", fontsize=7)
rbox(4.9, 4.6 + 0.5, 0.42, 0.42, "H", fc="#cfe0f0", fontsize=7)

ax.text(5.85, 3.05, "Embeddings", fontsize=8, rotation=90, ha="center", va="center", color="#3a7a3a")

# --- Vector DB ---
vdb_x, vdb_y = 6.85, 3.6
ax.add_patch(FancyBboxPatch((vdb_x - 0.55, vdb_y - 0.9), 1.1, 1.8,
                             boxstyle="round,pad=0.03,rounding_size=0.35",
                             fc="#2f6b4f", ec="#204a37", lw=1.2, zorder=2))
ax.text(vdb_x, vdb_y, "Vector\nDB", fontsize=8.5, ha="center", va="center", color="white", weight="bold")

# Query variations -> vector DB
rbox(6.85, 5.7, 1.3, 0.5, "Query\nVariations", fc="#e3e3e3", fontsize=7.5)
ax.text(6.2, 5.15, "Query", fontsize=8, color="#b03030", weight="bold")
arrow((6.85, 5.45), (6.85, 4.5), color="#c0622a")

# Retriever
rbox(6.85, 1.9, 1.3, 0.55, "Retriever", fc="#e3e3e3", fontsize=7.5)
arrow((6.85, 2.2), (6.85, 2.7), color="#c0622a")

# Augmentation
rbox(8.4, 3.6, 1.0, 0.7, "Augment-\nation", fc="#cfe0f0", fontsize=7.5)
arrow((7.4, 3.6), (7.9, 3.6))

# Generation
rbox(9.7, 3.6, 1.0, 0.7, "Generation", fc="#cfe0f0", fontsize=7.5)
arrow((8.9, 3.6), (9.2, 3.6))
ax.text(9.7, 4.15, "Answer", fontsize=8, ha="center", color="#3f6fb5", weight="bold")

# RAGAS panel
ragas_x = 11.2
ax.add_patch(FancyBboxPatch((ragas_x - 0.8, 2.0), 1.6, 3.6, boxstyle="round,pad=0.04,rounding_size=0.08",
                             fc="#8a8a8a", ec="#5c5c5c", lw=1.2, zorder=1))
for i, label in enumerate(["Context\nRecall", "Answer\nRelevancy", "Faithfulness"]):
    rbox(ragas_x, 4.9 - i * 1.05, 1.3, 0.85, label, fc="#e3e3e3", fontsize=7.5)
ax.text(ragas_x, 2.25, "RAGAS", fontsize=8, ha="center", color="white", weight="bold")
arrow((10.2, 3.6), (ragas_x - 0.8, 3.6))

# Output CSVs
rbox(13.7, 3.6, 2.0, 0.8, "Output\nCSVs", fc="#c0622a", fontcolor="white", fontsize=10.5, weight="bold")
ax.text(13.7, 4.5, "1,012 CSVs", fontsize=9.5, ha="center", color="#3f6fb5", weight="bold")
arrow((12.0, 3.6), (12.7, 3.6))

arrow((2.2, 3.9), (2.9, 3.7))

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved {OUT}")
