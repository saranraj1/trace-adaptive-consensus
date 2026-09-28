"""
Script to generate publication-grade figures for the TrACE reproduction and extensions report.
Outputs 4 figures at 300 DPI in the figures/ directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create output directory
os.makedirs("figures", exist_ok=True)

# Set high-quality visual style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 1.0,
    "grid.color": "#E5E7EB",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "figure.autolayout": True,
})

# Color palette (Deep Indigo, Teal, Coral, Slate)
PRIMARY_BLUE = "#1E3A8A"
ACCENT_TEAL = "#0D9488"
CORAL_RED = "#E11D48"
SLATE_GRAY = "#64748B"
PURPLE = "#7C3AED"
AMBER = "#D97706"


# ==============================================================================
# Figure 1: Accuracy vs Compute Pareto Frontier
# ==============================================================================
def generate_figure_1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)

    # Panel 1: GSM8K Math Reasoning
    ax1.plot([1.0, 2.0, 2.8, 4.0], [0.400, 0.450, 0.450, 0.450], linestyle=":", color=CORAL_RED, alpha=0.7, linewidth=2, label="Pareto Frontier")

    # Plot points
    ax1.scatter([1.0], [0.400], color=SLATE_GRAY, s=150, marker="o", edgecolors="black", linewidths=1.2, zorder=5)
    ax1.annotate("Greedy (k=1)\n1.0 call", (1.0, 0.400), xytext=(1.0, 0.375), ha="center", va="top",
                 fontsize=9.5, color=SLATE_GRAY, fontweight="medium",
                 arrowprops=dict(arrowstyle="->", color=SLATE_GRAY, lw=1.0, shrinkA=3, shrinkB=4))

    ax1.scatter([2.0], [0.450], color=CORAL_RED, s=240, marker="*", edgecolors="black", linewidths=1.2, zorder=6)
    ax1.annotate("Ext 1 & 2 (Groq)\n2.0 calls (-50%)", (2.0, 0.450), xytext=(2.0, 0.472), ha="center", va="bottom",
                 fontsize=9.5, color=CORAL_RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=CORAL_RED, lw=1.2, shrinkA=3, shrinkB=4))

    ax1.scatter([2.8], [0.450], color=ACCENT_TEAL, s=150, marker="^", edgecolors="black", linewidths=1.2, zorder=5)
    ax1.annotate("TrACE-4 (Paper)\n2.8 calls (-30%)", (2.8, 0.450), xytext=(2.2, 0.410), ha="center", va="top",
                 fontsize=9.5, color=ACCENT_TEAL, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=ACCENT_TEAL, lw=1.2, shrinkA=3, shrinkB=4))

    ax1.scatter([3.85], [0.450], color=PRIMARY_BLUE, s=150, marker="s", edgecolors="black", linewidths=1.2, zorder=5)
    ax1.annotate("SC-4\n(4 calls)", (3.85, 0.450), xytext=(3.85, 0.472), ha="center", va="bottom",
                 fontsize=9.5, color=PRIMARY_BLUE, fontweight="medium",
                 arrowprops=dict(arrowstyle="->", color=PRIMARY_BLUE, lw=1.2, shrinkA=3, shrinkB=4))

    ax1.scatter([4.15], [0.450], color=PURPLE, s=150, marker="^", edgecolors="black", linewidths=1.2, zorder=5)
    ax1.annotate("TrACE-8 (Paper)\n4.0 calls (-50%)", (4.15, 0.450), xytext=(4.75, 0.410), ha="center", va="top",
                 fontsize=9.5, color=PURPLE, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.2, shrinkA=3, shrinkB=4))

    ax1.scatter([8.0], [0.450], color=PRIMARY_BLUE, s=150, marker="s", edgecolors="black", linewidths=1.2, zorder=5)
    ax1.annotate("SC-8\n(8 calls)", (8.0, 0.450), xytext=(8.0, 0.472), ha="center", va="bottom",
                 fontsize=9.5, color=PRIMARY_BLUE, fontweight="medium",
                 arrowprops=dict(arrowstyle="->", color=PRIMARY_BLUE, lw=1.2, shrinkA=3, shrinkB=4))

    ax1.set_title("GSM8K (Math Reasoning)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Mean Inference Calls per Task", fontsize=11, fontweight="medium")
    ax1.set_ylabel("Accuracy", fontsize=11, fontweight="medium")
    ax1.set_xlim(0.2, 9.2)
    ax1.set_ylim(0.355, 0.495)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: MiniHouse Agent Navigation
    ax2.plot([5.67, 12.00], [1.000, 1.000], linestyle=":", color=ACCENT_TEAL, alpha=0.7, linewidth=2, label="Optimal Frontier")

    ax2.scatter([5.67], [1.000], color=SLATE_GRAY, s=150, marker="o", edgecolors="black", linewidths=1.2, zorder=5)
    ax2.annotate("Greedy (k=1)\n5.7 calls", (5.67, 1.000), xytext=(4.5, 0.935), ha="center", va="top",
                 fontsize=9.5, color=SLATE_GRAY, fontweight="medium",
                 arrowprops=dict(arrowstyle="->", color=SLATE_GRAY, lw=1.0, shrinkA=3, shrinkB=4))

    ax2.scatter([12.00], [1.000], color=ACCENT_TEAL, s=160, marker="^", edgecolors="black", linewidths=1.2, zorder=6)
    ax2.annotate("TrACE-4\n12.0 calls (-47.1%)", (12.00, 1.000), xytext=(12.00, 1.035), ha="center", va="bottom",
                 fontsize=9.5, color=ACCENT_TEAL, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=ACCENT_TEAL, lw=1.2, shrinkA=3, shrinkB=4))

    ax2.scatter([13.33], [1.000], color=PURPLE, s=160, marker="^", edgecolors="black", linewidths=1.2, zorder=6)
    ax2.annotate("TrACE-8\n13.3 calls (-70.6%)", (13.33, 1.000), xytext=(15.0, 0.955), ha="center", va="top",
                 fontsize=9.5, color=PURPLE, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.2, shrinkA=3, shrinkB=4))

    ax2.scatter([22.67], [1.000], color=PRIMARY_BLUE, s=150, marker="s", edgecolors="black", linewidths=1.2, zorder=5)
    ax2.annotate("SC-4\n(22.7 calls)", (22.67, 1.000), xytext=(22.67, 1.035), ha="center", va="bottom",
                 fontsize=9.5, color=PRIMARY_BLUE, fontweight="medium",
                 arrowprops=dict(arrowstyle="->", color=PRIMARY_BLUE, lw=1.2, shrinkA=3, shrinkB=4))

    ax2.scatter([45.33], [1.000], color=PRIMARY_BLUE, s=150, marker="s", edgecolors="black", linewidths=1.2, zorder=5)
    ax2.annotate("SC-8\n(45.3 calls)", (45.33, 1.000), xytext=(45.33, 1.035), ha="center", va="bottom",
                 fontsize=9.5, color=PRIMARY_BLUE, fontweight="medium",
                 arrowprops=dict(arrowstyle="->", color=PRIMARY_BLUE, lw=1.2, shrinkA=3, shrinkB=4))

    ax2.set_title("MiniHouse (Multi-Step Agent Navigation)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Mean Inference Calls per Trajectory", fontsize=11, fontweight="medium")
    ax2.set_ylabel("Success Rate", fontsize=11, fontweight="medium")
    ax2.set_xlim(0.0, 52.0)
    ax2.set_ylim(0.905, 1.075)
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Accuracy vs. Compute: Pareto Dominance of Adaptive Agreement", fontsize=15, fontweight="bold", y=1.03)
    fig.savefig("figures/fig1_pareto_frontier.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/fig1_pareto_frontier.png")


# ==============================================================================
# Figure 2: Compute & Token Reduction Breakdown
# ==============================================================================
def generate_figure_2():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    conditions = ["SC-4\n(Baseline)", "TrACE-4\n(Paper)", "Ext 1: Streaming\n(Prefix Halting)", "Ext 2: Calibrated\n(Conf-Weighted)"]
    call_reductions = [0.0, 50.0, 50.0, 50.0]
    token_reductions = [0.0, 49.8, 49.5, 49.8]

    x = np.arange(len(conditions))
    width = 0.35

    rects1 = ax.bar(x - width/2, call_reductions, width, label="Call Reduction %", color=PRIMARY_BLUE, edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, token_reductions, width, label="Token Reduction %", color=ACCENT_TEAL, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Reduction Relative to SC-4 Baseline (%)", fontsize=11, fontweight="medium")
    ax.set_title("Compute & Token Savings Across Controllers (GSM8K Benchmark)", fontsize=14, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10.5, fontweight="medium")
    ax.set_ylim(0, 65)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=10.5)

    # Add data labels
    for rect in rects1:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f"-{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                        fontweight="bold", color=PRIMARY_BLUE, fontsize=10)
        else:
            ax.annotate("0.0%\n(Base)", xy=(rect.get_x() + rect.get_width() / 2, 0),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                        fontweight="normal", color=SLATE_GRAY, fontsize=9)

    for rect in rects2:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f"-{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                        fontweight="bold", color=ACCENT_TEAL, fontsize=10)
        else:
            ax.annotate("0.0%\n(Base)", xy=(rect.get_x() + rect.get_width() / 2, 0),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                        fontweight="normal", color=SLATE_GRAY, fontsize=9)

    fig.savefig("figures/fig2_compute_reduction.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/fig2_compute_reduction.png")


# ==============================================================================
# Figure 3: Intra-Trajectory Step-by-Step Compute Accumulation
# ==============================================================================
def generate_figure_3():
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)

    steps = np.arange(1, 7)
    # Fixed SC-4 generates 4 candidates every single step
    sc4_cumulative = steps * 4
    # Adaptive TrACE exits at k=2 when rollouts agree
    trace4_cumulative = steps * 2

    ax.plot(steps, sc4_cumulative, marker="s", color=PRIMARY_BLUE, linewidth=2.5, markersize=8, label="Fixed Self-Consistency (SC-4, 4 calls/step)")
    ax.plot(steps, trace4_cumulative, marker="^", color=CORAL_RED, linewidth=2.5, markersize=8, label="Adaptive TrACE-4 (Consensus Early Exit, 2 calls/step)")

    # Shade the compute savings gap
    ax.fill_between(steps, trace4_cumulative, sc4_cumulative, color=CORAL_RED, alpha=0.15, label="Cumulative Compute Saved (50%)")

    # Annotate end points
    ax.annotate("24 Total Calls", xy=(6, 24), xytext=(5.3, 25), fontsize=10.5, fontweight="bold", color=PRIMARY_BLUE)
    ax.annotate("12 Total Calls (-50%)", xy=(6, 12), xytext=(5.1, 9.5), fontsize=10.5, fontweight="bold", color=CORAL_RED)

    ax.set_title("Intra-Trajectory Compute Accumulation over Agent Steps", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Agent Trajectory Step (MiniHouse Environment)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Cumulative Inference API Calls", fontsize=11, fontweight="medium")
    ax.set_xticks(steps)
    ax.set_ylim(0, 28)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=10, loc="upper left")

    fig.savefig("figures/fig3_trajectory_accumulation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/fig3_trajectory_accumulation.png")


# ==============================================================================
# Figure 4: Confidence Gating Mechanic (Extension 2)
# ==============================================================================
def generate_figure_4():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 2D Decision Space: Weighted Agreement alpha_w (X) vs Average Confidence w_bar (Y)
    # Dual Thresholds: alpha_w >= 0.75 and w_bar >= 0.50

    # Draw colored quadrants
    # Quadrant 1: Early Exit Allowed (Top-Right)
    rect_safe = patches.Rectangle((0.75, 0.50), 0.25, 0.50, linewidth=1.5, edgecolor=ACCENT_TEAL, facecolor="#CCFBF1", alpha=0.7, label="Safe Early Exit Region")
    ax.add_patch(rect_safe)

    # Quadrant 2: Hallucination Collusion Hazard (Bottom-Right)
    rect_hazard = patches.Rectangle((0.75, 0.0), 0.25, 0.50, linewidth=1.5, edgecolor=CORAL_RED, facecolor="#FFE4E6", alpha=0.7, label="Hallucination Veto Zone (Blocked)")
    ax.add_patch(rect_hazard)

    # Left Region: Divergence / Uncertainty (Left)
    rect_diverge = patches.Rectangle((0.0, 0.0), 0.75, 1.0, linewidth=1.5, edgecolor=SLATE_GRAY, facecolor="#F1F5F9", alpha=0.7, label="Divergence Region (Forced Expansion)")
    ax.add_patch(rect_diverge)

    # Threshold dashed lines
    ax.axvline(0.75, color=PRIMARY_BLUE, linestyle="--", linewidth=2.0, label="Agreement Threshold (tau_high = 0.75)")
    ax.axhline(0.50, color=CORAL_RED, linestyle="--", linewidth=2.0, label="Confidence Floor (tau_conf = 0.50)")

    # Scatter representative scenarios
    # Scenario A: High confidence agreement (e.g. math reasoning confident)
    ax.scatter([0.95], [0.88], color=ACCENT_TEAL, s=180, edgecolors="black", zorder=6)
    ax.annotate("Scenario A: Valid Consensus\n(alpha_w=0.95, w=0.88) -> EXIT", xy=(0.95, 0.88), xytext=(0.82, 0.80),
                fontsize=9.5, fontweight="bold", color=ACCENT_TEAL, arrowprops=dict(arrowstyle="->", color=ACCENT_TEAL, lw=1.5))

    # Scenario B: Low confidence collusion (hallucination agreement)
    ax.scatter([0.90], [0.25], color=CORAL_RED, s=180, edgecolors="black", zorder=6)
    ax.annotate("Scenario B: Spurious Hallucination\n(alpha_w=0.90, w=0.25) -> VETOED & EXPAND", xy=(0.90, 0.25), xytext=(0.77, 0.16),
                fontsize=9.5, fontweight="bold", color=CORAL_RED, arrowprops=dict(arrowstyle="->", color=CORAL_RED, lw=1.5))

    # Scenario C: Disagreement
    ax.scatter([0.50], [0.70], color=SLATE_GRAY, s=180, edgecolors="black", zorder=6)
    ax.annotate("Scenario C: High Uncertainty\n(alpha_w=0.50) -> EXPAND TO k_max", xy=(0.50, 0.70), xytext=(0.35, 0.60),
                fontsize=9.5, fontweight="bold", color=SLATE_GRAY, arrowprops=dict(arrowstyle="->", color=SLATE_GRAY, lw=1.5))

    ax.set_title("Extension 2: Dual-Gated Consensus & Hallucination Prevention Space", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Weighted Action Agreement Score (alpha_w)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Mean Sequence Confidence (w_bar)", fontsize=11, fontweight="medium")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.savefig("figures/fig4_confidence_gating.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/fig4_confidence_gating.png")


if __name__ == "__main__":
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    print("All 4 figures generated successfully at 300 DPI in figures/!")
