"""绘制级联失效敏感性热力图 — 全中文版"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

data = np.load("cascade_sensitivity_results.npz")
alpha_values = data["alpha_values"]
attack_sizes = data["attack_sizes"]
grid_cascade = data["grid_cascade"]
grid_total = data["grid_total"]
grid_steps = data["grid_steps"]

na, ns = len(alpha_values), len(attack_sizes)
xtick_labels = [f"Top-{k}" for k in attack_sizes]
ytick_labels = [f"{a:.2f}" for a in alpha_values]

# ============ 图1: 三面板热力图 ============
fig, axes = plt.subplots(1, 3, figsize=(24, 7))
plt.subplots_adjust(wspace=0.42, bottom=0.18)

font_kw = {"fontsize": 11}

# -- 面板1: 总失效比例 --
ax1 = axes[0]
vmax1 = max(15, grid_total.max())
im1 = ax1.imshow(grid_total, cmap="RdYlBu_r", aspect="auto", origin="lower",
                  vmin=0, vmax=vmax1)
ax1.set_xticks(range(ns)); ax1.set_yticks(range(na))
ax1.set_xticklabels(xtick_labels, **font_kw)
ax1.set_yticklabels(ytick_labels, **font_kw)
ax1.set_xlabel("蓄意攻击规模", fontsize=12, labelpad=10)
ax1.set_ylabel("容量冗余系数 α", fontsize=12, labelpad=15)
ax1.set_title("总失效节点比例 (%)", fontsize=14, fontweight="bold", pad=15)
for i in range(na):
    for j in range(ns):
        val = grid_total[i, j]
        color = "white" if val > vmax1 * 0.55 else "black"
        ax1.text(j, i, f"{val:.1f}", ha="center", va="center",
                 fontsize=9.5, fontweight="bold", color=color)
cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.82, pad=0.03)
cbar1.set_label("失效占全网%", fontsize=10)

# -- 面板2: 级联波及节点数 --
ax2 = axes[1]
vmax2 = max(10, grid_cascade.max())
im2 = ax2.imshow(grid_cascade, cmap="YlOrRd", aspect="auto", origin="lower",
                  norm=mcolors.LogNorm(vmin=max(1, grid_cascade.min()), vmax=vmax2))
ax2.set_xticks(range(ns)); ax2.set_yticks(range(na))
ax2.set_xticklabels(xtick_labels, **font_kw)
ax2.set_yticklabels(ytick_labels, **font_kw)
ax2.set_xlabel("蓄意攻击规模", fontsize=12, labelpad=10)
ax2.set_ylabel("容量冗余系数 α", fontsize=12, labelpad=15)
ax2.set_title("纯级联波及节点数 (对数刻度)", fontsize=14, fontweight="bold", pad=15)
for i in range(na):
    for j in range(ns):
        val = grid_cascade[i, j]
        color = "white" if val > vmax2 * 0.4 else "black"
        ax2.text(j, i, f"{int(val)}", ha="center", va="center",
                 fontsize=9.5, fontweight="bold", color=color)
cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.82, pad=0.03)
cbar2.set_label("波及节点数 (对数)", fontsize=10)

# -- 面板3: 级联步数 --
ax3 = axes[2]
vmax3 = max(1, grid_steps.max())
im3 = ax3.imshow(grid_steps, cmap="Purples", aspect="auto", origin="lower")
ax3.set_xticks(range(ns)); ax3.set_yticks(range(na))
ax3.set_xticklabels(xtick_labels, **font_kw)
ax3.set_yticklabels(ytick_labels, **font_kw)
ax3.set_xlabel("蓄意攻击规模", fontsize=12, labelpad=10)
ax3.set_ylabel("容量冗余系数 α", fontsize=12, labelpad=15)
ax3.set_title("级联迭代至稳定所需步数", fontsize=14, fontweight="bold", pad=15)
for i in range(na):
    for j in range(ns):
        val = grid_steps[i, j]
        color = "white" if val > vmax3 * 0.5 else "black"
        ax3.text(j, i, f"{int(val)}", ha="center", va="center",
                 fontsize=9.5, fontweight="bold", color=color)
cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.82, pad=0.03)
cbar3.set_label("迭代步数", fontsize=10)

# 关键发现放在面板下方（不在数据区域内）
fig.text(0.5, 0.04,
         "关键发现: α=0 → 任何攻击触发大规模级联 | α≥0.05 → Top-1永不触发级联 | "
         "Top-2始终触发级联 | Braess悖论: Top-3/Top-5的总失效可能少于Top-2",
         fontsize=11.5, fontstyle="italic", color="#333333", ha="center",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F0F0", edgecolor="none", alpha=0.8),
         transform=fig.transFigure)

plt.savefig("cascade_sensitivity_heatmap.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("[OK] cascade_sensitivity_heatmap.png (全中文)")

# ============ 图2: 级联触发阈值分析 ============
fig2, ax = plt.subplots(figsize=(11, 7.5))

first_cascade_size = []
for i, alpha in enumerate(alpha_values):
    found = None
    for j, ks in enumerate(attack_sizes):
        if grid_cascade[i, j] > 0:
            found = ks
            break
    first_cascade_size.append(found if found else None)

valid = [(i, v) for i, v in enumerate(first_cascade_size) if v is not None]

# 主曲线
ax.plot([alpha_values[i] for i, v in valid],
        [v for i, v in valid],
        "o-", color="#E74C3C", linewidth=3.5, markersize=14,
        markerfacecolor="#E74C3C", markeredgecolor="#8B0000", markeredgewidth=1.5,
        zorder=5, label="级联触发阈值")

# 绿色安全区
ax.fill_between([-0.05, 0.55], [0.5, 0.5], [0.5, 0.5],
                alpha=0.06, color="green")
ax.text(0.15, 0.7, "安全区\n(无级联)",
        ha="center", va="bottom", fontsize=13, color="#27AE60",
        fontweight="bold", fontstyle="italic")

ax.set_xlabel("容量冗余系数 α", fontsize=14, labelpad=12)
ax.set_ylabel("触发级联所需最小攻击规模", fontsize=14, labelpad=12)
ax.set_title("级联触发阈值分析", fontsize=16, fontweight="bold", pad=20)
ax.set_xlim(-0.02, 0.55)
ax.set_ylim(0, 22)
ax.set_yticks([1, 2, 3, 5, 7, 10, 15, 20])
ax.set_yticklabels([f"Top-{k}" for k in [1, 2, 3, 5, 7, 10, 15, 20]], fontsize=11)
ax.legend(fontsize=12, loc="lower left", framealpha=0.9, bbox_to_anchor=(0.02, 0.15))
ax.grid(True, linestyle=":", alpha=0.5)

# 文字说明放在右上角（坐标轴坐标，不挡数据）
ax.text(0.98, 0.96,
        "稳定性分析结论:\n\n"
        "• Top-1 在任何 α≥0.05 下\n"
        "  均不触发级联\n"
        "• Top-2 无论冗余系数多大\n"
        "  始终触发级联\n"
        "• 上海地铁具有强单点鲁棒性\n"
        "  但需防范双点协同攻击",
        transform=ax.transAxes, fontsize=10.5,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="wheat", edgecolor="#CC6600", alpha=0.8))

plt.tight_layout()
plt.savefig("cascade_threshold_analysis.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("[OK] cascade_threshold_analysis.png (全中文)")

print(f"\n{'='*60}")
print(f"  关键发现")
print(f"{'='*60}")
print(f"  1. α=0: 任何攻击 → 大规模级联 (Top-1 = 77.5%)")
print(f"  2. α≥0.05: Top-1 始终安全")
print(f"  3. Top-2 始终触发级联 (波及全网 10-16%)")
print(f"  4. Braess效应: Top-2 破坏可能大于 Top-5")
print(f"  5. 收益递减: 容量翻10倍 → 级联仅减半")
print(f"{'='*60}")
