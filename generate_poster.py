"""
学术会议海报生成 — 70×120cm 竖版
城市轨道交通网络韧性分析
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# Poster dimensions: 70cm × 120cm → 27.56" × 47.24"
# Use 200 DPI for print quality
FIG_W, FIG_H = 27.56, 47.24  # inches = 70cm × 120cm
DPI = 100  # ≈ 2756 × 4724 px, good balance for print

os.chdir(r"C:\Users\93966\Metro_Resilience_Project")

# ============ Figure Setup ============
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")

# Use GridSpec: 12 rows, 2 columns
gs = GridSpec(12, 2, figure=fig,
              left=0.05, right=0.95, top=0.98, bottom=0.02,
              hspace=0.25, wspace=0.18,
              height_ratios=[1.2, 1.5, 2.8, 2.8, 2.8, 2.8, 2.5, 1.5, 0.6, 0.6, 0.6, 0.4])

# ============ Color Palette ============
C_PRIMARY   = "#1A3A5C"   # dark navy — titles
C_ACCENT    = "#E74C3C"   # red — highlights
C_BLUE      = "#2980B9"   # blue — structural dimension
C_GREEN     = "#27AE60"   # green — passenger dimension
C_LIGHT_BG  = "#F8F9FA"   # light gray — section backgrounds
C_BORDER    = "#D5D8DC"   # light border
C_DARK_TEXT = "#2C3E50"   # main text
C_GRAY_TEXT = "#7F8C8D"   # secondary text

def add_section_bg(ax, color=C_LIGHT_BG):
    """Add subtle background to a section"""
    ax.set_facecolor(color)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(C_BORDER)
        spine.set_linewidth(0.5)

def section_title(ax, number, text, y=0.92):
    """Add a numbered section title"""
    ax.text(0.02, y, f"{number}. {text}",
            transform=ax.transAxes, fontsize=15, fontweight="bold",
            color=C_PRIMARY, va="top")

def body_text(ax, text, x=0.04, y=0.78, fontsize=12, color=C_DARK_TEXT, line_spacing=1.55):
    """Add body text with consistent formatting"""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize,
            color=color, va="top", linespacing=line_spacing)

# ================================================================
# ROW 1: TITLE BANNER (spans both columns, height_ratio=1.2)
# ================================================================
ax_title = fig.add_subplot(gs[0, :])
ax_title.set_facecolor(C_PRIMARY)
ax_title.set_xticks([]); ax_title.set_yticks([])
for spine in ax_title.spines.values():
    spine.set_visible(False)

ax_title.text(0.5, 0.62, "城市轨道交通网络韧性分析",
              transform=ax_title.transAxes, fontsize=32, fontweight="bold",
              color="white", ha="center", va="center")
ax_title.text(0.5, 0.28, "融合复杂网络理论与动态客流数据的级联失效评估框架",
              transform=ax_title.transAxes, fontsize=16,
              color="#BDC3C7", ha="center", va="center",
              fontstyle="italic")

# Author line
ax_title.text(0.5, 0.08, "作者 / 单位（在此填写）",
              transform=ax_title.transAxes, fontsize=11,
              color="#95A5A5", ha="center", va="center")

# ================================================================
# ROW 2: Left - Research Background | Right - Methodology (h_ratio=1.5)
# ================================================================
ax_bg = fig.add_subplot(gs[1, 0])
add_section_bg(ax_bg)
ax_bg.set_xticks([]); ax_bg.set_yticks([])
section_title(ax_bg, "1", "研究背景与问题", y=0.90)
body_text(ax_bg,
    "城市轨道交通网络规模持续扩大，自然灾害、设备故障及突发大客流\n"
    "等扰动事件频发，严重时可导致局部甚至全网运行中断。\n\n"
    "现有研究局限:\n"
    "  • 多基于静态拓扑结构，较少考虑动态客流影响\n"
    "  • 难以全面反映扰动下乘客出行服务的实际变化\n\n"
    "研究问题:\n"
    "  • 纳入AFC客流数据后，网络韧性评估是否显著不同？\n"
    "  • 蓄意攻击下是否触发级联失效？何种条件最危险？\n"
    "  • 灾后如何优化抢修顺序以最快恢复网络性能？",
    fontsize=10.5, x=0.05, y=0.78, line_spacing=1.5)

ax_method = fig.add_subplot(gs[1, 1])
add_section_bg(ax_method)
ax_method.set_xticks([]); ax_method.set_yticks([])
section_title(ax_method, "2", "方法论框架", y=0.90)
body_text(ax_method,
    "① Space-L拓扑建模\n"
    "   站点→节点，同线邻站→边 (302站 349边)\n\n"
    "② AFC客流加权\n"
    "   边权重=双向客流总量，距离=1/客流\n\n"
    "③ 熵权法客观赋权（修正版）\n"
    "   Degree + Betweenness + Strength → 综合重要度\n"
    "   权重: 度13.8% | 介数44.2% | 客流42.0%\n\n"
    "④ 级联失效模型\n"
    "   L_i = 加权介数×总客流，C_i = L_i×(1+α)\n"
    "   攻击→最短路径重分配→超载移除→迭代\n\n"
    "⑤ 韧性退化/恢复双阶段模拟\n"
    "   3种攻击 × 3种抢修策略交叉对比",
    fontsize=10.5, x=0.05, y=0.78, line_spacing=1.4)

# ================================================================
# ROW 3: FULL-WIDTH FIGURE — Degradation Curve (h_ratio=2.8)
# ================================================================
ax_fig1 = fig.add_subplot(gs[2, :])
add_section_bg(ax_fig1)
section_title(ax_fig1, "3", "网络韧性退化曲线 — 三种攻击策略对比", y=0.97)

if os.path.exists("degradation_curve.png"):
    img1 = mpimg.imread("degradation_curve.png")
    ax_fig1.imshow(img1, aspect="auto", extent=[0.02, 0.98, 0.02, 0.85])
ax_fig1.set_xticks([]); ax_fig1.set_yticks([])

body_text(ax_fig1,
    "发现: 蓄意攻击(按综合重要度)曲线下降最快 — 仅移除20%站点,网络效率骤降至11.5%。"
    "随机攻击影响较小(36.1%)。证明纳入客流信息后,韧性评估更敏感、更贴近实际运营。",
    x=0.03, y=0.06, fontsize=10.5, color=C_DARK_TEXT)

# ================================================================
# ROW 4: FULL-WIDTH — Cascade Sensitivity Heatmap (h_ratio=2.8)
# ================================================================
ax_fig2 = fig.add_subplot(gs[3, :])
add_section_bg(ax_fig2)
section_title(ax_fig2, "4", "级联失效敏感性分析 — α × 攻击规模扫描", y=0.97)

if os.path.exists("cascade_sensitivity_heatmap.png"):
    img2 = mpimg.imread("cascade_sensitivity_heatmap.png")
    ax_fig2.imshow(img2, aspect="auto", extent=[0.02, 0.98, 0.02, 0.85])
ax_fig2.set_xticks([]); ax_fig2.set_yticks([])

body_text(ax_fig2,
    "关键发现: (a) α=0时任何攻击触发大规模级联(Top-1即瘫痪77.5%)。"
    "(b) α≥0.05后Top-1永远安全,但Top-2始终触发级联(波及10-16%网络)。"
    "(c) Braess悖论效应: Top-3攻击的总失效可能少于Top-2。"
    "(d) 收益递减: α提升10倍仅使级联减半。",
    x=0.03, y=0.06, fontsize=10.5, color=C_DARK_TEXT)

# ================================================================
# ROW 5: FULL-WIDTH — Recovery Cross-Design (h_ratio=2.8)
# ================================================================
ax_fig3 = fig.add_subplot(gs[4, :])
add_section_bg(ax_fig3)
section_title(ax_fig3, "5", "恢复策略交叉对比 — 抢修顺序优化", y=0.97)

if os.path.exists("recovery_cross_design.png"):
    img3 = mpimg.imread("recovery_cross_design.png")
    ax_fig3.imshow(img3, aspect="auto", extent=[0.02, 0.98, 0.02, 0.85])
ax_fig3.set_xticks([]); ax_fig3.set_yticks([])

body_text(ax_fig3,
    "发现: 最优抢修策略取决于攻击模式 — 随机攻击后按重要度抢修最优(AUC=0.796),"
    "蓄意攻击后按度中心性抢修更优(AUC=0.475)。按综合重要度抢修较随机抢修效率提升10-26%。",
    x=0.03, y=0.06, fontsize=10.5, color=C_DARK_TEXT)

# ================================================================
# ROW 6: TWO-COLUMN — Topology Map + Dual Dimension (h_ratio=2.8)
# ================================================================
ax_fig4a = fig.add_subplot(gs[5, 0])
add_section_bg(ax_fig4a)
section_title(ax_fig4a, "6a", "关键枢纽地理分布", y=0.97)
ax_fig4a.set_xticks([]); ax_fig4a.set_yticks([])

# Load comprehensive visualization and crop topology part
if os.path.exists("comprehensive_visualization.png"):
    img_cv = mpimg.imread("comprehensive_visualization.png")
    # Show full image
    ax_fig4a.imshow(img_cv, aspect="auto", extent=[0.02, 0.98, 0.02, 0.88])

body_text(ax_fig4a,
    "修正后Top-5: 曲阜路、汉中路、人民广场、宜山路、新天地",
    x=0.03, y=0.06, fontsize=10, color=C_DARK_TEXT)

ax_fig4b = fig.add_subplot(gs[5, 1])
add_section_bg(ax_fig4b)
section_title(ax_fig4b, "6b", "双维度韧性拆解", y=0.97)
ax_fig4b.set_xticks([]); ax_fig4b.set_yticks([])

if os.path.exists("dual_dimension_degradation.png"):
    img_dual = mpimg.imread("dual_dimension_degradation.png")
    ax_fig4b.imshow(img_dual, aspect="auto", extent=[0.02, 0.98, 0.02, 0.88])

body_text(ax_fig4b,
    "蓄意攻击下客流维度更脆弱,随机攻击下结构维度更脆弱(差距2-4%)",
    x=0.03, y=0.06, fontsize=10, color=C_DARK_TEXT)

# ================================================================
# ROW 7: FULL-WIDTH — Conclusions (h_ratio=2.5)
# ================================================================
ax_conc = fig.add_subplot(gs[6, :])
add_section_bg(ax_conc)
ax_conc.set_xticks([]); ax_conc.set_yticks([])
section_title(ax_conc, "7", "结论与建议", y=0.95)

conclusions = [
    ("⚡ 客流加权评估更准确",
     "AFC客流加权的网络效率对蓄意攻击的敏感度远超纯拓扑指标(11.5% vs 13.6%剩余效率),"
     "证明纳入动态客流信息能够更准确刻画扰动事件对乘客出行服务的综合影响。"),

    ("🛡 单点鲁棒,双点脆弱",
     "上海地铁对单站故障具有天然韧性(Top-1永不触发级联,α≥0.05)。"
     "但银都路+上海体育馆双点协同攻击始终触发级联(波及45站/15%),需重点防范。"),

    ("📈 优化抢修策略可提速10-26%",
     "灾后抢修不应盲目随机: 按综合重要度或度中心性排序抢修,"
     "网络效率恢复速度可提升10-26%,尤其在蓄意攻击场景下效果显著。"),

    ("🔬 Braess悖论效应",
     "移除更多站点可能减少总级联损失 — 主动移除某些'级联受害者'可吸收重分配客流,"
     "为应急管制提供反直觉的决策参考。"),
]

for i, (title, desc) in enumerate(conclusions):
    y_pos = 0.78 - i * 0.17
    # Colored bullet
    ax_conc.text(0.03, y_pos + 0.02, "●", transform=ax_conc.transAxes,
                fontsize=14, color=C_ACCENT if i < 2 else C_BLUE, va="center")
    ax_conc.text(0.06, y_pos + 0.02, title, transform=ax_conc.transAxes,
                fontsize=13, fontweight="bold", color=C_PRIMARY, va="center")
    ax_conc.text(0.06, y_pos - 0.04, desc, transform=ax_conc.transAxes,
                fontsize=10.5, color=C_DARK_TEXT, va="top", linespacing=1.4)

# ================================================================
# ROW 8-11: Bottom metadata (h_ratios = 1.5, 0.6, 0.6, 0.6, 0.4)
# ================================================================
# Potential applications
ax_app = fig.add_subplot(gs[7, :])
add_section_bg(ax_app)
ax_app.set_xticks([]); ax_app.set_yticks([])
section_title(ax_app, "8", "应用价值", y=0.88)
body_text(ax_app,
    "• 关键设施保护: 识别Top-5枢纽站点,针对性加强物理防护与客流监控\n"
    "• 应急调度决策: 灾后按综合重要度排序抢修,而非盲目随机响应\n"
    "• 冗余规划: α≥0.15即可避免单点级联,为新建线路容量设计提供定量参考\n"
    "• 双点防御: 重点防范银都路-上海体育馆等关键站点对的同时失效",
    x=0.04, y=0.70, fontsize=10.5, line_spacing=1.5)

# Data & Methods note
ax_note = fig.add_subplot(gs[8, :])
ax_note.set_xticks([]); ax_note.set_yticks([])
for spine in ax_note.spines.values():
    spine.set_visible(False)
body_text(ax_note,
    "数据来源: 上海轨道交通AFC刷卡数据 (metroData_ODFlow.csv, 11GB, 约2.6亿条记录) | "
    "站点信息: stationInfo.csv (302站) | "
    "分析工具: Python + NetworkX + Scikit-learn | "
    "海报录用号: [填写]",
    x=0.02, y=0.7, fontsize=9.5, color=C_GRAY_TEXT)

# Contact / QR placeholder
ax_qr = fig.add_subplot(gs[9, :])
ax_qr.set_xticks([]); ax_qr.set_yticks([])
for spine in ax_qr.spines.values():
    spine.set_visible(False)
ax_qr.text(0.02, 0.5, "联系方式: [邮箱]  |  [二维码预留位置]",
           transform=ax_qr.transAxes, fontsize=9.5, color=C_GRAY_TEXT, va="center")

# ============ SAVE ============
out_path = "conference_poster.png"
print(f"正在保存 {FIG_W}×{FIG_H} 英寸 @ {DPI} DPI ...")
plt.savefig(out_path, dpi=DPI, facecolor="white", edgecolor="none",
            bbox_inches="tight", pad_inches=0.3)
plt.close()
print(f"[OK] 海报已保存: {out_path}")
print(f"     尺寸: {FIG_W*2.54:.1f}×{FIG_H*2.54:.1f} cm @ {DPI} DPI")
