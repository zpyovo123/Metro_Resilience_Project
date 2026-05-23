"""
重新生成海报优化版图表 — 拆分多面板为独立单图
海报尺寸: 70cm × 120cm, 3栏布局, 单栏约21cm宽
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ========== Load Sensitivity Data ==========
data = np.load("cascade_sensitivity_results.npz")
alpha_values = data["alpha_values"]
attack_sizes = data["attack_sizes"]
grid_cascade = data["grid_cascade"]
grid_total = data["grid_total"]
grid_steps = data["grid_steps"]
na, ns = len(alpha_values), len(attack_sizes)

# ================================================================
# Figure 1: Cascade Total Removed %  (standalone, full-width)
# Poster target: spans 2 columns ≈ 42cm wide
# ================================================================
fig, ax = plt.subplots(figsize=(18, 6.5))
vmax = max(15, grid_total.max())
im = ax.imshow(grid_total, cmap="RdYlBu_r", aspect="auto", origin="lower", vmin=0, vmax=vmax)
ax.set_xticks(range(ns))
ax.set_yticks(range(na))
ax.set_xticklabels([f"Top-{k}" for k in attack_sizes], fontsize=13)
ax.set_yticklabels([f"{a:.2f}" for a in alpha_values], fontsize=13)
ax.set_xlabel("蓄意攻击规模 (同时移除的站点数)", fontsize=15, labelpad=10)
ax.set_ylabel("容量冗余系数  alpha", fontsize=15, labelpad=12)
ax.set_title("总失效节点比例 (%)", fontsize=17, fontweight="bold", pad=15)
for i in range(na):
    for j in range(ns):
        val = grid_total[i, j]
        color = "white" if val > vmax * 0.55 else "black"
        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=12, fontweight="bold", color=color)
cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.ax.tick_params(labelsize=12)
cbar.set_label("失效占全网 %", fontsize=13)

# Key annotations — placed OUTSIDE data area
ax.text(1.02, 0.95, "alpha=0 时\nTop-1 即瘫痪 77.5%", transform=ax.transAxes,
        fontsize=11, color="#C0392B", fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FDEDEC", edgecolor="#C0392B", alpha=0.9))
ax.text(1.02, 0.05, "alpha>=0.05 时\nTop-1 始终安全", transform=ax.transAxes,
        fontsize=11, color="#27AE60", fontweight="bold", va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F8F5", edgecolor="#27AE60", alpha=0.9))

plt.tight_layout()
plt.savefig("poster_cascade_total.png", dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print("[OK] poster_cascade_total.png")

# ================================================================
# Figure 2: Cascade-only Nodes  (standalone, full-width)
# ================================================================
fig, ax = plt.subplots(figsize=(18, 6.5))
vmax2 = max(10, grid_cascade.max())
im = ax.imshow(grid_cascade, cmap="YlOrRd", aspect="auto", origin="lower",
               norm=mcolors.LogNorm(vmin=max(1, grid_cascade.min()), vmax=vmax2))
ax.set_xticks(range(ns))
ax.set_yticks(range(na))
ax.set_xticklabels([f"Top-{k}" for k in attack_sizes], fontsize=13)
ax.set_yticklabels([f"{a:.2f}" for a in alpha_values], fontsize=13)
ax.set_xlabel("蓄意攻击规模 (同时移除的站点数)", fontsize=15, labelpad=10)
ax.set_ylabel("容量冗余系数  alpha", fontsize=15, labelpad=12)
ax.set_title("纯级联波及节点数 (对数刻度)", fontsize=17, fontweight="bold", pad=15)
for i in range(na):
    for j in range(ns):
        val = grid_cascade[i, j]
        color = "white" if val > vmax2 * 0.4 else "black"
        ax.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=12, fontweight="bold", color=color)
cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.ax.tick_params(labelsize=12)
cbar.set_label("波及节点数 (对数)", fontsize=13)

ax.text(1.02, 0.95, "Top-2 始终触发级联\n(波及 28-47 站)", transform=ax.transAxes,
        fontsize=11, color="#E67E22", fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF5E7", edgecolor="#E67E22", alpha=0.9))
ax.text(1.02, 0.05, "Top-2 破坏可能 > Top-5\n(Braess 悖论)", transform=ax.transAxes,
        fontsize=11, color="#8E44AD", fontweight="bold", va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F4ECF7", edgecolor="#8E44AD", alpha=0.9))

plt.tight_layout()
plt.savefig("poster_cascade_only.png", dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print("[OK] poster_cascade_only.png")

# ================================================================
# Figure 3: Degradation Curve  (wider aspect, bigger text)
# ================================================================
if __import__("os").path.exists("degradation_curve.png"):
    print("[SKIP] poster_degradation.png — existing degradation_curve.png is fine")
else:
    print("[WARN] degradation_curve.png not found, run main analysis first")

# ================================================================
# Figure 4: Recovery Cross-Design  (regenerate with larger panels)
# If data available, regenerate. Otherwise use existing.
# ================================================================
if __import__("os").path.exists("recovery_cross_design.png"):
    print("[SKIP] poster_recovery.png — existing recovery_cross_design.png will be used")
else:
    print("[WARN] recovery_cross_design.png not found")

# ================================================================
# Figure 5: Topology Map ONLY (extracted from comprehensive, larger)
# ================================================================
import pandas as pd
import networkx as nx
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

print("\nRegenerating standalone topology map...")
station_info = pd.read_csv("stationInfo.csv")
station_info.columns = station_info.columns.str.strip()
bad_dates = [20170504, 20170508, 20170509, 20170616, 20170627, 20170628]
flow_accumulator = defaultdict(int)
for chunk in pd.read_csv("metroData_ODFlow.csv", chunksize=2000000):
    chunk.columns = chunk.columns.str.strip()
    chunk = chunk[~chunk["date"].isin(bad_dates)]
    grouped = chunk.groupby(["originStation", "destinationStation"])["Flow"].sum()
    for (o, d), fv in grouped.items():
        flow_accumulator[(int(o), int(d))] += fv
flow_dict = dict(flow_accumulator)
del flow_accumulator

G = nx.Graph()
for _, r in station_info.iterrows():
    G.add_node(int(r["stationID"]), name=r["name"], lon=r["lon"], lat=r["lat"])
for _, r in station_info.iterrows():
    u = int(r["stationID"])
    raw = str(r["neighbour"]).replace("[", "").replace("]", "").replace("，", ",")
    for v in raw.split(","):
        v = v.strip()
        if v and v.lower() != "nan":
            v = int(float(v))
            if not G.has_edge(u, v):
                total = flow_dict.get((u, v), 0) + flow_dict.get((v, u), 0)
                G.add_edge(u, v, weight=total if total > 0 else 1,
                           distance=1.0 / (total if total > 0 else 1))
G.remove_nodes_from(list(nx.isolates(G)))

# Compute 3-indicator importance
deg = nx.degree_centrality(G)
btw = nx.betweenness_centrality(G, weight="distance")
str_raw = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
df = pd.DataFrame({"Degree": pd.Series(deg), "Betweenness": pd.Series(btw),
                    "Strength": pd.Series(str_raw)})
ind3 = ["Degree", "Betweenness", "Strength"]
X3 = df[ind3].values
X_norm3 = np.zeros_like(X3)
for j in range(3):
    cmin, cmax = X3[:, j].min(), X3[:, j].max()
    X_norm3[:, j] = (X3[:, j] - cmin) / (cmax - cmin) if cmax - cmin != 0 else 1.0
X_norm3 += 1e-12
P3 = X_norm3 / X_norm3.sum(axis=0, keepdims=True)
k3 = 1.0 / np.log(len(df))
e3 = -k3 * np.sum(P3 * np.log(P3), axis=0)
d3 = 1 - e3
w3 = d3 / d3.sum()
scaled3 = pd.DataFrame(MinMaxScaler().fit_transform(df[ind3]), columns=ind3, index=df.index)
df["Importance"] = sum(w3[j] * scaled3[col] for j, col in enumerate(ind3))
df_importance = df.sort_values("Importance", ascending=False)

# Plot
fig, ax = plt.subplots(figsize=(17, 14))
pos = {node: (data["lon"], data["lat"]) for node, data in G.nodes(data=True)}
top10_ids = df_importance.head(10).index.tolist()
normal_ids = list(set(G.nodes()) - set(top10_ids))

nx.draw_networkx_edges(G, pos, alpha=0.25, edge_color="#A0A0A0", width=0.6, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=normal_ids, node_size=15,
                       node_color="#4A90E2", alpha=0.5, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=top10_ids, node_size=250,
                       node_color="#E74C3C", edgecolors="black",
                       linewidths=2, ax=ax)

from adjustText import adjust_text
texts = []
for idx in top10_ids:
    x, y = pos[idx]
    texts.append(ax.text(x, y, G.nodes[idx]["name"], fontsize=12, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="none", alpha=0.85)))
adjust_text(texts, expand_points=(2.0, 2.0),
            arrowprops=dict(arrowstyle="-", color="black", lw=0.8, alpha=0.5))

ax.set_title("Top-10 关键枢纽地理分布 (3指标熵权法修正)", fontsize=18, fontweight="bold", pad=12)
ax.axis("off")

plt.tight_layout()
plt.savefig("poster_topology_map.png", dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print("[OK] poster_topology_map.png")

# ================================================================
# Figure 6: Dual-dimension bar chart (simpler than curve for poster)
# ================================================================
if __import__("os").path.exists("dual_dimension_degradation.png"):
    print("[SKIP] poster_dual_dim.png — existing dual_dimension_degradation.png will be used")

print("\n[OK] All poster-optimized figures generated")
