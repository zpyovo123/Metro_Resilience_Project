"""
拆 comprehensive_visualization → 3 张独立图 + 逻辑修复
1. factor_loading.png       — 4指标因子载荷 (含Clustering, 展示为何双维度)
2. degree_vs_strength.png   — 原始度 vs 客流强度散点
3. topology_map.png         — Top-10 关键枢纽地理分布
"""
import pandas as pd, numpy as np, networkx as nx, warnings
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from adjustText import adjust_text
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============ Load & Build ============
print("[1] 加载数据...")
station_info = pd.read_csv("stationInfo.csv")
station_info.columns = station_info.columns.str.strip()
bad_dates = [20170504, 20170508, 20170509, 20170616, 20170627, 20170628]
flow_accumulator = defaultdict(int)
for cnt, chunk in enumerate(pd.read_csv("metroData_ODFlow.csv", chunksize=2000000)):
    chunk.columns = chunk.columns.str.strip()
    chunk = chunk[~chunk["date"].isin(bad_dates)]
    grouped = chunk.groupby(["originStation", "destinationStation"])["Flow"].sum()
    for (o, d), fv in grouped.items():
        flow_accumulator[(int(o), int(d))] += fv
flow_dict = dict(flow_accumulator)
del flow_accumulator

print("[2] 构建 Space-L 网络...")
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
print(f"  网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

print("[3] 计算指标...")
deg_c = nx.degree_centrality(G)
btw_c = nx.betweenness_centrality(G, weight="distance")
cls_c = nx.clustering(G, weight="weight")
str_c = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
metrics_df = pd.DataFrame({
    "Degree": pd.Series(deg_c), "Betweenness": pd.Series(btw_c),
    "Clustering": pd.Series(cls_c), "Strength": pd.Series(str_c),
    "RawDegree": pd.Series({n: G.degree(n) for n in G.nodes()})
})

# 3指标熵权（重要度计算用）
ind3 = ["Degree", "Betweenness", "Strength"]
X3 = metrics_df[ind3].values
X_norm3 = np.zeros_like(X3)
for j in range(3):
    cmin, cmax = X3[:, j].min(), X3[:, j].max()
    X_norm3[:, j] = (X3[:, j] - cmin) / (cmax - cmin) if cmax - cmin != 0 else 1.0
X_norm3 += 1e-12; P3 = X_norm3 / X_norm3.sum(axis=0, keepdims=True)
k3 = 1.0 / np.log(len(metrics_df))
e3 = -k3 * np.sum(P3 * np.log(P3), axis=0); d3 = 1 - e3; w3 = d3 / d3.sum()
scaled3 = pd.DataFrame(MinMaxScaler().fit_transform(metrics_df[ind3]), columns=ind3, index=metrics_df.index)
metrics_df["Importance"] = sum(w3[j] * scaled3[col] for j, col in enumerate(ind3))
df_importance = metrics_df.sort_values("Importance", ascending=False)

print(f"  熵权: Deg={w3[0]:.3f} Btw={w3[1]:.3f} Str={w3[2]:.3f}")

# ================================================================
# Figure 1: 因子载荷热力图（4指标，含Clustering）
# 修正: 用4指标做因子分析才有意义，展示"结构维"vs"客流维"的天然分离
# ================================================================
print("\n[4] 图1: 因子载荷热力图（4指标）...")
ind4 = ["Degree", "Betweenness", "Clustering", "Strength"]
fa = FactorAnalyzer(n_factors=2, rotation="varimax")
col_scaled4 = pd.DataFrame(
    MinMaxScaler().fit_transform(metrics_df[ind4]), columns=ind4, index=metrics_df.index)
fa.fit(col_scaled4)

# 方差贡献
var_contrib = fa.get_factor_variance()
f1_var = var_contrib[0][0] * 100
f2_var = var_contrib[0][1] * 100
total_var = var_contrib[2][0] * 100

loadings = pd.DataFrame(
    fa.loadings_,
    index=["度中心性", "介数中心性", "聚类系数", "客流强度"],
    columns=[f"因子1\n({f1_var:.0f}%方差)", f"因子2\n({f2_var:.0f}%方差)"])

fig, ax = plt.subplots(figsize=(11, 7))
im = ax.imshow(loadings.values, cmap="RdYlBu", aspect="auto", vmin=-1, vmax=1)
ax.set_xticks([0, 1]); ax.set_yticks(range(4))
ax.set_xticklabels(loadings.columns, fontsize=11)
ax.set_yticklabels(loadings.index, fontsize=11)
for i in range(4):
    for j in range(2):
        val = loadings.values[i, j]
        color = "white" if abs(val) > 0.55 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=13, fontweight="bold", color=color)

# 标注两个维度
ax.annotate("结构供给\n维度",
            xy=(0.15, 0.8), xytext=(-0.6, 0.8),
            fontsize=12, color="#C0392B", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.8))
ax.annotate("客流服务\n维度",
            xy=(0.85, 3.2), xytext=(-0.6, 3.2),
            fontsize=12, color="#2471A3", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2471A3", lw=1.8))

ax.set_title(f"因子分析: 4指标 → 2个潜在维度 (累计方差{total_var:.0f}%)",
             fontsize=14, fontweight="bold", pad=15)
cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("因子载荷", fontsize=11)
plt.tight_layout()
plt.savefig("factor_loading.png", dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print("  -> factor_loading.png")

# ================================================================
# Figure 2: 原始度 vs 客流强度散点图
# ================================================================
print("[5] 图2: 原始度 vs 客流强度散点...")
top5 = df_importance.head(5)
fig, ax = plt.subplots(figsize=(10, 7))

normal_mask = ~metrics_df.index.isin(top5.index)
ax.scatter(metrics_df.loc[normal_mask, "RawDegree"],
           metrics_df.loc[normal_mask, "Strength"],
           c="#B0C4DE", alpha=0.55, edgecolors="white", s=70,
           label="普通站点 (297站)")

ax.scatter(top5["RawDegree"], top5["Strength"],
           c="#E74C3C", marker="*", s=400, edgecolors="black",
           linewidths=1.5, label="Top-5 关键枢纽", zorder=6)

texts = []
for idx in top5.index:
    name = G.nodes[idx]["name"]
    texts.append(ax.text(top5.loc[idx, "RawDegree"],
                         top5.loc[idx, "Strength"],
                         name, fontsize=11, fontweight="bold",
                         ha="center", va="bottom"))
adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
            expand_points=(3.5, 3.5), force_text=(1.5, 1.5),
            force_points=(0.3, 0.3))

ax.set_xlabel("节点原始度（连边数）", fontsize=13)
ax.set_ylabel("节点客流强度", fontsize=13)
ax.set_title("结构连通度 vs 实际客流量 — Top-5 关键枢纽", fontsize=15, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("degree_vs_strength.png", dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print("  -> degree_vs_strength.png")

# ================================================================
# Figure 3: 地理拓扑图（Top-10）
# ================================================================
print("[6] 图3: 地理拓扑图...")
top10_ids = df_importance.head(10).index.tolist()
fig, ax = plt.subplots(figsize=(16, 13))

pos = {node: (data["lon"], data["lat"]) for node, data in G.nodes(data=True)}
normal_ids = list(set(G.nodes()) - set(top10_ids))

nx.draw_networkx_edges(G, pos, alpha=0.22, edge_color="#A0A0A0", width=0.6, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=normal_ids, node_size=14,
                       node_color="#4A90E2", alpha=0.5, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=top10_ids, node_size=240,
                       node_color="#E74C3C", edgecolors="black",
                       linewidths=1.8, ax=ax, )

texts = []
for idx in top10_ids:
    x, y = pos[idx]
    imp_val = metrics_df.loc[idx, "Importance"]
    rank = df_importance.index.tolist().index(idx) + 1
    texts.append(ax.text(x, y, f"#{rank} {G.nodes[idx]['name']}",
                         fontsize=10, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.25",
                                   facecolor="white", edgecolor="none", alpha=0.85)))
adjust_text(texts, expand_points=(2.2, 2.2),
            arrowprops=dict(arrowstyle="-", color="black", lw=0.8, alpha=0.5))

ax.set_title("Top-10 关键枢纽地理分布 (3指标熵权法修正)", fontsize=17, fontweight="bold", pad=12)
ax.axis("off")
plt.tight_layout()
plt.savefig("topology_map.png", dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print("  -> topology_map.png")

# ================================================================
print("\n[完成] 3张独立图已生成")
print("  factor_loading.png      — 4指标因子载荷热力图")
print("  degree_vs_strength.png  — 原始度 vs 客流强度散点")
print("  topology_map.png        — Top-10 关键枢纽地理分布")
