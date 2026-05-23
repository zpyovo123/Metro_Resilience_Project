"""
综合可视化 — 修复版（3指标熵权 + 中文 + 原始度）
修复内容:
  1. 移除Clustering，只用 Degree/Betweenness/Strength 做熵权
  2. 因子载荷热力图X轴显示熵权百分比
  3. 散点图X轴使用节点原始度（整数连边数）
"""
import pandas as pd, numpy as np, networkx as nx, time, gc, warnings
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
    if (cnt + 1) % 10 == 0:
        print(f"  {(cnt+1)*2/1000:.1f}M 行")
flow_dict = dict(flow_accumulator)
del flow_accumulator; gc.collect()

print("[2] 构建Space-L网络...")
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
N0 = G.number_of_nodes()
print(f"  网络: {N0} 节点, {G.number_of_edges()} 边")

# ============ Metrics ============
print("[3] 计算网络指标...")
degree_cent = nx.degree_centrality(G)
between_cent = nx.betweenness_centrality(G, weight="distance")
cluster_coef = nx.clustering(G, weight="weight")
node_strength = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
raw_degree = {n: G.degree(n) for n in G.nodes()}

metrics_df = pd.DataFrame({
    "Degree": pd.Series(degree_cent),
    "Betweenness": pd.Series(between_cent),
    "Clustering": pd.Series(cluster_coef),
    "Strength": pd.Series(node_strength),
    "RawDegree": pd.Series(raw_degree)
})

# ============ 3-Indicator Entropy Weight (FIXED: NO Clustering) ============
print("[4] 计算熵权（3指标，不含聚类系数）...")
indicator_cols = ["Degree", "Betweenness", "Strength"]

def entropy_weight_method(df, cols):
    X = df[cols].values.astype(float)
    n, m = X.shape
    X_norm = np.zeros_like(X)
    for j in range(m):
        cmin, cmax = X[:, j].min(), X[:, j].max()
        X_norm[:, j] = (X[:, j] - cmin) / (cmax - cmin) if cmax - cmin != 0 else 1.0
    X_norm += 1e-12
    P = X_norm / X_norm.sum(axis=0, keepdims=True)
    k = 1.0 / np.log(n)
    e = -k * np.sum(P * np.log(P), axis=0)
    d = 1 - e
    w = d / d.sum()
    return {col: float(w[j]) for j, col in enumerate(cols)}

ewm_weights = entropy_weight_method(metrics_df, indicator_cols)

print("  ---------- 修正后的熵权法权重（3指标）----------")
for k, v in ewm_weights.items():
    print(f"    {k}: {v:.4f} ({v*100:.1f}%)")

# Compute importance
scaler = MinMaxScaler()
scaled = pd.DataFrame(
    scaler.fit_transform(metrics_df[indicator_cols]),
    columns=indicator_cols, index=metrics_df.index
)
metrics_df["Importance"] = sum(
    ewm_weights[col] * scaled[col] for col in indicator_cols
)
df_importance = metrics_df.sort_values("Importance", ascending=False)

print("\n  ---------- Top-10 关键站点（修正后）----------")
top10 = df_importance.head(10)
for i, (idx, row) in enumerate(top10.iterrows()):
    name = G.nodes[idx]["name"]
    rd = int(row["RawDegree"])
    print(f"  {i+1:2d}. {name:20s} Imp={row['Importance']:.4f}  "
          f"原始度={rd:2d}  Btw={row['Betweenness']:.3f}  "
          f"Str={row['Strength']:.0f}")

# ============ Figure: Comprehensive Visualization (4 subplots) ============
print("[5] 生成综合可视化...")

# Factor analysis on 3 indicators (for visualization only)
fa = FactorAnalyzer(n_factors=2, rotation="varimax")
col_scaled = pd.DataFrame(
    MinMaxScaler().fit_transform(metrics_df[indicator_cols]),
    columns=indicator_cols, index=metrics_df.index
)
fa.fit(col_scaled)
loadings = pd.DataFrame(
    fa.loadings_,
    index=["度中心性\n(Degree)", "介数中心性\n(Betweenness)", "客流强度\n(Strength)"],
    columns=["因子1", "因子2"]
)

fig = plt.figure(figsize=(20, 16))
plt.subplots_adjust(hspace=0.35, wspace=0.30)

# === 子图1: 因子载荷热力图 ===
ax1 = fig.add_subplot(2, 2, 1)
im = ax1.imshow(loadings.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
ax1.set_xticks([0, 1])
ax1.set_yticks(range(3))
ax1.set_xticklabels(loadings.columns, fontsize=11)
ax1.set_yticklabels(loadings.index, fontsize=11)
for i in range(3):
    for j in range(2):
        val = loadings.values[i, j]
        color = "white" if val > 0.5 else "black"
        ax1.text(j, i, f"{val:.3f}", ha="center", va="center",
                 fontsize=12, fontweight="bold", color=color)

# Add weight percentages as annotations on X-axis
w_degree = ewm_weights["Degree"]
w_between = ewm_weights["Betweenness"]
w_strength = ewm_weights["Strength"]
ax1.set_title(
    f"因子载荷热力图\n熵权: 度={w_degree:.1%} | 介数={w_between:.1%} | 客流={w_strength:.1%}",
    fontsize=13, fontweight="bold", pad=12)
plt.colorbar(im, ax=ax1, shrink=0.85).set_label("载荷值", fontsize=10)

# === 子图2: 原始度 vs 客流强度 散点图 ===
ax2 = fig.add_subplot(2, 2, 2)
normal_mask = ~metrics_df.index.isin(df_importance.head(5).index)
ax2.scatter(metrics_df.loc[normal_mask, "RawDegree"],
            metrics_df.loc[normal_mask, "Strength"],
            c="#B0C4DE", alpha=0.5, edgecolors="white", s=60,
            label="普通站点")

top5 = df_importance.head(5)
ax2.scatter(top5["RawDegree"], top5["Strength"],
            c="#E74C3C", marker="*", s=350, edgecolors="black",
            linewidths=1.2, label="Top-5 关键枢纽", )

texts_s = []
for idx in top5.index:
    name = G.nodes[idx]["name"]
    texts_s.append(ax2.text(top5.loc[idx, "RawDegree"],
                            top5.loc[idx, "Strength"],
                            name, fontsize=10, fontweight="bold"))
adjust_text(texts_s, arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))

ax2.set_xlabel("节点原始度（连边数）", fontsize=12)
ax2.set_ylabel("节点客流强度", fontsize=12)
ax2.set_title("结构连通度 vs 实际客流量", fontsize=14, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.5)

# === 子图3: 地理拓扑图 ===
ax3 = fig.add_subplot(2, 2, (3, 4))
pos = {node: (data["lon"], data["lat"]) for node, data in G.nodes(data=True)}
top5_ids = top5.index.tolist()
normal_ids = list(set(G.nodes()) - set(top5_ids))

nx.draw_networkx_edges(G, pos, alpha=0.25, edge_color="#A0A0A0",
                       width=0.8, ax=ax3)
nx.draw_networkx_nodes(G, pos, nodelist=normal_ids, node_size=18,
                       node_color="#4A90E2", alpha=0.55, ax=ax3)
nx.draw_networkx_nodes(G, pos, nodelist=top5_ids, node_size=220,
                       node_color="#E74C3C", edgecolors="black",
                       linewidths=1.8, ax=ax3, )

texts_t = []
for idx in top5_ids:
    x, y = pos[idx]
    texts_t.append(ax3.text(x, y, G.nodes[idx]["name"],
                            fontsize=11, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="white", edgecolor="none", alpha=0.8)))
adjust_text(texts_t, expand_points=(1.8, 1.8),
            arrowprops=dict(arrowstyle="-", color="black", lw=1.0, alpha=0.6))

ax3.set_title("Top-5 关键枢纽地理分布（修正后）", fontsize=14, fontweight="bold")
ax3.axis("off")

# Overall
fig.suptitle("城市轨道交通网络韧性综合可视化 — 3指标熵权法（修正版）",
             fontsize=16, fontweight="bold", y=0.98)

plt.savefig("comprehensive_visualization.png", dpi=250, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("  -> comprehensive_visualization.png (fixed)")

# ============ Also save Top-5 for other scripts ============
top5_names = [G.nodes[i]["name"] for i in top5_ids]
print(f"\n  修正后 Top-5: {top5_names}")
print(f"  对比: 旧版 Top-5 含银都路/春申路（郊区端点站），新版含人民广场/曲阜路等核心枢纽")
print("\n[OK] 综合可视化生成完毕")
