"""Generate bubble chart: x=degree, y=strength, size=betweenness, color=importance"""
import pandas as pd, numpy as np, networkx as nx, warnings
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from adjustText import adjust_text
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# Load & build
station_info = pd.read_csv("stationInfo.csv"); station_info.columns = station_info.columns.str.strip()
bad_dates = [20170504, 20170508, 20170509, 20170616, 20170627, 20170628]
flow_accumulator = defaultdict(int)
for chunk in pd.read_csv("metroData_ODFlow.csv", chunksize=2000000):
    chunk.columns = chunk.columns.str.strip()
    chunk = chunk[~chunk["date"].isin(bad_dates)]
    grouped = chunk.groupby(["originStation", "destinationStation"])["Flow"].sum()
    for (o, d), fv in grouped.items(): flow_accumulator[(int(o), int(d))] += fv
flow_dict = dict(flow_accumulator)

G = nx.Graph()
for _, r in station_info.iterrows(): G.add_node(int(r["stationID"]), name=r["name"], lon=r["lon"], lat=r["lat"])
for _, r in station_info.iterrows():
    u = int(r["stationID"]); raw = str(r["neighbour"]).replace("[", "").replace("]", "").replace(",", ",")
    for v in raw.split(","):
        v = v.strip()
        if v and v.lower() != "nan":
            v = int(float(v))
            if not G.has_edge(u, v):
                total = flow_dict.get((u, v), 0) + flow_dict.get((v, u), 0)
                G.add_edge(u, v, weight=total if total > 0 else 1, distance=1.0 / (total if total > 0 else 1))
G.remove_nodes_from(list(nx.isolates(G)))

# Metrics
deg_c = nx.degree_centrality(G)
btw_c = nx.betweenness_centrality(G, weight="distance")
cls_c = nx.clustering(G, weight="weight")
str_c = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
raw_deg = {n: G.degree(n) for n in G.nodes()}

metrics_df = pd.DataFrame({
    "Degree": pd.Series(deg_c), "Betweenness": pd.Series(btw_c),
    "Clustering": pd.Series(cls_c), "Strength": pd.Series(str_c),
    "RawDegree": pd.Series(raw_deg)
})

# 3-indicator entropy weight
ind3 = ["Degree", "Betweenness", "Strength"]
X = metrics_df[ind3].values
X_norm = np.zeros_like(X)
for j in range(3):
    cmin, cmax = X[:, j].min(), X[:, j].max()
    X_norm[:, j] = (X[:, j] - cmin) / (cmax - cmin) if cmax - cmin != 0 else 1.0
X_norm += 1e-12; P = X_norm / X_norm.sum(axis=0, keepdims=True)
k = 1.0 / np.log(len(metrics_df))
e = -k * np.sum(P * np.log(P), axis=0); d = 1 - e; w = d / d.sum()
scaled = pd.DataFrame(MinMaxScaler().fit_transform(metrics_df[ind3]), columns=ind3, index=metrics_df.index)
metrics_df["Importance"] = sum(w[j] * scaled[col] for j, col in enumerate(ind3))
df_importance = metrics_df.sort_values("Importance", ascending=False)

# Normalize betweenness to 10-300 for bubble size
btw_min = metrics_df["Betweenness"].min()
btw_max = metrics_df["Betweenness"].max()
metrics_df["BubbleSize"] = 10 + (metrics_df["Betweenness"] - btw_min) / (btw_max - btw_min) * 290

# Plot
fig, ax = plt.subplots(figsize=(11, 7.5))

top5_ids = df_importance.head(5).index
normal_mask = ~metrics_df.index.isin(top5_ids)

# All stations as bubbles
sc = ax.scatter(
    metrics_df.loc[normal_mask, "RawDegree"],
    metrics_df.loc[normal_mask, "Strength"],
    s=metrics_df.loc[normal_mask, "BubbleSize"],
    c=metrics_df.loc[normal_mask, "Importance"],
    cmap="YlOrRd", alpha=0.6, edgecolors="white", linewidth=0.4, vmin=0, vmax=1
)

# Top-5: red border
top5_data = metrics_df.loc[top5_ids]
ax.scatter(
    top5_data["RawDegree"], top5_data["Strength"],
    s=top5_data["BubbleSize"],
    c=top5_data["Importance"],
    cmap="YlOrRd", edgecolors="#E74C3C", linewidth=2.5, zorder=6, vmin=0, vmax=1
)

# Labels for Top-5
texts = []
for idx in top5_ids:
    name = G.nodes[idx]["name"]
    x0 = metrics_df.loc[idx, "RawDegree"]
    y0 = metrics_df.loc[idx, "Strength"]
    texts.append(ax.text(x0 + 0.3, y0, name, fontsize=11, fontweight="bold"))
adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
            expand_points=(4.0, 4.0), force_text=(2.0, 2.0),
            force_points=(0.5, 0.5), expand_text=(1.8, 1.8),
            only_move={'points': 'xy', 'text': 'xy'})

cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
cbar.set_label("综合重要度", fontsize=12)

ax.set_xlabel("节点度（连边数）", fontsize=13)
ax.set_ylabel("客流强度", fontsize=13)
ax.set_title("结构连通度 vs 客流量 vs 介数中心性", fontsize=15, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.35)

# Legend for bubble size
for btw_val, label in [(0.05, "介数低"), (0.20, "介数中"), (0.41, "介数高")]:
    size = 10 + (btw_val - btw_min) / (btw_max - btw_min) * 290
    ax.scatter([], [], s=size, c="gray", alpha=0.4, edgecolors="black", linewidth=0.5,
               label=f"{label} ({btw_val:.2f})")
ax.legend(title="气泡大小 = 介数中心性", fontsize=9, title_fontsize=10, loc="upper left")

plt.tight_layout()
plt.savefig("degree_vs_strength.png", dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print("[OK] degree_vs_strength.png (bubble chart)")
