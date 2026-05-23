"""Regenerate factor_loading.png — clean, no annotation arrows"""
import pandas as pd, numpy as np, networkx as nx, warnings
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

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

deg_c = nx.degree_centrality(G); btw_c = nx.betweenness_centrality(G, weight="distance")
cls_c = nx.clustering(G, weight="weight")
str_c = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
metrics_df = pd.DataFrame({
    "Degree": pd.Series(deg_c), "Betweenness": pd.Series(btw_c),
    "Clustering": pd.Series(cls_c), "Strength": pd.Series(str_c)
})

ind3 = ["Degree", "Betweenness", "Strength"]
fa = FactorAnalyzer(n_factors=2, rotation="varimax")
col_scaled3 = pd.DataFrame(MinMaxScaler().fit_transform(metrics_df[ind3]), columns=ind3, index=metrics_df.index)
fa.fit(col_scaled3)
var_contrib = fa.get_factor_variance()

loadings = pd.DataFrame(
    fa.loadings_,
    index=["度中心性", "介数中心性", "客流强度"],
    columns=[f"因子1 ({var_contrib[0][0]*100:.0f}%)", f"因子2 ({var_contrib[0][1]*100:.0f}%)"])

fig, ax = plt.subplots(figsize=(8, 4.2))
im = ax.imshow(loadings.values, cmap="RdYlBu", aspect="auto", vmin=-1, vmax=1)
ax.set_xticks([0, 1]); ax.set_yticks(range(3))
ax.set_xticklabels(loadings.columns, fontsize=10)
ax.set_yticklabels(loadings.index, fontsize=10)
for i in range(3):
    for j in range(2):
        val = loadings.values[i, j]
        color = "white" if abs(val) > 0.55 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=12, fontweight="bold", color=color)

cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("因子载荷", fontsize=10)
plt.tight_layout()
plt.savefig("factor_loading.png", dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print("[OK] factor_loading.png (clean)")
