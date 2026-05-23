"""Diagnose Top-5 station ranking issue"""
import pandas as pd, numpy as np, networkx as nx, warnings
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

# Load and build network (same pipeline)
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

# Metrics
deg = nx.degree_centrality(G)
btw = nx.betweenness_centrality(G, weight="distance")
cls = nx.clustering(G, weight="weight")
str_raw = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
raw_degree = {n: G.degree(n) for n in G.nodes()}

df = pd.DataFrame({
    "Degree": pd.Series(deg), "Betweenness": pd.Series(btw),
    "Clustering": pd.Series(cls), "Strength": pd.Series(str_raw),
    "RawDegree": pd.Series(raw_degree)
})

# Build name lookup
name_to_id = {}
for n, d in G.nodes(data=True):
    name_to_id[d["name"]] = n

# Check key stations
check_names = [
    "Peoples Square", "Century Avenue", "Xujiahui",
    "Shanghai Railway Station", "Lujiazui", "Yindu Road",
    "Chunshen Road", "Shanghai Indoor Stadium", "Qufu Road",
    "Hanzhong Road", "Xintiandi", "Yishan Road",
    "Hongqiao Road", "Changshou Road"
]
print("=" * 75)
print(f"{'Station':25s} {'Deg':>6s} {'Btw':>8s} {'Cls':>7s} {'Str':>10s} {'RawDeg':>7s}")
print("-" * 75)
for name in check_names:
    if name in name_to_id:
        nid = name_to_id[name]
        print(f"{name:25s} {deg[nid]:6.4f} {btw[nid]:8.4f} {cls[nid]:7.4f} "
              f"{str_raw[nid]:10.0f} {raw_degree[nid]:7d}")
    else:
        print(f"{name:25s} NOT FOUND")

# Entropy weights with 4 indicators
ind_cols = ["Degree", "Betweenness", "Clustering", "Strength"]
X = df[ind_cols].values
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
w4 = d / d.sum()
print("\n4-indicator entropy weights:")
for col, wi in zip(ind_cols, w4):
    print(f"  {col:15s}: {wi:.4f} ({wi*100:.1f}%)")

scaled4 = pd.DataFrame(
    MinMaxScaler().fit_transform(df[ind_cols]),
    columns=ind_cols, index=df.index
)
df["Imp_4ind"] = sum(w4[j] * scaled4[col] for j, col in enumerate(ind_cols))
top10_4 = df.sort_values("Imp_4ind", ascending=False).head(10)
print("\nTop-10 WITH Clustering (current, BUG):")
for i, (idx, row) in enumerate(top10_4.iterrows()):
    print(f"  {i+1:2d}. {G.nodes[idx]['name']:20s} Imp={row['Imp_4ind']:.4f} "
          f"RawDeg={int(row['RawDegree'])} Btw={row['Betweenness']:.4f} "
          f"Cls={row['Clustering']:.4f} Str={row['Strength']:.0f}")

# Entropy weights with 3 indicators (NO Clustering)
ind3 = ["Degree", "Betweenness", "Strength"]
X3 = df[ind3].values
n3, m3 = X3.shape
X_norm3 = np.zeros_like(X3)
for j in range(m3):
    cmin, cmax = X3[:, j].min(), X3[:, j].max()
    X_norm3[:, j] = (X3[:, j] - cmin) / (cmax - cmin) if cmax - cmin != 0 else 1.0
X_norm3 += 1e-12
P3 = X_norm3 / X_norm3.sum(axis=0, keepdims=True)
k3 = 1.0 / np.log(n3)
e3 = -k3 * np.sum(P3 * np.log(P3), axis=0)
d3 = 1 - e3
w3 = d3 / d3.sum()
print("\n3-indicator entropy weights (NO Clustering):")
for col, wi in zip(ind3, w3):
    print(f"  {col:15s}: {wi:.4f} ({wi*100:.1f}%)")

scaled3 = pd.DataFrame(
    MinMaxScaler().fit_transform(df[ind3]),
    columns=ind3, index=df.index
)
df["Imp_3ind"] = sum(w3[j] * scaled3[col] for j, col in enumerate(ind3))
top10_3 = df.sort_values("Imp_3ind", ascending=False).head(10)
print("\nTop-10 WITHOUT Clustering (FIXED):")
for i, (idx, row) in enumerate(top10_3.iterrows()):
    print(f"  {i+1:2d}. {G.nodes[idx]['name']:20s} Imp={row['Imp_3ind']:.4f} "
          f"RawDeg={int(row['RawDegree'])} Btw={row['Betweenness']:.4f} "
          f"Cls={row['Clustering']:.4f} Str={row['Strength']:.0f}")

print("\n[OK] Diagnosis complete")
