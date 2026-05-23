"""Quick check Top-8 destructiveness coordinates"""
import pandas as pd, numpy as np, networkx as nx, warnings
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

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
    raw = str(r["neighbour"]).replace("[", "").replace("]", "").replace(",", ",")
    for v in raw.split(","):
        v = v.strip()
        if v and v.lower() != "nan":
            v = int(float(v))
            if not G.has_edge(u, v):
                total = flow_dict.get((u, v), 0) + flow_dict.get((v, u), 0)
                G.add_edge(u, v, weight=total if total > 0 else 1,
                           distance=1.0 / (total if total > 0 else 1))
G.remove_nodes_from(list(nx.isolates(G)))

deg = nx.degree_centrality(G)
btw = nx.betweenness_centrality(G, weight="distance")
strength = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
df = pd.DataFrame({"Degree": pd.Series(deg), "Betweenness": pd.Series(btw), "Strength": pd.Series(strength)})
ind3 = ["Degree", "Betweenness", "Strength"]
X = df[ind3].values
X_norm = np.zeros_like(X)
for j in range(3):
    cmin, cmax = X[:, j].min(), X[:, j].max()
    X_norm[:, j] = (X[:, j] - cmin) / (cmax - cmin) if cmax - cmin != 0 else 1.0
X_norm += 1e-12
P = X_norm / X_norm.sum(axis=0, keepdims=True)
k = 1.0 / np.log(len(df))
e = -k * np.sum(P * np.log(P), axis=0)
d = 1 - e
w = d / d.sum()
scaled = pd.DataFrame(MinMaxScaler().fit_transform(df[ind3]), columns=ind3, index=df.index)
df["Importance"] = sum(w[j] * scaled[col] for j, col in enumerate(ind3))
df_importance = df.sort_values("Importance", ascending=False)

def cascading_failure(G_original, target_nodes, alpha):
    G = G_original.copy()
    bc_init = nx.betweenness_centrality(G, weight="distance")
    total_flow = sum(d["weight"] for _, _, d in G.edges(data=True))
    capacity = {n: bc_init[n] * total_flow * (1 + alpha) for n in G.nodes()}
    actual = [n for n in target_nodes if n in G]
    G.remove_nodes_from(actual)
    all_removed = list(actual)
    steps = 0
    while steps < 100 and G.number_of_nodes() > 1:
        steps += 1
        try:
            bc_new = nx.betweenness_centrality(G, weight="distance")
        except Exception:
            break
        current_load = {n: bc_new[n] * total_flow for n in G.nodes()}
        overloaded = [n for n in G.nodes() if current_load[n] > capacity[n]]
        if not overloaded:
            break
        G.remove_nodes_from(overloaded)
        all_removed.extend(overloaded)
    return len(all_removed) - len(actual), len(all_removed), steps

top30 = df_importance.head(30).index.tolist()
results = []
for nid in top30:
    cn, tot, st = cascading_failure(G, [nid], 0.2)
    results.append({"name": G.nodes[nid]["name"], "importance": df.loc[nid, "Importance"],
                    "total": tot, "cascade": cn})

df_dest = pd.DataFrame(results).sort_values("total", ascending=False)

print("Top-8 by destructiveness (x=importance, y=total):")
for i, (_, row) in enumerate(df_dest.head(8).iterrows()):
    print(f"  {i+1}. {row['name']:30s}  x={row['importance']:.4f}  y={int(row['total'])}")
