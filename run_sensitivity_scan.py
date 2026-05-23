"""级联失效敏感性扫描 — 独立运行脚本"""
import pandas as pd, numpy as np, networkx as nx, time, gc, warnings
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

# ============ Part A: Load & Build ============
print("[1] Loading station info...")
station_info = pd.read_csv("stationInfo.csv")
station_info.columns = station_info.columns.str.strip()

print("[2] Loading OD flow...")
bad_dates = [20170504, 20170508, 20170509, 20170616, 20170627, 20170628]
flow_accumulator = defaultdict(int)
for chunk_count, chunk in enumerate(pd.read_csv("metroData_ODFlow.csv", chunksize=2000000)):
    chunk.columns = chunk.columns.str.strip()
    if "date" in chunk.columns:
        chunk = chunk[~chunk["date"].isin(bad_dates)]
    grouped = chunk.groupby(["originStation", "destinationStation"])["Flow"].sum()
    for (o, d), flow_val in grouped.items():
        flow_accumulator[(int(o), int(d))] += flow_val
    if (chunk_count + 1) % 10 == 0:
        print(f"  {(chunk_count + 1) * 2 / 1000:.1f}M rows processed")
flow_dict = dict(flow_accumulator)
del flow_accumulator; gc.collect()

print("[3] Building Space-L network...")
G = nx.Graph()
for _, row in station_info.iterrows():
    G.add_node(int(row["stationID"]), name=row["name"], lon=row["lon"], lat=row["lat"])
for _, row in station_info.iterrows():
    u = int(row["stationID"])
    raw = str(row["neighbour"]).replace("[", "").replace("]", "").replace(",", ",")
    for v in raw.split(","):
        v = v.strip()
        if v and v.lower() != "nan":
            v = int(float(v))
            if not G.has_edge(u, v):
                total = flow_dict.get((u, v), 0) + flow_dict.get((v, u), 0)
                G.add_edge(u, v, weight=total if total > 0 else 1,
                           distance=1.0/(total if total > 0 else 1))
isolated = list(nx.isolates(G))
G.remove_nodes_from(isolated)
print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

print("[4] Computing metrics...")
degree_cent = nx.degree_centrality(G)
between_cent = nx.betweenness_centrality(G, weight="distance")
cluster_coef = nx.clustering(G, weight="weight")
node_strength = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
metrics_df = pd.DataFrame({
    "Degree": pd.Series(degree_cent), "Betweenness": pd.Series(between_cent),
    "Clustering": pd.Series(cluster_coef), "Strength": pd.Series(node_strength)
})

indicator_cols = ["Degree", "Betweenness", "Clustering", "Strength"]
X = metrics_df[indicator_cols].values.astype(float)
n, m = X.shape
X_norm = np.zeros_like(X)
for j in range(m):
    col_min, col_max = X[:, j].min(), X[:, j].max()
    X_norm[:, j] = (X[:, j] - col_min) / (col_max - col_min) if col_max - col_min != 0 else 1.0
X_norm += 1e-12
P = X_norm / X_norm.sum(axis=0, keepdims=True)
k_const = 1.0 / np.log(n)
e = -k_const * np.sum(P * np.log(P), axis=0)
d = 1 - e
w = d / d.sum()
weights = {col: float(w[j]) for j, col in enumerate(indicator_cols)}

scaler = MinMaxScaler()
scaled = pd.DataFrame(scaler.fit_transform(metrics_df[indicator_cols]),
                      columns=indicator_cols, index=metrics_df.index)
metrics_df["Importance"] = sum(weights[col] * scaled[col] for col in indicator_cols)
df_importance = metrics_df.sort_values("Importance", ascending=False)
N0 = G.number_of_nodes()

top_name = G.nodes[df_importance.index[0]]["name"]
top_imp = df_importance.iloc[0]["Importance"]
print(f"  Top-1: {top_name} (Imp={top_imp:.4f})")

# ============ Part B: Cascading Failure (multi-target) ============
def cascading_failure_multi(G_original, target_nodes, capacity_factor):
    G = G_original.copy()
    bc_init = nx.betweenness_centrality(G, weight="distance")
    total_flow = sum(d["weight"] for _, _, d in G.edges(data=True))
    capacity = {n: bc_init[n] * total_flow * (1 + capacity_factor) for n in G.nodes()}

    actual = [n for n in target_nodes if n in G]
    G.remove_nodes_from(actual)
    all_removed = list(actual)
    cascade_step = 0

    while cascade_step < 100 and G.number_of_nodes() > 1:
        cascade_step += 1
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
        if G.number_of_nodes() <= 1:
            break

    return len(all_removed) - len(actual), len(all_removed), cascade_step

# ============ Part C: Sensitivity Scan ============
alpha_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
attack_sizes = [1, 2, 3, 5, 7, 10, 15, 20]
na, ns = len(alpha_values), len(attack_sizes)

grid_cascade = np.zeros((na, ns))
grid_total = np.zeros((na, ns))
grid_steps = np.zeros((na, ns))

print(f"\n[5] Scanning {na}*{ns} = {na*ns} scenarios...")
t0 = time.time()
for i, alpha in enumerate(alpha_values):
    for j, ks in enumerate(attack_sizes):
        targets = df_importance.index[:ks].tolist()
        cn, tot, st = cascading_failure_multi(G, targets, alpha)
        grid_cascade[i, j] = cn
        grid_total[i, j] = tot / N0 * 100
        grid_steps[i, j] = st
        ctag = "C+" if cn > 5 else ("C" if cn > 0 else "-")
        print(f"  alpha={alpha:5.2f}  Top-{ks:2d}  |  "
              f"total={int(tot):3d} ({tot/N0*100:5.1f}%)  "
              f"cascade_only={int(cn):3d}  steps={int(st)}  [{ctag}]")

elapsed = time.time() - t0
print(f"\n[OK] Scan completed in {elapsed/60:.1f} minutes")

np.savez("cascade_sensitivity_results.npz",
         alpha_values=alpha_values,
         attack_sizes=attack_sizes,
         grid_cascade=grid_cascade,
         grid_total=grid_total,
         grid_steps=grid_steps)

# Quick summary
print(f"\nKey findings preview:")
for i, alpha in enumerate(alpha_values):
    first_cascade = None
    for j, ks in enumerate(attack_sizes):
        if grid_cascade[i, j] > 0:
            first_cascade = (ks, int(grid_cascade[i, j]), grid_total[i, j])
            break
    if first_cascade:
        print(f"  alpha={alpha:.2f}: cascade starts at Top-{first_cascade[0]} "
              f"({first_cascade[1]} cascade nodes, {first_cascade[2]:.1f}% total)")
    else:
        print(f"  alpha={alpha:.2f}: NO cascade up to Top-{attack_sizes[-1]}")
