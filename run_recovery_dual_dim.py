"""
恢复模型交叉验证 + 双维度韧性拆解
修复: 打破攻击-恢复循环论证
新增: 运营供给 vs 出行服务 两维度独立韧性评估
"""
import pandas as pd, numpy as np, networkx as nx, time, gc, warnings
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

N0 = 302  # will be set after network construction

# ============ Part A: Load & Build (same pipeline) ============
print("[1/5] Loading data and building network...")
station_info = pd.read_csv("stationInfo.csv")
station_info.columns = station_info.columns.str.strip()

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
        print(f"  {(chunk_count+1)*2/1000:.1f}M rows", flush=True)
flow_dict = dict(flow_accumulator)
del flow_accumulator; gc.collect()

G = nx.Graph()
for _, row in station_info.iterrows():
    G.add_node(int(row["stationID"]), name=row["name"], lon=row["lon"], lat=row["lat"])
for _, row in station_info.iterrows():
    u = int(row["stationID"])
    raw = str(row["neighbour"]).replace("[", "").replace("]", "").replace("，", ",")
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
N0 = G.number_of_nodes()
print(f"  Network: {N0} nodes, {G.number_of_edges()} edges")

# ============ Part B: Compute metrics & importance ============
print("[2/5] Computing node importance...")
degree_cent = nx.degree_centrality(G)
between_cent = nx.betweenness_centrality(G, weight="distance")
cluster_coef = nx.clustering(G, weight="weight")
node_strength = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
metrics_df = pd.DataFrame({
    "Degree": pd.Series(degree_cent), "Betweenness": pd.Series(between_cent),
    "Clustering": pd.Series(cluster_coef), "Strength": pd.Series(node_strength)
})

# Entropy weight
indicator_cols = ["Degree", "Betweenness", "Clustering", "Strength"]
X_arr = metrics_df[indicator_cols].values.astype(float)
n, m = X_arr.shape
X_norm = np.zeros_like(X_arr)
for j in range(m):
    col_min, col_max = X_arr[:, j].min(), X_arr[:, j].max()
    X_norm[:, j] = (X_arr[:, j] - col_min) / (col_max - col_min) if col_max - col_min != 0 else 1.0
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

# ============ Part C: Weighted Efficiency ============
def calc_weighted_efficiency(graph):
    if graph.number_of_nodes() < 2:
        return 0.0
    try:
        lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="distance"))
    except Exception:
        return 0.0
    eff = 0.0
    for u in lengths:
        for v in lengths[u]:
            if u != v and lengths[u][v] > 0:
                eff += 1.0 / lengths[u][v]
    return eff / (N0 * (N0 - 1))

# ============ Part C (cont): Dual-Dimension Efficiency ============
def calc_structural_efficiency(graph):
    """纯拓扑效率 — 边权=1 (不考虑客流), 所有边等权"""
    if graph.number_of_nodes() < 2:
        return 0.0
    try:
        lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight=None))  # unweighted
    except Exception:
        return 0.0
    eff = 0.0
    for u in lengths:
        for v in lengths[u]:
            if u != v and lengths[u][v] > 0:
                eff += 1.0 / lengths[u][v]
    return eff / (N0 * (N0 - 1))

def calc_passenger_efficiency(graph):
    """客流加权效率 — 边权=distance=1/flow"""
    return calc_weighted_efficiency(graph)  # same as weighted

initial_eff = calc_weighted_efficiency(G)
print(f"  Initial weighted efficiency: {initial_eff:.6f}")

# ============ Part D: Recovery Model — Cross Design ============
print("[3/5] Recovery model — cross attack-recovery design...")

attack_ratio = 0.20
num_attack = int(N0 * attack_ratio)  # 60 nodes

# Attack strategies
nodes_by_importance = df_importance.index.tolist()
nodes_random = list(G.nodes())
np.random.seed(42)
np.random.shuffle(nodes_random)

attack_targets = {
    "Intentional": nodes_by_importance[:num_attack],
    "Random": nodes_random[:num_attack],
}

# Recovery strategies
def get_recovery_order(failed, strategy):
    if strategy == "Random":
        order = list(failed)
        np.random.shuffle(order)
        return order
    else:
        valid = [n for n in failed if n in metrics_df.index]
        return metrics_df.loc[valid].sort_values(by=strategy, ascending=False).index.tolist()

def simulate_recovery_cross(G_original, failed_nodes, recovery_order, steps=20):
    G_cur = G_original.copy()
    G_cur.remove_nodes_from(failed_nodes)
    eff_hist = [calc_weighted_efficiency(G_cur) / initial_eff]
    batches = np.array_split(recovery_order, steps)
    for batch in batches:
        if len(batch) == 0:
            continue
        for node in batch:
            G_cur.add_node(node, **G_original.nodes[node])
            edges_to_add = [(u, v, d) for u, v, d in G_original.edges(node, data=True)
                           if v in G_cur.nodes]
            G_cur.add_edges_from(edges_to_add)
        eff_hist.append(calc_weighted_efficiency(G_cur) / initial_eff)
    return eff_hist

# Run all 6 cross-combinations: 2 attacks x 3 recoveries
recovery_strategies = ["Random", "Degree", "Importance"]
all_recovery_results = {}
t0 = time.time()

for att_name, att_nodes in attack_targets.items():
    for rec_name in recovery_strategies:
        key = f"{att_name}->{rec_name}"
        rec_order = get_recovery_order(att_nodes, rec_name)
        curve = simulate_recovery_cross(G, att_nodes, rec_order)
        all_recovery_results[key] = curve
        print(f"  {key}: start_eff={curve[0]:.3f}, end_eff={curve[-1]:.3f}")

print(f"  Recovery scan done in {time.time()-t0:.1f}s")

# ============ Part E: Dual-Dimension Degradation ============
print("[4/5] Dual-dimension degradation analysis...")

def simulate_dual_degradation(graph, attack_list, step_pct=0.05):
    """同时追踪拓扑效率和客流效率的退化"""
    G_temp = graph.copy()
    structural_curve = [1.0]
    passenger_curve = [1.0]
    step = max(1, int(N0 * step_pct))

    initial_struct = calc_structural_efficiency(graph)
    initial_pass = calc_passenger_efficiency(graph)

    for i in range(0, len(attack_list), step):
        nodes_to_remove = attack_list[i:i+step]
        G_temp.remove_nodes_from(nodes_to_remove)
        if G_temp.number_of_nodes() > 1:
            structural_curve.append(calc_structural_efficiency(G_temp) / initial_struct)
            passenger_curve.append(calc_passenger_efficiency(G_temp) / initial_pass)
        else:
            structural_curve.append(0.0)
            passenger_curve.append(0.0)
            break
    return structural_curve, passenger_curve

# Intentional attack (by importance)
nodes_intentional = df_importance.index.tolist()
struct_int, pass_int = simulate_dual_degradation(G, nodes_intentional)

# Random attack (averaged over 5 runs for stability)
struct_rand_avg = None
pass_rand_avg = None
n_runs = 5
for run in range(n_runs):
    nodes_r = list(G.nodes())
    np.random.seed(run * 100)
    np.random.shuffle(nodes_r)
    sr, pr = simulate_dual_degradation(G, nodes_r)
    if struct_rand_avg is None:
        struct_rand_avg = np.array(sr)
        pass_rand_avg = np.array(pr)
    else:
        # Align to same length
        while len(sr) < len(struct_rand_avg):
            sr.append(0.0)
        while len(struct_rand_avg) < len(sr):
            struct_rand_avg = np.append(struct_rand_avg, 0.0)
        struct_rand_avg = (struct_rand_avg * run + np.array(sr)) / (run + 1)

        while len(pr) < len(pass_rand_avg):
            pr.append(0.0)
        while len(pass_rand_avg) < len(pr):
            pass_rand_avg = np.append(pass_rand_avg, 0.0)
        pass_rand_avg = (pass_rand_avg * run + np.array(pr)) / (run + 1)

# Find values at precisely 20% failure point
idx_20pct = min(len(struct_int) - 1, int(0.20 / 0.05))
print(f"  Intentional at 20%: struct={struct_int[idx_20pct]:.3f}, passenger={pass_int[idx_20pct]:.3f}")
print(f"  Random at 20%:    struct={struct_rand_avg[idx_20pct]:.3f}, passenger={pass_rand_avg[idx_20pct]:.3f}")

# ============ Part F: Generate Poster Figures ============
print("[5/5] Generating poster-quality figures...")

# ---- Figure 1: Recovery Cross-Design (clean) ----
attack_labels_ordered = [("Intentional", "蓄意攻击 (按综合重要度)"),
                         ("Random", "随机攻击")]
colors_rec = {"Random": "#95A5A6", "Degree": "#2980B9", "Importance": "#27AE60"}
linestyles_rec = {"Random": ":", "Degree": "-.", "Importance": "-"}
markers_rec = {"Random": "o", "Degree": "^", "Importance": "s"}
label_map = {"Random": "随机抢修", "Degree": "按度中心性抢修", "Importance": "按综合重要度抢修"}

fig1, axes = plt.subplots(1, 2, figsize=(16, 6))
plt.subplots_adjust(wspace=0.25)

for ax_idx, (att_name, att_label) in enumerate(attack_labels_ordered):
    ax = axes[ax_idx]
    for rec_name in recovery_strategies:
        key = f"{att_name}->{rec_name}"
        curve = all_recovery_results[key]
        x = np.linspace(0, 100, len(curve))
        ax.plot(x, curve, color=colors_rec[rec_name],
                linestyle=linestyles_rec[rec_name], linewidth=2.8,
                marker=markers_rec[rec_name],
                markevery=max(1, len(curve)//10), markersize=9,
                label=label_map[rec_name])
    ax.set_xlabel("已修复节点比例 (%)", fontsize=13)
    ax.set_ylabel("标准化网络效率", fontsize=13)
    ax.set_title(att_label, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylim(0, 1.05)

    # AUC对比框，放在左上角（坐标轴坐标，不挡曲线）
    imp_key = f"{att_name}->Importance"
    deg_key = f"{att_name}->Degree"
    rnd_key = f"{att_name}->Random"
    x_norm = np.linspace(0, 1, len(all_recovery_results[imp_key]))
    auc_rnd = np.trapz(all_recovery_results[rnd_key], x_norm)
    auc_deg = np.trapz(all_recovery_results[deg_key], x_norm)
    auc_imp = np.trapz(all_recovery_results[imp_key], x_norm)
    best_name = "按综合重要度" if auc_imp == max(auc_imp, auc_deg, auc_rnd) else "按度中心性"
    ax.text(0.03, 0.97,
            f"AUC (越大恢复越快):\n"
            f"  随机抢修: {auc_rnd:.3f}\n"
            f"  按度抢修: {auc_deg:.3f}\n"
            f"  按重要度抢修: {auc_imp:.3f}\n"
            f"  最优策略: {best_name}",
            transform=ax.transAxes, fontsize=9.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor="#CC6600", alpha=0.85))

fig1.suptitle("抢修策略交叉对比 — 不同攻击模式 × 不同恢复策略",
              fontsize=15, fontweight="bold", y=1.01)
plt.savefig("recovery_cross_design.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("  -> recovery_cross_design.png")

# ---- Figure 2: Dual-Dimension Degradation (clean) ----
fig2, axes = plt.subplots(1, 2, figsize=(16, 6.5))
plt.subplots_adjust(wspace=0.25)
idx_20 = min(len(struct_int) - 1, int(0.20 / 0.05))

# -- Left: Intentional attack --
ax = axes[0]
x_int = np.linspace(0, 100, len(struct_int))
ax.plot(x_int, struct_int, "o-", color="#2980B9", linewidth=2.8, markersize=6,
        markevery=3, label="结构韧性 (纯拓扑)")
ax.plot(x_int, pass_int, "s-", color="#E74C3C", linewidth=2.8, markersize=6,
        markevery=3, label="客流韧性 (AFC加权)")
ax.fill_between(x_int, struct_int, pass_int, alpha=0.12, color="gray",
                label="两维度差异")
ax.set_xlabel("节点失效比例 (%)", fontsize=13)
ax.set_ylabel("标准化效率", fontsize=13)
ax.set_title("蓄意攻击下的双维度韧性退化", fontsize=14, fontweight="bold")
ax.legend(fontsize=10.5, loc="lower left")
ax.grid(True, linestyle=":", alpha=0.5)
ax.set_ylim(0, 1.05)

gap_val = abs(pass_int[idx_20] - struct_int[idx_20]) * 100
which_fragile = "客流维度" if pass_int[idx_20] < struct_int[idx_20] else "结构维度"
ax.text(0.97, 0.97,
        f"20%节点失效时:\n"
        f"  结构效率: {struct_int[idx_20]:.3f}\n"
        f"  客流效率: {pass_int[idx_20]:.3f}\n"
        f"  → {which_fragile}更脆弱 ({gap_val:.1f}%)",
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF9E7",
                  edgecolor="#F39C12", alpha=0.9))

# -- Right: Random attack --
ax = axes[1]
x_rand = np.linspace(0, 100, len(struct_rand_avg))
ax.plot(x_rand, struct_rand_avg, "o-", color="#2980B9", linewidth=2.8, markersize=6,
        markevery=3, label="结构韧性 (纯拓扑)")
ax.plot(x_rand, pass_rand_avg, "s-", color="#E74C3C", linewidth=2.8, markersize=6,
        markevery=3, label="客流韧性 (AFC加权)")
ax.fill_between(x_rand, struct_rand_avg, pass_rand_avg, alpha=0.12, color="gray",
                label="两维度差异")
ax.set_xlabel("节点失效比例 (%)", fontsize=13)
ax.set_ylabel("标准化效率", fontsize=13)
ax.set_title("随机攻击下的双维度韧性退化 (5次平均)", fontsize=14, fontweight="bold")
ax.legend(fontsize=10.5, loc="lower left")
ax.grid(True, linestyle=":", alpha=0.5)
ax.set_ylim(0, 1.05)

gap_val = abs(pass_rand_avg[idx_20] - struct_rand_avg[idx_20]) * 100
which_fragile = "客流维度" if pass_rand_avg[idx_20] < struct_rand_avg[idx_20] else "结构维度"
ax.text(0.97, 0.97,
        f"20%节点失效时:\n"
        f"  结构效率: {struct_rand_avg[idx_20]:.3f}\n"
        f"  客流效率: {pass_rand_avg[idx_20]:.3f}\n"
        f"  → {which_fragile}更脆弱 ({gap_val:.1f}%)",
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF9E7",
                  edgecolor="#F39C12", alpha=0.9))

fig2.suptitle("网络韧性双维度拆解: 结构供给能力 vs 客流服务保障",
              fontsize=15, fontweight="bold", y=1.01)
plt.savefig("dual_dimension_degradation.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("  -> dual_dimension_degradation.png")

# ---- 图3: 因子载荷热力图 (全中文) ----
fig3, ax = plt.subplots(figsize=(9, 7))
from factor_analyzer import FactorAnalyzer
fa = FactorAnalyzer(n_factors=2, rotation="varimax")
col_scaled = pd.DataFrame(
    MinMaxScaler().fit_transform(metrics_df[indicator_cols]),
    columns=indicator_cols, index=metrics_df.index
)
fa.fit(col_scaled)
loadings = pd.DataFrame(
    fa.loadings_,
    index=["度中心性", "介数中心性", "聚类系数", "客流强度"],
    columns=["因子1\n(结构供给)", "因子2\n(客流服务)"])

im = ax.imshow(loadings.values, cmap="RdYlBu", aspect="auto", vmin=-1, vmax=1)
ax.set_xticks([0, 1])
ax.set_yticks(range(4))
ax.set_xticklabels(loadings.columns, fontsize=11)
ax.set_yticklabels(loadings.index, fontsize=11)

for i in range(4):
    for j in range(2):
        val = loadings.values[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)

ax.set_title("因子分析: 站点重要度的两个潜在维度",
             fontsize=14, fontweight="bold", pad=15)
cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("因子载荷", fontsize=11)

plt.tight_layout()
plt.savefig("factor_loading_poster.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("  -> factor_loading_poster.png")

# ============ Print Summary ============
print(f"\n{'='*60}")
print(f"  RECOVERY CROSS-DESIGN FINDINGS")
print(f"{'='*60}")
for att_name, _att_label in attack_labels_ordered:
    imp_key = f"{att_name}->Importance"
    deg_key = f"{att_name}->Degree"
    rnd_key = f"{att_name}->Random"
    imp_curve = all_recovery_results[imp_key]
    deg_curve = all_recovery_results[deg_key]
    rnd_curve = all_recovery_results[rnd_key]
    mid = len(imp_curve) // 2
    print(f"  {att_name} attack:")
    print(f"    After 50% repaired: Random={rnd_curve[mid]:.3f}, "
          f"Degree={deg_curve[mid]:.3f}, "
          f"Importance={imp_curve[mid]:.3f}")
    print(f"    Importance advantage at midpoint: "
          f"+{(imp_curve[mid] - rnd_curve[mid])*100:.1f}%")

    # AUC (trapezoidal rule, higher = faster recovery)
    x_norm = np.linspace(0, 1, len(imp_curve))
    for label, curve in [("Random", rnd_curve), ("Degree", deg_curve), ("Importance", imp_curve)]:
        auc = np.trapz(curve, x_norm)
        print(f"    AUC ({label}): {auc:.3f}")

print(f"\n{'='*60}")
print(f"  DUAL-DIMENSION FINDINGS (at 20% failure)")
print(f"{'='*60}")
idx_20 = min(len(struct_int) - 1, int(0.20 / 0.05))
print(f"  Intentional attack:")
print(f"    Structural eff:  {struct_int[idx_20]:.4f}")
print(f"    Passenger eff:   {pass_int[idx_20]:.4f}")
print(f"    Gap: {abs(pass_int[idx_20] - struct_int[idx_20])*100:.1f}%")
print(f"    -> {'Passenger efficiency MORE FRAGILE' if pass_int[idx_20] < struct_int[idx_20] else 'Structural MORE FRAGILE'}")
print(f"  Random attack (5-run avg):")
print(f"    Structural eff:  {struct_rand_avg[idx_20]:.4f}")
print(f"    Passenger eff:   {pass_rand_avg[idx_20]:.4f}")
print(f"    Gap: {abs(pass_rand_avg[idx_20] - struct_rand_avg[idx_20])*100:.1f}%")
print(f"    -> {'Passenger efficiency MORE FRAGILE' if pass_rand_avg[idx_20] < struct_rand_avg[idx_20] else 'Structural MORE FRAGILE'}")

print(f"\n[OK] All analyses complete")
