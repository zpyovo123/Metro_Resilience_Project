"""
级联失效 + 恢复曲线 + 重要度vs破坏度 — 全拆分优化版
所有图表独立输出，不合成多面板
"""
import pandas as pd, numpy as np, networkx as nx, time, gc, warnings
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ================================================================
# Part A: 数据加载、网络构建、3指标熵权法
# ================================================================
print("=" * 55)
print("  级联失效 + 恢复曲线 + 破坏度分析")
print("=" * 55)

print("\n[1] 加载数据...")
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
N0 = G.number_of_nodes()
print(f"  网络: {N0} 节点, {G.number_of_edges()} 边")

print("[3] 3指标熵权法...")
deg = nx.degree_centrality(G)
btw = nx.betweenness_centrality(G, weight="distance")
strength = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
metrics_df = pd.DataFrame({
    "Degree": pd.Series(deg), "Betweenness": pd.Series(btw),
    "Strength": pd.Series(strength),
    "RawDegree": pd.Series({n: G.degree(n) for n in G.nodes()})
})
ind3 = ["Degree", "Betweenness", "Strength"]
X = metrics_df[ind3].values
X_norm = np.zeros_like(X)
for j in range(3):
    cmin, cmax = X[:, j].min(), X[:, j].max()
    X_norm[:, j] = (X[:, j] - cmin) / (cmax - cmin) if cmax - cmin != 0 else 1.0
X_norm += 1e-12
P = X_norm / X_norm.sum(axis=0, keepdims=True)
k = 1.0 / np.log(len(metrics_df))
e = -k * np.sum(P * np.log(P), axis=0)
d = 1 - e; w = d / d.sum()
weights = {col: float(w[j]) for j, col in enumerate(ind3)}
print(f"  权重: Deg={weights['Degree']:.3f} Btw={weights['Betweenness']:.3f} Str={weights['Strength']:.3f}")

scaled = pd.DataFrame(MinMaxScaler().fit_transform(metrics_df[ind3]), columns=ind3, index=metrics_df.index)
metrics_df["Importance"] = sum(weights[col] * scaled[col] for col in ind3)
df_importance = metrics_df.sort_values("Importance", ascending=False)

# ================================================================
# Part B: 级联失效函数
# ================================================================
def cascading_failure(G_original, target_nodes, alpha):
    """多节点同时攻击，返回 (级联波及数, 总移除数, 迭代步数)"""
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
        if G.number_of_nodes() <= 1:
            break

    cascade_only = len(all_removed) - len(actual)
    return cascade_only, len(all_removed), steps

# ================================================================
# Part C: α × k 级联敏感性扫描 → 3 张独立热力图
# ================================================================
print("\n[4] 级联敏感性扫描 (9α × 8k)...")
alpha_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
attack_sizes = [1, 2, 3, 5, 7, 10, 15, 20]
na, ns = len(alpha_values), len(attack_sizes)
grid_total = np.zeros((na, ns))
grid_cascade = np.zeros((na, ns))
grid_steps = np.zeros((na, ns))

t0 = time.time()
for i, alpha in enumerate(alpha_values):
    for j, kk in enumerate(attack_sizes):
        targets = df_importance.index[:kk].tolist()
        cn, tot, st = cascading_failure(G, targets, alpha)
        grid_cascade[i, j] = cn
        grid_total[i, j] = tot / N0 * 100
        grid_steps[i, j] = st
print(f"  耗时 {time.time()-t0:.1f}s")

# 保存供后续使用
np.savez("cascade_sensitivity_results.npz",
         alpha_values=alpha_values, attack_sizes=attack_sizes,
         grid_cascade=grid_cascade, grid_total=grid_total, grid_steps=grid_steps)

xtick_labels = [f"Top-{k}" for k in attack_sizes]
ytick_labels = [f"{a:.2f}" for a in alpha_values]

# ---- 热力图 1: 总失效比例 ----
fig, ax = plt.subplots(figsize=(12, 8))
vmax = max(15, grid_total.max())
im = ax.imshow(grid_total, cmap="RdYlBu_r", aspect="auto", origin="lower", vmin=0, vmax=vmax)
ax.set_xticks(range(ns)); ax.set_yticks(range(na))
ax.set_xticklabels(xtick_labels, fontsize=11)
ax.set_yticklabels(ytick_labels, fontsize=11)
ax.set_xlabel("蓄意攻击规模 (同时移除Top-k站点)", fontsize=13, labelpad=10)
ax.set_ylabel("容量冗余系数  alpha", fontsize=13, labelpad=10)
ax.set_title("级联失效总破坏比例 (%)", fontsize=15, fontweight="bold", pad=15)
for i in range(na):
    for j in range(ns):
        val = grid_total[i, j]
        color = "white" if val > vmax * 0.55 else "black"
        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=10, fontweight="bold", color=color)
cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("失效占全网 %", fontsize=11)
plt.tight_layout()
plt.savefig("cascade_total_pct.png", dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print("  -> cascade_total_pct.png")

# ---- 热力图 2: 纯级联波及数 ----
fig, ax = plt.subplots(figsize=(12, 8))
vmax2 = max(10, grid_cascade.max())
im = ax.imshow(grid_cascade, cmap="YlOrRd", aspect="auto", origin="lower",
               norm=mcolors.LogNorm(vmin=max(1, grid_cascade.min()), vmax=vmax2))
ax.set_xticks(range(ns)); ax.set_yticks(range(na))
ax.set_xticklabels(xtick_labels, fontsize=11)
ax.set_yticklabels(ytick_labels, fontsize=11)
ax.set_xlabel("蓄意攻击规模 (同时移除Top-k站点)", fontsize=13, labelpad=10)
ax.set_ylabel("容量冗余系数  alpha", fontsize=13, labelpad=10)
ax.set_title("纯级联波及节点数 (对数刻度)", fontsize=15, fontweight="bold", pad=15)
for i in range(na):
    for j in range(ns):
        val = grid_cascade[i, j]
        color = "white" if val > vmax2 * 0.4 else "black"
        ax.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=10, fontweight="bold", color=color)
cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("级联波及节点数 (对数)", fontsize=11)
plt.tight_layout()
plt.savefig("cascade_only_nodes.png", dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print("  -> cascade_only_nodes.png")

# ---- 热力图 3: 级联迭代步数 ----
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(grid_steps, cmap="Purples", aspect="auto", origin="lower")
ax.set_xticks(range(ns)); ax.set_yticks(range(na))
ax.set_xticklabels(xtick_labels, fontsize=11)
ax.set_yticklabels(ytick_labels, fontsize=11)
ax.set_xlabel("蓄意攻击规模 (同时移除Top-k站点)", fontsize=13, labelpad=10)
ax.set_ylabel("容量冗余系数  alpha", fontsize=13, labelpad=10)
ax.set_title("级联迭代至稳定所需步数", fontsize=15, fontweight="bold", pad=15)
for i in range(na):
    for j in range(ns):
        val = grid_steps[i, j]
        color = "white" if val > grid_steps.max() * 0.5 else "black"
        ax.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=10, fontweight="bold", color=color)
cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("迭代步数", fontsize=11)
plt.tight_layout()
plt.savefig("cascade_steps.png", dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print("  -> cascade_steps.png")

# ================================================================
# Part D: 综合重要度 vs 级联破坏度 — Top-30 散点图
# ================================================================
print("\n[5] 重要度 vs 破坏度 分析 (Top-30)...")

top30_ids = df_importance.head(30).index.tolist()
imp_vs_dest = []

for nid in top30_ids:
    cn, tot, st = cascading_failure(G, [nid], alpha=0.2)
    imp_score = metrics_df.loc[nid, "Importance"]
    imp_vs_dest.append({
        "station_id": nid,
        "name": G.nodes[nid]["name"],
        "importance": imp_score,
        "cascade_only": cn,
        "total_removed": tot,
        "steps": st,
        "degree": int(metrics_df.loc[nid, "RawDegree"]),
        "betweenness": metrics_df.loc[nid, "Betweenness"],
        "strength": metrics_df.loc[nid, "Strength"],
    })

df_dest = pd.DataFrame(imp_vs_dest)
df_dest = df_dest.sort_values("total_removed", ascending=False)

print(f"  Top-5 破坏度 (单站攻击, α=0.2):")
for _, row in df_dest.head(5).iterrows():
    print(f"    {row['name']:20s} Imp={row['importance']:.4f} "
          f"总移除={int(row['total_removed'])} 级联波及={int(row['cascade_only'])}")

print(f"\n  Top-5 重要度 (熵权法):")
for _, row in df_importance.head(5).iterrows():
    nid = row.name
    name = G.nodes[nid]["name"]
    # find in df_dest
    drow = df_dest[df_dest["station_id"] == nid]
    if len(drow) > 0:
        print(f"    {name:20s} Imp={row['Importance']:.4f} "
              f"总移除={int(drow.iloc[0]['total_removed'])} 级联波及={int(drow.iloc[0]['cascade_only'])}")

# ---- 散点图: 重要度 vs 总移除数 (手动绝对坐标) ----
fig, ax = plt.subplots(figsize=(13, 9))

# 所有 Top-30 浅色背景点
ax.scatter(df_dest["importance"], df_dest["total_removed"],
           c="#B0C4DE", s=120, edgecolors="white", linewidth=0.5, zorder=2)

# 绝对坐标标注表: 站名 -> (xytext_x, xytext_y)
label_positions = {
    "Longde Road":          (0.385, 65),
    "Xujiahui":             (0.445, 64),
    "Jiangsu Road":         (0.415, 57),
    "South Shaanxi Road":   (0.510, 56),
    "Lancun Road":          (0.510, 51),
    "Xintiandi":            (0.660, 62),
    "Jiaotong University":  (0.670, 57),
    "Changshou Road":       (0.680, 51),
    "Jiangning Road":       (0.415, 52),
    "Changning Road":       (0.390, 48),
    "Beiyangjing Road":     (0.355, 52),
    "Deping Road":          (0.415, 48),
    "Century Avenue":       (0.570, 61),
}

for _, row in df_dest.iterrows():
    name = row["name"]
    if name in label_positions:
        ax.scatter(row["importance"], row["total_removed"],
                   c="#E74C3C", s=200, edgecolors="black", linewidth=1.5, zorder=5)
        ax.annotate(name,
                    xy=(row["importance"], row["total_removed"]),
                    xytext=label_positions[name],
                    xycoords="data", textcoords="data",
                    fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#CCCCCC", alpha=0.85))

ax.set_xlabel("综合重要度 (熵权法得分)", fontsize=13)
ax.set_ylabel("级联总破坏站数 (单站攻击, alpha=0.2)", fontsize=13)
ax.set_title("综合重要度 vs 级联破坏度 — Top-30 站点 (alpha=0.2)",
             fontsize=15, fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.4)
plt.tight_layout()
plt.savefig("importance_vs_destructiveness.png", dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print("  -> importance_vs_destructiveness.png")

# ---- 并列对比条形图 ----
fig, ax = plt.subplots(figsize=(14, 8))
top15 = df_dest.head(15)
x = np.arange(len(top15))
width = 0.35

# 重要度排名
imp_ranks = [df_importance.index.tolist().index(sid) + 1 for sid in top15["station_id"]]
# 破坏度排名
dest_ranks = list(range(1, len(top15) + 1))

bars1 = ax.bar(x - width/2, imp_ranks, width, color="#2980B9", edgecolor="white",
               label="综合重要度排名 (越低越重要)")
bars2 = ax.bar(x + width/2, dest_ranks, width, color="#E74C3C", edgecolor="white",
               label="级联破坏度排名 (越低破坏越大)")

# 标注错位
for i, (ir, dr) in enumerate(zip(imp_ranks, dest_ranks)):
    if abs(ir - dr) > 5:
        ax.annotate("", xy=(i + width/2, max(ir, dr)),
                    xytext=(i - width/2, min(ir, dr)),
                    arrowprops=dict(arrowstyle="->", color="#F39C12", lw=1.5))

ax.set_xticks(x)
ax.set_xticklabels(top15["name"], rotation=45, ha="right", fontsize=10)
ax.set_ylabel("排名", fontsize=13)
ax.set_title("Top-15 站点: 综合重要度排名 vs 级联破坏度排名",
             fontsize=15, fontweight="bold")
ax.legend(fontsize=11, loc="upper left")
ax.invert_yaxis()  # 1在最上
ax.grid(axis="y", linestyle=":", alpha=0.4)
plt.tight_layout()
plt.savefig("importance_vs_destructiveness_rank.png", dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print("  -> importance_vs_destructiveness_rank.png")

# ================================================================
# Part E: 恢复曲线 — 拆分独立图
# ================================================================
print("\n[6] 恢复曲线...")

def calc_efficiency(graph):
    """客流加权全局效率"""
    if graph.number_of_nodes() < 2:
        return 0.0
    try:
        lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="distance"))
    except Exception:
        return 0.0
    total = 0.0
    for u in lengths:
        for v in lengths[u]:
            if u != v and lengths[u][v] > 0:
                total += 1.0 / lengths[u][v]
    return total / (N0 * (N0 - 1))

initial_eff = calc_efficiency(G)
print(f"  初始效率: {initial_eff:.4f}")

# 攻击20%站点
attack_ratio = 0.20
num_attack = int(N0 * attack_ratio)

# 攻击序列
intentional_nodes = df_importance.index[:num_attack].tolist()
random_nodes = list(G.nodes())
np.random.seed(42)
np.random.shuffle(random_nodes)
random_nodes = random_nodes[:num_attack]

def simulate_recovery(G_original, failed_nodes, strategy_label, steps=20):
    """恢复模拟：按给定策略排序抢修，返回曲线"""
    G_damaged = G_original.copy()
    G_damaged.remove_nodes_from(failed_nodes)

    if strategy_label == "random":
        order = list(failed_nodes)
        np.random.shuffle(order)
    else:
        valid = [n for n in failed_nodes if n in metrics_df.index]
        order = metrics_df.loc[valid].sort_values(
            by=strategy_label, ascending=False).index.tolist()

    G_cur = G_damaged.copy()
    eff_hist = [calc_efficiency(G_cur) / initial_eff]
    batches = np.array_split(order, steps)
    for batch in batches:
        if len(batch) == 0:
            continue
        for node in batch:
            G_cur.add_node(node, **G_original.nodes[node])
            edges = [(u, v, d) for u, v, d in G_original.edges(node, data=True) if v in G_cur.nodes]
            G_cur.add_edges_from(edges)
        eff_hist.append(calc_efficiency(G_cur) / initial_eff)
    return eff_hist

recovery_results = {}
for att_name, att_nodes in [("蓄意攻击", intentional_nodes), ("随机攻击", random_nodes)]:
    for strat_name, strat_col in [("随机抢修", "random"), ("度中心性抢修", "Degree"), ("重要度抢修", "Importance")]:
        key = f"{att_name}-{strat_name}"
        curve = simulate_recovery(G, att_nodes, strat_col)
        recovery_results[key] = curve
        auc = np.trapz(curve, np.linspace(0, 1, len(curve)))
        print(f"  {key}: AUC={auc:.4f}")

# ---- 恢复图 1: 蓄意攻击后 ----
fig, ax = plt.subplots(figsize=(10, 7))
colors_rec = {"随机抢修": "#95A5A6", "度中心性抢修": "#2980B9", "重要度抢修": "#27AE60"}
styles_rec = {"随机抢修": ":", "度中心性抢修": "-.", "重要度抢修": "-"}
markers_rec = {"随机抢修": "o", "度中心性抢修": "^", "重要度抢修": "s"}

for strat_label in ["随机抢修", "度中心性抢修", "重要度抢修"]:
    key = f"蓄意攻击-{strat_label}"
    curve = recovery_results[key]
    x = np.linspace(0, 100, len(curve))
    auc = np.trapz(curve, np.linspace(0, 1, len(curve)))
    ax.plot(x, curve, color=colors_rec[strat_label], linestyle=styles_rec[strat_label],
            linewidth=2.5, marker=markers_rec[strat_label],
            markevery=max(1, len(curve)//8), markersize=8,
            label=f"{strat_label} (AUC={auc:.3f})")

ax.set_xlabel("已修复节点比例 (%)", fontsize=13)
ax.set_ylabel("标准化网络效率", fontsize=13)
ax.set_title("蓄意攻击后的恢复曲线 — 三种抢修策略对比", fontsize=15, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, linestyle=":", alpha=0.4)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig("recovery_intentional.png", dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print("  -> recovery_intentional.png")

# ---- 恢复图 2: 随机攻击后 ----
fig, ax = plt.subplots(figsize=(10, 7))
for strat_label in ["随机抢修", "度中心性抢修", "重要度抢修"]:
    key = f"随机攻击-{strat_label}"
    curve = recovery_results[key]
    x = np.linspace(0, 100, len(curve))
    auc = np.trapz(curve, np.linspace(0, 1, len(curve)))
    ax.plot(x, curve, color=colors_rec[strat_label], linestyle=styles_rec[strat_label],
            linewidth=2.5, marker=markers_rec[strat_label],
            markevery=max(1, len(curve)//8), markersize=8,
            label=f"{strat_label} (AUC={auc:.3f})")

ax.set_xlabel("已修复节点比例 (%)", fontsize=13)
ax.set_ylabel("标准化网络效率", fontsize=13)
ax.set_title("随机攻击后的恢复曲线 — 三种抢修策略对比", fontsize=15, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, linestyle=":", alpha=0.4)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig("recovery_random.png", dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print("  -> recovery_random.png")

# ================================================================
print(f"\n[完成] 总耗时: {(time.time()-t0)/60:.1f} 分钟")
print("  生成图表:")
print("    cascade_total_pct.png        — 级联总失效比例热力图")
print("    cascade_only_nodes.png        — 纯级联波及节点数热力图")
print("    cascade_steps.png             — 级联迭代步数热力图")
print("    importance_vs_destructiveness.png      — 重要度vs破坏度散点")
print("    importance_vs_destructiveness_rank.png — 重要度vs破坏度排名对比")
print("    recovery_intentional.png      — 蓄意攻击后恢复曲线")
print("    recovery_random.png           — 随机攻击后恢复曲线")
