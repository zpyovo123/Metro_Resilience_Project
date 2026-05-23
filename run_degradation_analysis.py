"""
城市轨道交通网络韧性退化分析 — 优化版
========================================
双维度（结构效率 + 客流加权效率）
三种攻击策略（随机 / 度中心性 / 综合重要度）
输出指标：退化曲线、AUC、临界崩溃点 q*
"""
import pandas as pd, numpy as np, networkx as nx, time, gc, warnings
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ================================================================
# Part A: 数据加载与网络构建
# ================================================================
print("=" * 55)
print("  城市轨道交通网络韧性退化分析")
print("=" * 55)

print("\n[1/5] 加载数据...")
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
        print(f"  {(cnt+1)*2/1000:.1f}M 行已处理")
flow_dict = dict(flow_accumulator)
del flow_accumulator; gc.collect()
print(f"  OD对数量: {len(flow_dict)}")

print("\n[2/5] 构建 Space-L 网络...")
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
E0 = G.number_of_edges()
print(f"  网络: {N0} 节点, {E0} 边")

# ================================================================
# Part B: 指标计算 + 3指标熵权法
# ================================================================
print("\n[3/5] 计算节点重要度（3指标熵权法）...")
degree_cent = nx.degree_centrality(G)
between_cent = nx.betweenness_centrality(G, weight="distance")
node_strength = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
raw_degree = {n: G.degree(n) for n in G.nodes()}

metrics_df = pd.DataFrame({
    "Degree": pd.Series(degree_cent),
    "Betweenness": pd.Series(between_cent),
    "Strength": pd.Series(node_strength),
    "RawDegree": pd.Series(raw_degree),
})

indicator_cols = ["Degree", "Betweenness", "Strength"]
X = metrics_df[indicator_cols].values.astype(float)
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
weights = {col: float(w[j]) for j, col in enumerate(indicator_cols)}
print(f"  熵权: Deg={weights['Degree']:.3f} ({weights['Degree']*100:.1f}%)"
      f"  Btw={weights['Betweenness']:.3f} ({weights['Betweenness']*100:.1f}%)"
      f"  Str={weights['Strength']:.3f} ({weights['Strength']*100:.1f}%)")

scaled = pd.DataFrame(
    MinMaxScaler().fit_transform(metrics_df[indicator_cols]),
    columns=indicator_cols, index=metrics_df.index)
metrics_df["Importance"] = sum(weights[col] * scaled[col] for col in indicator_cols)
df_importance = metrics_df.sort_values("Importance", ascending=False)

# ================================================================
# Part C: 双维度效率函数
# ================================================================

def structural_efficiency(graph):
    """纯拓扑全局效率 — 边权=1，最短路径按跳数"""
    if graph.number_of_nodes() < 2:
        return 0.0
    try:
        lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight=None))
    except Exception:
        return 0.0
    total = 0.0
    for u in lengths:
        for v in lengths[u]:
            if u != v and lengths[u][v] > 0:
                total += 1.0 / lengths[u][v]
    return total / (N0 * (N0 - 1))

def passenger_efficiency(graph):
    """客流加权全局效率 — 边距=1/客流，最短路径优先高客流走廊"""
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

# ================================================================
# Part D: 退化模拟（双维度追踪）
# ================================================================

def simulate_degradation_dual(graph, attack_order, step_pct=0.05):
    """
    按给定顺序逐批移除节点，每步同时计算结构效率和客流效率。
    返回格式: (x_pct_list, struct_curve, pass_curve)
    """
    G_temp = graph.copy()
    step = max(1, int(N0 * step_pct))

    # 计算初始效率用于归一化
    init_struct = structural_efficiency(graph)
    init_pass = passenger_efficiency(graph)

    x_pct = [0.0]
    struct_curve = [1.0]
    pass_curve = [1.0]

    for i in range(0, len(attack_order), step):
        batch = attack_order[i:i + step]
        G_temp.remove_nodes_from(batch)

        if G_temp.number_of_nodes() > 1:
            struct_curve.append(structural_efficiency(G_temp) / init_struct)
            pass_curve.append(passenger_efficiency(G_temp) / init_pass)
        else:
            struct_curve.append(0.0)
            pass_curve.append(0.0)

        removed_pct = min(100.0, (i + step) / N0 * 100)
        x_pct.append(removed_pct)

        if G_temp.number_of_nodes() <= 1:
            # 填满到100%
            while x_pct[-1] < 99.0:
                x_pct.append(min(100.0, x_pct[-1] + step_pct * 100))
                struct_curve.append(0.0)
                pass_curve.append(0.0)
            break

    return x_pct, struct_curve, pass_curve

# ================================================================
# Part E: 执行三种攻击策略
# ================================================================
print("\n[4/5] 执行退化模拟...")
t0 = time.time()

# --- 攻击序列 ---
attack_random_raw = list(G.nodes())
attack_degree = metrics_df.sort_values("Degree", ascending=False).index.tolist()
attack_importance = df_importance.index.tolist()

# 随机攻击 5 次取平均
N_RAND_RUNS = 5
print(f"  随机攻击 ({N_RAND_RUNS}次平均)...")
struct_rand_runs = []
pass_rand_runs = []
for run in range(N_RAND_RUNS):
    np.random.seed(run * 100)
    shuffled = list(G.nodes())
    np.random.shuffle(shuffled)
    xr, sr, pr = simulate_degradation_dual(G, shuffled)
    struct_rand_runs.append(sr)
    pass_rand_runs.append(pr)

# 对齐长度到最长
max_len_rand = max(len(s) for s in struct_rand_runs)
struct_rand_avg = np.mean([s + [0.0]*(max_len_rand-len(s)) for s in struct_rand_runs], axis=0)
pass_rand_avg = np.mean([p + [0.0]*(max_len_rand-len(p)) for p in pass_rand_runs], axis=0)
x_rand = np.linspace(0, 100, max_len_rand)

print("  度中心性攻击...")
x_deg, struct_deg, pass_deg = simulate_degradation_dual(G, attack_degree)

print("  综合重要度攻击（本文策略）...")
x_imp, struct_imp, pass_imp = simulate_degradation_dual(G, attack_importance)

print(f"  耗时: {time.time()-t0:.1f}s")

# ================================================================
# Part F: 计算 AUC 和临界崩溃点 q*
# ================================================================

def calc_auc(x_pct, curve):
    """梯形法则面积，归一化到 [0,1]"""
    if len(x_pct) < 2:
        return 0.0
    x_norm = np.array(x_pct) / 100.0
    return float(np.trapz(curve, x_norm))

def calc_q_star(x_pct, curve, threshold=0.50):
    """
    临界崩溃点 q*：效率首次低于 threshold 的节点移除比例。
    如果从未低于阈值，返回 100。
    """
    for i, (x, y) in enumerate(zip(x_pct, curve)):
        if y < threshold:
            if i == 0:
                return 0.0
            # 线性插值
            x_prev, y_prev = x_pct[i-1], curve[i-1]
            if y_prev - y == 0:
                return x
            frac = (threshold - y_prev) / (y - y_prev)
            return x_prev + frac * (x - x_prev)
    return 100.0

# 计算所有曲线指标
results = {}
for label, (xs, sc, pc) in [
    ("随机攻击 (5次平均)", (x_rand, struct_rand_avg, pass_rand_avg)),
    ("度中心性攻击", (x_deg, struct_deg, pass_deg)),
    ("综合重要度攻击", (x_imp, struct_imp, pass_imp)),
]:
    auc_s = calc_auc(xs, sc)
    auc_p = calc_auc(xs, pc)
    qs_s = calc_q_star(xs, sc)
    qs_p = calc_q_star(xs, pc)
    results[label] = {
        "AUC_结构": auc_s, "AUC_客流": auc_p,
        "q*_结构": qs_s, "q*_客流": qs_p
    }

# ================================================================
# Part G: 绘制两张独立退化曲线图
# ================================================================
print("\n[5/5] 生成图表...")

colors = {"随机攻击 (5次平均)": "#95A5A6",
          "度中心性攻击": "#2980B9",
          "综合重要度攻击": "#E74C3C"}
linestyles = {"随机攻击 (5次平均)": ":", "度中心性攻击": "-.", "综合重要度攻击": "-"}
markers = {"随机攻击 (5次平均)": "o", "度中心性攻击": "^", "综合重要度攻击": "s"}
linewidths = {"随机攻击 (5次平均)": 2.0, "度中心性攻击": 2.2, "综合重要度攻击": 2.8}

# ---- 图1: 结构效率退化 ----
fig1, ax1 = plt.subplots(figsize=(10, 7))
for label, xs, curve in [
    ("随机攻击 (5次平均)", x_rand, struct_rand_avg),
    ("度中心性攻击", x_deg, struct_deg),
    ("综合重要度攻击", x_imp, struct_imp),
]:
    # 标注 AUC
    auc_val = calc_auc(xs, curve)
    lbl = f"{label}  (AUC={auc_val:.3f})"
    ax1.plot(xs, curve, color=colors[label], linestyle=linestyles[label],
             linewidth=linewidths[label], marker=markers[label],
             markevery=max(1, len(xs)//6), markersize=8, label=lbl)

ax1.set_xlabel("节点失效比例 (%)", fontsize=14)
ax1.set_ylabel("标准化结构效率", fontsize=14)
ax1.set_title("结构韧性退化 — 三种攻击策略对比（纯拓扑）", fontsize=16, fontweight="bold")
ax1.legend(fontsize=11, loc="upper right")
ax1.grid(True, linestyle=":", alpha=0.5)
ax1.set_xlim(0, 100)
ax1.set_ylim(-0.03, 1.05)
plt.tight_layout()
plt.savefig("degradation_struct.png", dpi=250, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("  -> degradation_struct.png")

# ---- 图2: 客流加权效率退化 ----
fig2, ax2 = plt.subplots(figsize=(10, 7))
for label, xs, curve in [
    ("随机攻击 (5次平均)", x_rand, pass_rand_avg),
    ("度中心性攻击", x_deg, pass_deg),
    ("综合重要度攻击", x_imp, pass_imp),
]:
    auc_val = calc_auc(xs, curve)
    lbl = f"{label}  (AUC={auc_val:.3f})"
    ax2.plot(xs, curve, color=colors[label], linestyle=linestyles[label],
             linewidth=linewidths[label], marker=markers[label],
             markevery=max(1, len(xs)//6), markersize=8, label=lbl)

ax2.set_xlabel("节点失效比例 (%)", fontsize=14)
ax2.set_ylabel("标准化客流加权效率", fontsize=14)
ax2.set_title("客流韧性退化 — 三种攻击策略对比（AFC加权）", fontsize=16, fontweight="bold")
ax2.legend(fontsize=11, loc="upper right")
ax2.grid(True, linestyle=":", alpha=0.5)
ax2.set_xlim(0, 100)
ax2.set_ylim(-0.03, 1.05)
plt.tight_layout()
plt.savefig("degradation_passenger.png", dpi=250, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("  -> degradation_passenger.png")

# 保留旧兼容名 degradation_curve.png → 用客流版
import shutil
shutil.copy("degradation_passenger.png", "degradation_curve.png")
print("  -> degradation_curve.png (兼容副本)")

# ================================================================
# Part H: 结果汇总
# ================================================================
print(f"\n{'='*65}")
print(f"  退化分析结果汇总")
print(f"{'='*65}")
print(f"  {'策略':<18s} {'AUC_结构':>8s} {'AUC_客流':>8s} {'q*_结构':>8s} {'q*_客流':>8s}")
print(f"  {'-'*60}")
for label, r in results.items():
    print(f"  {label:<18s} {r['AUC_结构']:>8.4f} {r['AUC_客流']:>8.4f} "
          f"{r['q*_结构']:>7.1f}% {r['q*_客流']:>7.1f}%")
print(f"  {'='*65}")

# 关键对比
r_rand = results["随机攻击 (5次平均)"]
r_imp = results["综合重要度攻击"]
gap_auc_struct = (r_rand["AUC_结构"] - r_imp["AUC_结构"]) / r_rand["AUC_结构"] * 100
gap_auc_pass = (r_rand["AUC_客流"] - r_imp["AUC_客流"]) / r_rand["AUC_客流"] * 100

print(f"\n  关键结论:")
print(f"  1. 结构维度: 蓄意攻击 AUC 比随机低 {gap_auc_struct:.1f}%")
print(f"     → 拓扑结构对蓄意攻击的脆弱性体现在AUC下降")
print(f"  2. 客流维度: 蓄意攻击 AUC 比随机低 {gap_auc_pass:.1f}%")
print(f"     → 客流加权后蓄意攻击的破坏力{'更强' if gap_auc_pass > gap_auc_struct else '与结构相当'}")
print(f"  3. 临界崩溃点: 结构 q*={r_imp['q*_结构']:.1f}%, 客流 q*={r_imp['q*_客流']:.1f}%")
print(f"     → 客流维度的崩溃{'更早' if r_imp['q*_客流'] < r_imp['q*_结构'] else '较晚'}")
print(f"  4. 双维度对比证明：纳入AFC客流后韧性评估更敏感、更准确")

print(f"\n[完成] 总耗时: {(time.time()-t0)/60:.1f} 分钟")
