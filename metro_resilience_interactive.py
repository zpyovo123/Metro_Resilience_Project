# -*- coding: utf-8 -*-
"""
城市轨道交通网络韧性分析 — 交互式版本 (VS Code Python Interactive Window)
======================================================================
使用方法:
  1. 在 VS Code 中打开此文件
  2. 安装 Python 扩展 (ms-python.python)
  3. 右下角选择解释器: D:\miniconda\envs\network_env\python.exe
  4. 每个 # %% 标记一个单元格, 点击 "Run Cell" 或 Shift+Enter 逐格运行
  5. 变量在单元格之间共享, 可任意修改后重复运行某个格

数据: metroData_ODFlow.csv (11GB) + stationInfo.csv (302站)
"""

# %% [markdown]
# # 城市轨道交通网络韧性分析
# ## 融合复杂网络理论与动态客流数据的级联失效模型
#
# 按 Shift+Enter 逐格运行，可随时修改代码后重新运行任意格。

# %% Cell 1: 依赖库导入
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import gc
import warnings
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from factor_analyzer import FactorAnalyzer
from adjustText import adjust_text

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print('[OK] 依赖库导入成功')
print(f'NetworkX 版本: {nx.__version__}')

# %% Cell 2: OD 客流数据高效加载 (约 2 分钟)
def _find_column(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f'无法匹配列名, 现有: {cols}, 候选: {candidates}')

od_file_path = 'metroData_ODFlow.csv'
bad_dates = [20170504, 20170508, 20170509, 20170616, 20170627, 20170628]
chunksize = 2000000

first_chunk = pd.read_csv(od_file_path, nrows=5)
first_chunk.columns = first_chunk.columns.str.strip()
cols = first_chunk.columns.tolist()
print(f'真实列名: {cols}')

date_col = _find_column(cols, ['date', 'Date', 'DATE'])
o_col    = _find_column(cols, ['originStation', 'O-Station'])
d_col    = _find_column(cols, ['destinationStation', 'D-Station'])
f_col    = _find_column(cols, ['Flow', 'flow', 'FLOW'])
print(f'匹配: date={date_col}, O={o_col}, D={d_col}, Flow={f_col}')

flow_accumulator = defaultdict(int)
chunk_count = 0
total_rows = 0
t0 = time.time()

for chunk in pd.read_csv(od_file_path, chunksize=chunksize):
    chunk.columns = chunk.columns.str.strip()
    chunk_count += 1
    total_rows += len(chunk)
    chunk = chunk[~chunk[date_col].isin(bad_dates)]
    grouped = chunk.groupby([o_col, d_col])[f_col].sum()
    for (o, d), flow_val in grouped.items():
        flow_accumulator[(int(o), int(d))] += flow_val
    if chunk_count % 10 == 0:
        print(f'  已处理 {total_rows/1e6:.0f}M 行, {len(flow_accumulator):,} 对 OD')

flow_dict = dict(flow_accumulator)
del flow_accumulator, chunk, grouped; gc.collect()

elapsed = time.time() - t0
print(f'[OK] OD 数据加载完成: {len(flow_dict):,} 对 OD, 耗时 {elapsed/60:.1f} 分钟')

# %% Cell 3: 客流加权 Space-L 网络构建
station_info = pd.read_csv('stationInfo.csv')
station_info.columns = station_info.columns.str.strip()

id_col, name_col, lon_col, lat_col, neighbor_col = (
    'stationID', 'name', 'lon', 'lat', 'neighbour'
)

G = nx.Graph()
for _, row in station_info.iterrows():
    G.add_node(int(row[id_col]), name=row[name_col],
               lon=row[lon_col], lat=row[lat_col])

for _, row in station_info.iterrows():
    u = int(row[id_col])
    raw = str(row[neighbor_col]).replace('[', '').replace(']', '').replace('，', ',')
    for v in raw.split(','):
        v = v.strip()
        if v and v.lower() != 'nan':
            v = int(float(v))
            if not G.has_edge(u, v):
                flow_uv = flow_dict.get((u, v), 0)
                flow_vu = flow_dict.get((v, u), 0)
                total = flow_uv + flow_vu
                weight = total if total > 0 else 1
                G.add_edge(u, v, weight=weight, distance=1.0/weight)

isolated = list(nx.isolates(G))
G.remove_nodes_from(isolated)
if isolated:
    print(f'  剔除 {len(isolated)} 个孤立节点: {isolated}')

print(f'[OK] 网络构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边')

# %% Cell 4: 网络特征指标计算
degree_cent  = nx.degree_centrality(G)
between_cent = nx.betweenness_centrality(G, weight='distance')
cluster_coef = nx.clustering(G, weight='weight')
node_strength = {
    node: sum(data['weight'] for _, _, data in G.edges(node, data=True))
    for node in G.nodes()
}

metrics_df = pd.DataFrame({
    'Degree':       pd.Series(degree_cent),
    'Betweenness':  pd.Series(between_cent),
    'Clustering':   pd.Series(cluster_coef),
    'Strength':     pd.Series(node_strength)
})

print(f'[OK] 指标计算完成: {len(metrics_df)} 个节点')
print(metrics_df.describe().round(4))

# %% Cell 5: 熵权法客观赋权 + 综合重要度计算
def entropy_weight_method(df, indicator_cols):
    """修正版熵权法: 直接对原始指标赋权, 而非对正交因子赋权"""
    X = df[indicator_cols].values.astype(float)
    n, m = X.shape
    X_norm = np.zeros_like(X)
    for j in range(m):
        col_min, col_max = X[:, j].min(), X[:, j].max()
        if col_max - col_min == 0:
            X_norm[:, j] = 1.0
        else:
            X_norm[:, j] = (X[:, j] - col_min) / (col_max - col_min)
    X_norm = X_norm + 1e-12
    P = X_norm / X_norm.sum(axis=0, keepdims=True)
    k = 1.0 / np.log(n)
    e = -k * np.sum(P * np.log(P), axis=0)
    d = 1 - e
    w = d / d.sum()
    return {col: float(w[j]) for j, col in enumerate(indicator_cols)}

indicator_cols = ['Degree', 'Betweenness', 'Clustering', 'Strength']
ewm_weights = entropy_weight_method(metrics_df, indicator_cols)

print('---------- 修正后的熵权法权重 ----------')
for k, v in ewm_weights.items():
    print(f'  {k:15s}: {v:.4f}')

scaler = MinMaxScaler()
scaled = pd.DataFrame(
    scaler.fit_transform(metrics_df[indicator_cols]),
    columns=indicator_cols, index=metrics_df.index
)

metrics_df['Importance'] = sum(ewm_weights[col] * scaled[col] for col in indicator_cols)
df_importance = metrics_df.sort_values('Importance', ascending=False)

print('\n---------- Top-10 关键站点 (综合重要度) ----------')
top10 = df_importance.head(10)
for i, (idx, row) in enumerate(top10.iterrows()):
    name = G.nodes[idx].get('name', str(idx))
    print(f'  {i+1:2d}. {name:15s} (ID={idx:4d})  '
          f'Imp={row["Importance"]:.4f}  '
          f'Deg={row["Degree"]:.3f}  Btw={row["Betweenness"]:.3f}  '
          f'Str={row["Strength"]:.0f}')

# %% Cell 6: 级联失效模型 (Cascading Failure)
def cascading_failure(G_original, df_importance, capacity_factor=0.2, verbose=True):
    """
    级联失效模拟:
      L_i = 加权介数 * 总客流,  C_i = L_i * (1+alpha)
      攻击 Top-1 → 最短路径重分配 → 超载移除 → 迭代
    """
    G = G_original.copy()
    n_original = G.number_of_nodes()

    bc_init = nx.betweenness_centrality(G, weight='distance')
    total_flow = sum(data['weight'] for _, _, data in G.edges(data=True))
    initial_load = {n: bc_init[n] * total_flow for n in G.nodes()}
    capacity = {n: L * (1 + capacity_factor) for n, L in initial_load.items()}

    top_node = df_importance.index[0]
    top_name = G.nodes[top_node].get('name', str(top_node))
    top_score = df_importance.iloc[0]['Importance']

    if verbose:
        print(f'\n{"="*55}')
        print(f'  级联失效模拟开始')
        print(f'{"="*55}')
        print(f'  初始攻击目标: {top_name} (ID={top_node})')
        print(f'  综合重要度:   {top_score:.4f}')
        print(f'  初始负荷:     {initial_load[top_node]:.2f}')

    G.remove_node(top_node)
    removal_sequence = [top_node]
    overload_events = []
    cascade_step = 0
    max_steps = 100

    while cascade_step < max_steps and G.number_of_nodes() > 1:
        cascade_step += 1
        current_nodes = set(G.nodes())
        try:
            bc_new = nx.betweenness_centrality(G, weight='distance')
        except Exception:
            if verbose:
                print(f'  [步 {cascade_step}] 网络破碎，终止')
            break

        current_load = {n: bc_new[n] * total_flow for n in current_nodes}
        overloaded = [(n, current_load[n], capacity[n])
                      for n in current_nodes if current_load[n] > capacity[n]]

        if not overloaded:
            if verbose:
                print(f'  [步 {cascade_step}] [OK] 无超载，网络已稳定')
            break

        overloaded.sort(key=lambda x: x[1], reverse=True)
        overload_events.append({
            'step': cascade_step, 'n_overloaded': len(overloaded),
            'nodes': [x[0] for x in overloaded]
        })

        if verbose:
            print(f'  [步 {cascade_step}] 超载 {len(overloaded)} 个节点:')
            for nid, ld, cap in overloaded[:5]:
                nm = G_original.nodes[nid].get('name', str(nid))
                print(f'    {nm}: load={ld:.1f} > cap={cap:.1f} (超载{(ld/cap-1)*100:.0f}%)')
            if len(overloaded) > 5:
                print(f'    ... 其余 {len(overloaded)-5} 个')

        for n, _, _ in overloaded:
            removal_sequence.append(n)
            G.remove_node(n)

        if G.number_of_nodes() <= 1:
            if verbose:
                print(f'  [步 {cascade_step}] 网络基本崩溃 ({G.number_of_nodes()} 节点剩余)')
            break

    result = {
        'initial_target': top_node,
        'initial_target_name': top_name,
        'removal_sequence': removal_sequence,
        'total_removed': len(removal_sequence),
        'cascade_steps': cascade_step,
        'overload_events': overload_events,
        'final_graph': G,
        'n_surviving': G.number_of_nodes(),
        'n_original': n_original,
    }

    if verbose:
        print(f'\n  -------- 级联失效汇总 --------')
        print(f'  初始攻击:     {top_name}')
        print(f'  级联步数:     {cascade_step}')
        print(f'  总移除节点:   {len(removal_sequence)}/{n_original} '
              f'({len(removal_sequence)/n_original*100:.1f}%)')
        print(f'  其中级联波及: {len(removal_sequence)-1}')
        print(f'  幸存节点:     {G.number_of_nodes()}')

    return result

# %% Cell 7: 执行级联失效模拟 + 对比分析
N0 = G.number_of_nodes()

cascade_result = cascading_failure(G, df_importance, capacity_factor=0.2, verbose=True)

G_cascade = cascade_result['final_graph']
G_no_cascade = G.copy()
G_no_cascade.remove_node(cascade_result['initial_target'])

print(f'\n{"="*60}')
print(f'  级联失效 vs 仅移除Top-1 对比')
print(f'{"="*60}')
print(f'  {"指标":<20} {"仅移除Top-1":>12} {"Top-1+级联":>12}')

n_no = G_no_cascade.number_of_nodes()
n_cas = G_cascade.number_of_nodes()
print(f'  {"幸存节点数":<20} {n_no:>12} {n_cas:>12}')
print(f'  {"幸存比例":<20} {n_no/N0:>11.2%} {n_cas/N0:>11.2%}')

eff_no = nx.global_efficiency(G_no_cascade) if n_no > 1 else 0
eff_cas = nx.global_efficiency(G_cascade) if n_cas > 1 else 0
print(f'  {"全局效率":<20} {eff_no:>12.6f} {eff_cas:>12.6f}')

lcc_no = len(max(nx.connected_components(G_no_cascade), key=len)) if n_no > 1 else 0
lcc_cas = len(max(nx.connected_components(G_cascade), key=len)) if n_cas > 1 else 0
print(f'  {"最大连通分量":<20} {lcc_no:>12} {lcc_cas:>12}')
print(f'  {"LCC比例":<20} {lcc_no/N0:>11.2%} {lcc_cas/N0:>11.2%}')

# %% Cell 8: 网络韧性退化模拟 (3 种攻击策略)
def calc_weighted_efficiency(graph):
    """计算客流加权网络效率"""
    if graph.number_of_nodes() < 2:
        return 0.0
    try:
        lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight='distance'))
    except Exception:
        return 0.0
    eff = 0.0
    for u in lengths:
        for v in lengths[u]:
            if u != v and lengths[u][v] > 0:
                eff += 1.0 / lengths[u][v]
    return eff / (N0 * (N0 - 1))

def simulate_attacks(graph, attack_list, initial_eff, step_pct=0.05):
    G_temp = graph.copy()
    eff_curve = [1.0]
    step = max(1, int(N0 * step_pct))
    for i in range(0, len(attack_list), step):
        nodes_to_remove = attack_list[i:i+step]
        G_temp.remove_nodes_from(nodes_to_remove)
        if G_temp.number_of_nodes() > 1:
            eff_curve.append(calc_weighted_efficiency(G_temp) / initial_eff)
        else:
            eff_curve.append(0.0)
            break
    return eff_curve

initial_eff = calc_weighted_efficiency(G)
print(f'初始客流加权网络效率: {initial_eff:.6f}')

nodes_random = list(G.nodes())
np.random.shuffle(nodes_random)
nodes_static = metrics_df.sort_values('Degree', ascending=False).index.tolist()
nodes_intentional = df_importance.index.tolist()

print('正在模拟三种攻击策略 (随机 / 度中心性 / 综合重要度)...')
t_attack = time.time()
eff_random = simulate_attacks(G, nodes_random, initial_eff)
eff_static = simulate_attacks(G, nodes_static, initial_eff)
eff_intentional = simulate_attacks(G, nodes_intentional, initial_eff)
print(f'  耗时 {time.time()-t_attack:.1f}s')

# 对齐长度
max_len = max(len(eff_random), len(eff_static), len(eff_intentional))
for curve in [eff_random, eff_static, eff_intentional]:
    curve.extend([0.0] * (max_len - len(curve)))
x_axis = np.linspace(0, 100, max_len)

plt.figure(figsize=(10, 6), dpi=200)
plt.plot(x_axis, eff_random, label='随机攻击', color='#95A5A6', marker='o', markevery=3)
plt.plot(x_axis, eff_static, label='传统策略 (按度中心性)', color='#2980B9',
         linestyle='-.', linewidth=2, marker='^', markevery=3)
plt.plot(x_axis, eff_intentional, label='本文策略 (按综合重要度)', color='#E74C3C',
         linewidth=2.5, marker='s', markevery=3)
plt.xlabel('节点失效比例 (%)', fontsize=14)
plt.ylabel('网络性能保持率 (标准化)', fontsize=14)
plt.title('城市轨道交通网络韧性退化曲线', fontsize=16, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('degradation_curve.png', dpi=200)
plt.show()
print('[OK] 退化曲线已保存为 degradation_curve.png')

# %% Cell 9: 网络韧性恢复模拟 (3 种抢修策略)
def simulate_recovery(G_original, G_damaged, failed_nodes, strategy, initial_eff, steps=20):
    if strategy == 'random':
        recovery_seq = list(failed_nodes)
        np.random.shuffle(recovery_seq)
    else:
        valid = [n for n in failed_nodes if n in metrics_df.index]
        recovery_seq = metrics_df.loc[valid].sort_values(
            by=strategy, ascending=False).index.tolist()

    G_cur = G_damaged.copy()
    eff_hist = [calc_weighted_efficiency(G_cur) / initial_eff]
    batches = np.array_split(recovery_seq, steps)

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

attack_ratio = 0.20
num_attack = int(N0 * attack_ratio)
failed_nodes = df_importance.index[:num_attack].tolist()

G_damaged = G.copy()
G_damaged.remove_nodes_from(failed_nodes)
print(f'瘫痪 {num_attack} 个节点 ({attack_ratio*100:.0f}%), 剩余 {G_damaged.number_of_nodes()} 个')

print('正在模拟三种抢修策略 (随机 / 度中心性 / 综合重要度)...')
t_rec = time.time()
eff_rec_random  = simulate_recovery(G, G_damaged, failed_nodes, 'random', initial_eff)
eff_rec_static  = simulate_recovery(G, G_damaged, failed_nodes, 'Degree', initial_eff)
eff_rec_proposed = simulate_recovery(G, G_damaged, failed_nodes, 'Importance', initial_eff)
print(f'  耗时 {time.time()-t_rec:.1f}s')

x_rec = np.linspace(0, 100, len(eff_rec_proposed))

plt.figure(figsize=(10, 6), dpi=200)
plt.plot(x_rec, eff_rec_random, label='盲目随机抢修', color='#95A5A6', marker='o', markevery=3)
plt.plot(x_rec, eff_rec_static, label='按度中心性抢修 (传统)', color='#2980B9',
         linestyle='-.', linewidth=2, marker='^', markevery=3)
plt.plot(x_rec, eff_rec_proposed, label='按综合重要度抢修 (本文)', color='#27AE60',
         linewidth=2.5, marker='s', markevery=3)
plt.xlabel('已修复节点占比 (%)', fontsize=14)
plt.ylabel('网络性能保持率 (标准化)', fontsize=14)
plt.title('网络韧性恢复过程 (20%节点失效)', fontsize=16, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('recovery_curve.png', dpi=200)
plt.show()
print('[OK] 恢复曲线已保存为 recovery_curve.png')

# %% Cell 10: 级联失效过程可视化
events = cascade_result['overload_events']

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

if events:
    steps = [e['step'] for e in events]
    n_over = [e['n_overloaded'] for e in events]
    axes[0].bar(steps, n_over, color='tomato', edgecolor='black')
    axes[0].set_xlabel('级联步数', fontsize=12)
    axes[0].set_ylabel('超载节点数', fontsize=12)
    axes[0].set_title('各步新增超载节点', fontsize=14)
    axes[0].set_xticks(steps)
    axes[0].grid(axis='y', alpha=0.3)

    cumulative = [1] + list(np.cumsum(n_over))
    axes[1].plot(range(len(cumulative)), cumulative, 'o-',
                 color='darkred', linewidth=2, markersize=8)
    axes[1].set_xlabel('迭代步数 (0=初始攻击)', fontsize=12)
    axes[1].set_ylabel('累积移除节点数', fontsize=12)
    axes[1].set_title('累积失效节点曲线', fontsize=14)
    axes[1].grid(alpha=0.3)
else:
    axes[0].text(0.5, 0.5, '无级联事件发生\n网络韧性良好', ha='center', va='center',
                 fontsize=16, transform=axes[0].transAxes)
    axes[0].set_title('级联失效: 无事件', fontsize=14)
    axes[1].text(0.5, 0.5, f'仅移除初始攻击节点\n幸存 {cascade_result["n_surviving"]}/{cascade_result["n_original"]} 站',
                 ha='center', va='center', fontsize=16, transform=axes[1].transAxes)
    axes[1].set_title('网络状态: 稳定', fontsize=14)

plt.tight_layout()
plt.savefig('cascade_process.png', dpi=150)
plt.show()

# %% Cell 11: 综合可视化 (因子载荷 + 散点图 + 地理拓扑图)
fa = FactorAnalyzer(n_factors=2, rotation='varimax')
col_scaled = pd.DataFrame(
    MinMaxScaler().fit_transform(metrics_df[indicator_cols]),
    columns=indicator_cols, index=metrics_df.index
)
fa.fit(col_scaled)
loadings = pd.DataFrame(fa.loadings_, index=indicator_cols, columns=['Factor_1', 'Factor_2'])

fig = plt.figure(figsize=(18, 14))

# -- 子图 1: 因子载荷热力图 --
ax1 = fig.add_subplot(2, 2, 1)
im = ax1.imshow(loadings.values, cmap='Blues', aspect='auto')
y_labels = ['度中心性', '介数中心性', '聚类系数', '客流强度']
x_labels = ['因子1\n(方差贡献)', '因子2\n(方差贡献)']
ax1.set_xticks([0, 1]); ax1.set_yticks(range(4))
ax1.set_xticklabels(x_labels, fontsize=10)
ax1.set_yticklabels(y_labels, fontsize=10)
for i in range(4):
    for j in range(2):
        color = 'white' if loadings.values[i, j] > 0.5 else 'black'
        ax1.text(j, i, f'{loadings.values[i, j]:.3f}', ha='center', va='center',
                 color=color, fontweight='bold')
ax1.set_title('因子载荷热力图', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax1, shrink=0.8)

# -- 子图 2: Degree vs Strength 散点图 --
ax2 = fig.add_subplot(2, 2, 2)
ax2.scatter(metrics_df['Degree'], metrics_df['Strength'],
           c='#B0C4DE', alpha=0.6, edgecolors='white', s=50, label='普通站点')
top5 = df_importance.head(5)
ax2.scatter(top5['Degree'], top5['Strength'],
           c='#E74C3C', marker='*', s=300, edgecolors='black', label='Top-5 关键枢纽')
texts_s = []
for idx in top5.index:
    name = G.nodes[idx].get('name', str(idx))
    texts_s.append(ax2.text(top5.loc[idx, 'Degree'], top5.loc[idx, 'Strength'],
                            name, fontsize=9, fontweight='bold'))
adjust_text(texts_s, arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
ax2.set_xlabel('度中心性', fontsize=11)
ax2.set_ylabel('节点客流强度', fontsize=11)
ax2.set_title('结构连通度 vs 实际客流量', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, linestyle='--', alpha=0.5)

# -- 子图 3: 地理拓扑图 --
ax3 = fig.add_subplot(2, 2, (3, 4))
pos = {node: (data['lon'], data['lat']) for node, data in G.nodes(data=True)}
top5_ids = top5.index.tolist()
normal_ids = list(set(G.nodes()) - set(top5_ids))
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='#A0A0A0', width=1.0, ax=ax3)
nx.draw_networkx_nodes(G, pos, nodelist=normal_ids, node_size=20,
                       node_color='#4A90E2', alpha=0.6, ax=ax3)
nx.draw_networkx_nodes(G, pos, nodelist=top5_ids, node_size=200,
                       node_color='#E74C3C', edgecolors='black', linewidths=1.5, ax=ax3)
texts_t = []
for idx in top5_ids:
    x, y = pos[idx]
    texts_t.append(ax3.text(x, y, G.nodes[idx]['name'], fontsize=10, fontweight='bold'))
adjust_text(texts_t, expand_points=(1.5, 1.5),
            arrowprops=dict(arrowstyle='-', color='black', lw=1.2, alpha=0.7))
ax3.set_title('关键枢纽地理分布', fontsize=14, fontweight='bold')
ax3.axis('off')

plt.tight_layout()
plt.savefig('comprehensive_visualization.png', dpi=200, bbox_inches='tight')
plt.show()
print('[OK] 综合可视化已保存为 comprehensive_visualization.png')

# %% Cell 12: 关键保护建议 (总结)
print('\n' + '='*60)
print('  分析完毕! 关键保护建议')
print('='*60)
top5_ids = df_importance.head(5).index.tolist()
top5_names = [G.nodes[i].get('name', str(i)) for i in top5_ids]
print(f'  重点保护 Top-5 站点: {top5_names}')
print(f'  级联波及额外站点数: {cascade_result["total_removed"] - 1}')
final_eff = nx.global_efficiency(cascade_result['final_graph']) if cascade_result['n_surviving'] > 1 else 0
print(f'  遭受攻击后网络全局效率: {final_eff:.6f}')
print(f'\n  生成图表:')
print(f'    - degradation_curve.png           (退化曲线)')
print(f'    - recovery_curve.png              (恢复曲线)')
print(f'    - cascade_process.png             (级联失效过程)')
print(f'    - comprehensive_visualization.png (综合可视化)')

# %% Cell 13: 级联失效敏感性分析 — α × 攻击规模 扫描
def cascading_failure_multi(G_original, df_importance, target_nodes, capacity_factor=0.2, verbose=False):
    """
    多节点同时攻击 + 级联失效模拟。
    先移除所有 target_nodes，然后迭代检测超载-移除，直到稳定。
    """
    G = G_original.copy()
    n_original = G.number_of_nodes()

    # 初始状态 (攻击前)
    bc_init = nx.betweenness_centrality(G, weight='distance')
    total_flow = sum(data['weight'] for _, _, data in G.edges(data=True))
    initial_load = {n: bc_init[n] * total_flow for n in G.nodes()}
    capacity = {n: L * (1 + capacity_factor) for n, L in initial_load.items()}

    # 同时移除所有攻击目标
    actual_targets = [n for n in target_nodes if n in G]
    G.remove_nodes_from(actual_targets)

    removal_sequence = list(actual_targets)
    cascade_step = 0
    max_steps = 100

    while cascade_step < max_steps and G.number_of_nodes() > 1:
        cascade_step += 1
        current_nodes = set(G.nodes())
        try:
            bc_new = nx.betweenness_centrality(G, weight='distance')
        except Exception:
            break

        current_load = {n: bc_new[n] * total_flow for n in current_nodes}
        overloaded = [n for n in current_nodes if current_load[n] > capacity[n]]

        if not overloaded:
            break

        for n in overloaded:
            removal_sequence.append(n)
            G.remove_node(n)

        if G.number_of_nodes() <= 1:
            break

    return {
        'total_removed': len(removal_sequence),
        'initial_attacked': len(actual_targets),
        'cascade_removed': len(removal_sequence) - len(actual_targets),
        'cascade_steps': cascade_step,
        'n_surviving': G.number_of_nodes(),
        'n_original': n_original,
    }


# 扫描网格
alpha_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
attack_sizes = [1, 2, 3, 5, 7, 10, 15, 20]

n_alpha = len(alpha_values)
n_attack = len(attack_sizes)

grid_total_pct = np.zeros((n_alpha, n_attack))    # 总移除比例 %
grid_cascade_n = np.zeros((n_alpha, n_attack))     # 纯级联波及节点数
grid_steps = np.zeros((n_alpha, n_attack))         # 级联步数

print(f'开始 α×攻击规模扫描 ({n_alpha}×{n_attack} = {n_alpha*n_attack} 次模拟)...')
t_scan = time.time()

for i, alpha in enumerate(alpha_values):
    for j, k in enumerate(attack_sizes):
        targets = df_importance.index[:k].tolist()
        r = cascading_failure_multi(G, df_importance, targets, capacity_factor=alpha)

        grid_total_pct[i, j] = r['total_removed'] / N0 * 100
        grid_cascade_n[i, j] = r['cascade_removed']
        grid_steps[i, j] = r['cascade_steps']

        badge = '!'
        if r['cascade_removed'] > 0:
            badge = 'C' if r['cascade_removed'] <= 5 else 'C+'

        print(f'  α={alpha:.2f} | Top-{k:<2d} | '
              f'移除 {r["total_removed"]:>3d} ({grid_total_pct[i, j]:5.1f}%) | '
              f'其中级联 {r["cascade_removed"]:>3d} 站 | '
              f'步数 {r["cascade_steps"]} | [{badge}]')

print(f'[OK] 敏感性扫描完成, 耗时 {(time.time()-t_scan)/60:.1f} 分钟')

# %% Cell 14: 级联失效敏感性热力图 (海报核心图)
import matplotlib.colors as mcolors

fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

# --- 统一参数 ---
font_kw = {'fontsize': 11}
xtick_labels = [f'Top-{k}' for k in attack_sizes]
ytick_labels = [f'{a:.2f}' for a in alpha_values]

# ========== 子图 1: 级联波及节点数 ==========
ax1 = axes[0]
im1 = ax1.imshow(grid_cascade_n, cmap='YlOrRd', aspect='auto', origin='lower',
                 norm=mcolors.LogNorm(vmin=max(1, grid_cascade_n.min()), vmax=max(10, grid_cascade_n.max())))
ax1.set_xticks(range(n_attack))
ax1.set_yticks(range(n_alpha))
ax1.set_xticklabels(xtick_labels, **font_kw)
ax1.set_yticklabels(ytick_labels, **font_kw)
ax1.set_xlabel('蓄意攻击规模', fontsize=12)
ax1.set_ylabel('容量冗余系数 α', fontsize=12)
ax1.set_title('纯级联波及节点数', fontsize=14, fontweight='bold')

# 在格子内标注数值
for i in range(n_alpha):
    for j in range(n_attack):
        val = grid_cascade_n[i, j]
        color = 'white' if val > grid_cascade_n.max() / 2 else 'black'
        ax1.text(j, i, f'{int(val)}', ha='center', va='center',
                 fontsize=9, fontweight='bold', color=color)
cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
cbar1.set_label('节点数 (log)', fontsize=10)

# ========== 子图 2: 总失效比例 % ==========
ax2 = axes[1]
im2 = ax2.imshow(grid_total_pct, cmap='Reds', aspect='auto', origin='lower')
ax2.set_xticks(range(n_attack))
ax2.set_yticks(range(n_alpha))
ax2.set_xticklabels(xtick_labels, **font_kw)
ax2.set_yticklabels(ytick_labels, **font_kw)
ax2.set_xlabel('蓄意攻击规模', fontsize=12)
ax2.set_ylabel('容量冗余系数 α', fontsize=12)
ax2.set_title('总失效节点比例 (%)', fontsize=14, fontweight='bold')

for i in range(n_alpha):
    for j in range(n_attack):
        val = grid_total_pct[i, j]
        color = 'white' if val > 15 else 'black'
        ax2.text(j, i, f'{val:.1f}', ha='center', va='center',
                 fontsize=9, fontweight='bold', color=color)
cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
cbar2.set_label('%', fontsize=10)

# ========== 子图 3: 级联步数 ==========
ax3 = axes[2]
im3 = ax3.imshow(grid_steps, cmap='Purples', aspect='auto', origin='lower')
ax3.set_xticks(range(n_attack))
ax3.set_yticks(range(n_alpha))
ax3.set_xticklabels(xtick_labels, **font_kw)
ax3.set_yticklabels(ytick_labels, **font_kw)
ax3.set_xlabel('蓄意攻击规模', fontsize=12)
ax3.set_ylabel('容量冗余系数 α', fontsize=12)
ax3.set_title('级联迭代步数', fontsize=14, fontweight='bold')

for i in range(n_alpha):
    for j in range(n_attack):
        val = grid_steps[i, j]
        color = 'white' if val > 3 else 'black'
        ax3.text(j, i, f'{int(val)}', ha='center', va='center',
                 fontsize=9, fontweight='bold', color=color)
cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
cbar3.set_label('步数', fontsize=10)

plt.tight_layout()
plt.savefig('cascade_sensitivity_heatmap.png', dpi=250, bbox_inches='tight')
plt.show()
print('[OK] 级联敏感性热力图已保存为 cascade_sensitivity_heatmap.png')

# ---- 关键阈值分析 ----
print(f'\n{"="*60}')
print(f'  级联失效敏感性分析 — 关键发现')
print(f'{"="*60}')

# 找出"安全区"的边界: 对每个 α，多大的攻击开始引发级联
print(f'  ┌{"─"*10}┬{"─"*10}┬{"─"*12}┬{"─"*12}┐')
print(f'  │ {"α":>8} │ {"攻击规模":>8} │ {"级联波及":>10} │ {"总失效%":>10} │')
print(f'  ├{"─"*10}┼{"─"*10}┼{"─"*12}┼{"─"*12}┤')
for i, alpha in enumerate(alpha_values):
    for j, k in enumerate(attack_sizes):
        if grid_cascade_n[i, j] > 0:
            print(f'  │ {alpha:>8.2f} │ Top-{k:<5d} │ {int(grid_cascade_n[i, j]):>10d} │ '
                  f'{grid_total_pct[i, j]:>10.1f}% │')
            break
    else:
        print(f'  │ {alpha:>8.2f} │ 始终安全   │ {"--":>10} │ {"--":>10} │')
print(f'  └{"─"*10}┴{"─"*10}┴{"─"*12}┴{"─"*12}┘')
