"""
城市轨道交通网络级联失效 (Cascading Failure) 模型
====================================================
基于真实 AFC 客流数据 + Space-L 拓扑网络
运行方式: D:\miniconda\envs\network_env\python.exe cascading_failure_model.py

模型逻辑：
  1. 负荷 L_i = 节点加权介数中心性 (使用 distance 权重)
  2. 容量 C_i = L_i * (1 + alpha),  alpha = 0.2
  3. 蓄意攻击：瘫痪综合重要度 Top-1 站点
  4. 重新分配最短路径 -> 更新各节点负荷
  5. 移除所有超载节点 (current_load > capacity)
  6. 迭代 4-5 直到网络稳定或无节点可移除
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import gc
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# Part A: 数据加载与网络构建
# ============================================================

def load_station_info(filepath='stationInfo.csv'):
    """加载站点信息，清理 neighbour 列并返回 DataFrame"""
    station_info = pd.read_csv(filepath)
    station_info.columns = station_info.columns.str.strip()
    print(f"[1/6] 站点信息加载完成: {len(station_info)} 个站点")
    return station_info


def load_od_flow(filepath='metroData_ODFlow.csv', chunksize=2000000):
    """
    高效加载 OD 客流数据，使用 defaultdict 累积避免内存爆炸
    (这是对你原 Cell 2 的优化版本)
    """
    bad_dates = [20170504, 20170508, 20170509, 20170616, 20170627, 20170628]
    flow_accumulator = defaultdict(int)

    print(f"[2/6] 开始处理 OD 客流数据 (chunksize={chunksize})...")

    chunk_count = 0
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        chunk_count += 1
        chunk.columns = chunk.columns.str.strip()

        # 过滤异常日期
        if 'date' in chunk.columns:
            chunk = chunk[~chunk['date'].isin(bad_dates)]

        # 使用真实列名
        o_col = 'originStation'
        d_col = 'destinationStation'
        f_col = 'Flow'

        # 按 (O, D) 对汇总本 chunk，然后累积到 dict
        grouped = chunk.groupby([o_col, d_col])[f_col].sum()
        for (o, d), flow_val in grouped.items():
            flow_accumulator[(int(o), int(d))] += flow_val

        if chunk_count % 10 == 0:
            print(f"  已处理 {chunk_count * chunksize / 1e6:.0f}M 行, "
                  f"当前 {len(flow_accumulator)} 对 OD...")

    print(f"[2/6] OD 数据处理完成: {len(flow_accumulator)} 对 OD 关系")
    return flow_accumulator


def build_weighted_network(station_info, flow_dict):
    """构建客流加权的 Space-L 网络"""
    print("[3/6] 构建加权 Space-L 网络...")

    id_col = 'stationID'
    name_col = 'name'
    lon_col = 'lon'
    lat_col = 'lat'
    neighbor_col = 'neighbour'

    G = nx.Graph()

    # 添加节点
    for _, row in station_info.iterrows():
        G.add_node(int(row[id_col]),
                   name=row[name_col],
                   lon=row[lon_col],
                   lat=row[lat_col])

    # 添加边
    for _, row in station_info.iterrows():
        u = int(row[id_col])
        raw_neighbor_str = str(row[neighbor_col])
        clean_str = raw_neighbor_str.replace('[', '').replace(']', '').replace('，', ',')
        neighbors = clean_str.split(',')

        for v in neighbors:
            v = v.strip()
            if v and v.lower() != 'nan':
                v = int(float(v))
                if not G.has_edge(u, v):
                    flow_u_v = flow_dict.get((u, v), 0)
                    flow_v_u = flow_dict.get((v, u), 0)
                    total_flow = flow_u_v + flow_v_u

                    weight = total_flow if total_flow > 0 else 1
                    distance = 1.0 / weight

                    G.add_edge(u, v, weight=weight, distance=distance)

    # 剔除孤立节点
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    if isolated_nodes:
        print(f"  剔除 {len(isolated_nodes)} 个孤立节点")

    print(f"[3/6] 网络构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    return G


# ============================================================
# Part B: 网络指标计算
# ============================================================

def compute_all_metrics(G):
    """计算度中心性、加权介数中心性、聚类系数、节点客流强度"""
    print("[4/6] 计算网络特征指标...")

    degree_cent = nx.degree_centrality(G)
    between_cent = nx.betweenness_centrality(G, weight='distance')
    cluster_coeff = nx.clustering(G, weight='weight')
    node_strength = {node: sum(data['weight'] for _, _, data in G.edges(node, data=True))
                     for node in G.nodes()}

    df = pd.DataFrame({
        'Degree': pd.Series(degree_cent),
        'Betweenness': pd.Series(between_cent),
        'Clustering': pd.Series(cluster_coeff),
        'Strength': pd.Series(node_strength)
    })

    print(f"[4/6] 指标计算完成: {len(df)} 个节点")
    return df


# ============================================================
# Part C: 熵权法 (修正版) -- 直接对原始指标赋权
# ============================================================

def entropy_weight_method(df, indicator_cols):
    """
    熵权法客观赋权 (修正版)

    修正要点:
    - 直接对原始指标计算熵权，而非先因子分析降维再赋权
    - 原代码将熵权法应用于已正交的因子得分上，权重必然趋近均等 (0.5, 0.5)，
      失去了"按信息量赋权"的意义。修正后直接对 4 个原始指标赋权。

    公式:
      P_ij = x'_ij / sum_i(x'_ij)       (x' 为 min-max 归一化 + 非负平移)
      e_j  = -k * sum_i(P_ij * ln(P_ij))
      w_j  = (1 - e_j) / sum_j(1 - e_j)
    """
    X = df[indicator_cols].values.astype(float)
    n, m = X.shape

    # 归一化
    X_norm = np.zeros_like(X)
    for j in range(m):
        col_min, col_max = X[:, j].min(), X[:, j].max()
        if col_max - col_min == 0:
            X_norm[:, j] = 1.0
        else:
            X_norm[:, j] = (X[:, j] - col_min) / (col_max - col_min)

    # 非负平移 (避免 ln(0))
    X_norm = X_norm + 1e-12

    # 比重矩阵
    P = X_norm / X_norm.sum(axis=0, keepdims=True)

    # 熵值
    k = 1.0 / np.log(n)
    e = -k * np.sum(P * np.log(P), axis=0)

    # 熵权
    d = 1 - e
    w = d / d.sum()

    weights = {col: float(w[j]) for j, col in enumerate(indicator_cols)}
    return weights


def compute_importance_entropy(df_metrics):
    """
    使用修正后的熵权法计算站点综合重要度。
    直接对 Degree, Betweenness, Clustering, Strength 四个原始指标赋权。
    """
    from sklearn.preprocessing import MinMaxScaler

    indicator_cols = ['Degree', 'Betweenness', 'Clustering', 'Strength']

    weights = entropy_weight_method(df_metrics, indicator_cols)

    print("\n---------- 修正后的熵权法权重 ----------")
    for k, v in weights.items():
        print(f"  {k:15s}: {v:.4f}")

    # 归一化后加权求和
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df_metrics[indicator_cols]),
        columns=indicator_cols,
        index=df_metrics.index
    )

    df_metrics = df_metrics.copy()
    df_metrics['Importance'] = sum(
        weights[col] * scaled[col] for col in indicator_cols
    )

    return df_metrics.sort_values('Importance', ascending=False)


# ============================================================
# Part D: 级联失效模型 (核心新增)
# ============================================================

def cascading_failure(G_original, df_importance, capacity_factor=0.2, verbose=True):
    """
    级联失效模拟

    参数
    ----
    G_original : nx.Graph
        初始网络
    df_importance : pd.DataFrame
        按 Importance 降序排列的站点表 (index 为 stationID)
    capacity_factor : float
        容量冗余系数 (默认 0.2, 即容量 = 初始负荷 * 1.2)
    verbose : bool
        是否打印详细过程

    返回
    ----
    result : dict
    """
    G = G_original.copy()
    n_original = G.number_of_nodes()

    # ----- 1. 计算初始负荷 (使用加权介数中心性 * 总客流) -----
    bc_init = nx.betweenness_centrality(G, weight='distance')
    total_flow = sum(data['weight'] for _, _, data in G.edges(data=True))

    initial_load = {n: bc_init[n] * total_flow for n in G.nodes()}
    capacity = {n: L * (1 + capacity_factor) for n, L in initial_load.items()}

    # ----- 2. 蓄意攻击：瘫痪综合重要度 Top-1 站点 -----
    top_node = df_importance.index[0]
    top_name = G.nodes[top_node].get('name', str(top_node))
    top_score = df_importance.iloc[0]['Importance']

    if verbose:
        print(f"\n{'='*55}")
        print(f"  级联失效模拟开始")
        print(f"{'='*55}")
        print(f"  初始攻击目标: {top_name} (ID={top_node})")
        print(f"  综合重要度:   {top_score:.4f}")
        print(f"  初始负荷:     {initial_load[top_node]:.2f}")
        print(f"  容量:         {capacity[top_node]:.2f}")

    G.remove_node(top_node)

    # ----- 3. 级联迭代 -----
    removal_sequence = [top_node]
    overload_events = []
    cascade_step = 0
    max_steps = 100

    while cascade_step < max_steps and G.number_of_nodes() > 1:
        cascade_step += 1
        current_nodes = set(G.nodes())

        # 重新计算加权介数中心性
        try:
            bc_new = nx.betweenness_centrality(G, weight='distance')
        except Exception:
            if verbose:
                print(f"  [步 {cascade_step}] 网络破碎，终止")
            break

        # 更新负荷
        current_load = {n: bc_new[n] * total_flow for n in current_nodes}

        # 检查超载
        overloaded = []
        for n in current_nodes:
            if current_load[n] > capacity[n]:
                overloaded.append((n, current_load[n], capacity[n]))

        if not overloaded:
            if verbose:
                print(f"  [步 {cascade_step}] [OK] 无超载，网络已稳定")
            break

        # 记录
        overloaded.sort(key=lambda x: x[1], reverse=True)
        overload_events.append({
            'step': cascade_step,
            'n_overloaded': len(overloaded),
            'nodes': [x[0] for x in overloaded],
        })

        if verbose:
            print(f"  [步 {cascade_step}] 超载 {len(overloaded)} 个节点:")
            for name, load, cap in overloaded[:5]:
                node_name = G_original.nodes[name].get('name', str(name))
                print(f"    {node_name}: load={load:.1f} > cap={cap:.1f} "
                      f"(超载 {(load/cap - 1)*100:.0f}%)")
            if len(overloaded) > 5:
                print(f"    ... 其余 {len(overloaded) - 5} 个")

        # 移除超载节点
        for n, _, _ in overloaded:
            removal_sequence.append(n)
            G.remove_node(n)

        if G.number_of_nodes() <= 1:
            if verbose:
                print(f"  [步 {cascade_step}] 网络基本崩溃 ({G.number_of_nodes()} 节点剩余)")
            break

    # ----- 4. 结果汇总 -----
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
        print(f"\n  -------- 级联失效汇总 --------")
        print(f"  初始攻击:     {top_name}")
        print(f"  级联步数:     {cascade_step}")
        print(f"  总移除节点:   {len(removal_sequence)} / {n_original} "
              f"({len(removal_sequence)/n_original*100:.1f}%)")
        print(f"  其中级联波及: {len(removal_sequence) - 1}")
        print(f"  幸存节点:     {G.number_of_nodes()}")

    return result


# ============================================================
# Part E: 对比实验与可视化
# ============================================================

def compare_cascade_vs_nocascade(G, df_importance, capacity_factor=0.2):
    """对比无级联 vs 有级联的效果差异"""
    n_original = G.number_of_nodes()

    # ---- A: 无级联 ----
    G_no_cascade = G.copy()
    top_node = df_importance.index[0]
    G_no_cascade.remove_node(top_node)
    eff_no = nx.global_efficiency(G_no_cascade) if G_no_cascade.number_of_nodes() > 1 else 0

    lcc_no = 0
    if G_no_cascade.number_of_nodes() > 0:
        lcc_no = len(max(nx.connected_components(G_no_cascade), key=len))

    # ---- B: 有级联 ----
    cascade_result = cascading_failure(G, df_importance, capacity_factor, verbose=True)
    G_cascade = cascade_result['final_graph']
    eff_cas = nx.global_efficiency(G_cascade) if G_cascade.number_of_nodes() > 1 else 0

    lcc_cas = 0
    if G_cascade.number_of_nodes() > 0:
        lcc_cas = len(max(nx.connected_components(G_cascade), key=len))

    print(f"\n{'='*60}")
    print(f"  级联失效 vs 无级联 对比")
    print(f"{'='*60}")
    print(f"  {'指标':<20} {'仅移除Top-1':>12} {'Top-1+级联':>12}")
    print(f"  {'幸存节点数':<20} {G_no_cascade.number_of_nodes():>12} {G_cascade.number_of_nodes():>12}")
    print(f"  {'幸存比例':<20} {G_no_cascade.number_of_nodes()/n_original:>12.2%} "
          f"{G_cascade.number_of_nodes()/n_original:>12.2%}")
    print(f"  {'最大连通分量':<20} {lcc_no:>12} {lcc_cas:>12}")
    print(f"  {'LCC比例':<20} {lcc_no/n_original:>12.2%} {lcc_cas/n_original:>12.2%}")
    print(f"  {'全局效率':<20} {eff_no:>12.6f} {eff_cas:>12.6f}")

    return cascade_result


def plot_cascade_process(cascade_result):
    """可视化级联失效过程"""
    events = cascade_result['overload_events']
    if not events:
        print("\n(无超载事件，跳过绘图)")
        return

    steps = [e['step'] for e in events]
    n_over = [e['n_overloaded'] for e in events]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左: 每步超载数
    axes[0].bar(steps, n_over, color='tomato', edgecolor='black')
    axes[0].set_xlabel('级联步数', fontsize=12)
    axes[0].set_ylabel('超载节点数', fontsize=12)
    axes[0].set_title('各步新增超载节点', fontsize=14)
    axes[0].set_xticks(steps)
    axes[0].grid(axis='y', alpha=0.3)

    # 右: 累积移除
    cumulative = [1] + list(np.cumsum(n_over))
    axes[1].plot(range(len(cumulative)), cumulative, 'o-',
                 color='darkred', linewidth=2, markersize=8)
    axes[1].set_xlabel('迭代步数 (0 = 初始攻击)', fontsize=12)
    axes[1].set_ylabel('累积移除节点数', fontsize=12)
    axes[1].set_title('累积失效节点曲线', fontsize=14)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('cascade_process.png', dpi=150)
    plt.show()
    print("\n[图已保存] cascade_process.png")


# ============================================================
# Part F: 主程序
# ============================================================

if __name__ == "__main__":
    import time
    t0 = time.time()

    print("=" * 60)
    print("  城市轨道交通网络韧性分析 -- 级联失效模型")
    print("=" * 60)

    # 切换工作目录 (如需要)
    import os
    os.chdir(r'C:\Users\93966\Metro_Resilience_Project')

    # ---- 加载数据 ----
    station_info = load_station_info('stationInfo.csv')
    flow_dict = load_od_flow('metroData_ODFlow.csv', chunksize=2000000)

    # ---- 构建网络 ----
    G = build_weighted_network(station_info, flow_dict)

    # ---- 计算指标 ----
    df_metrics = compute_all_metrics(G)

    # ---- 修正后熵权法计算综合重要度 ----
    df_importance = compute_importance_entropy(df_metrics)
    print("\n---------- Top-10 关键站点 (综合重要度) ----------")
    top10 = df_importance.head(10)
    for i, (idx, row) in enumerate(top10.iterrows()):
        name = G.nodes[idx].get('name', str(idx))
        print(f"  {i+1:2d}. {name:12s} (ID={idx:4d})  "
              f"Importance={row['Importance']:.4f}  "
              f"Deg={row['Degree']:.3f}  Btw={row['Betweenness']:.3f}  "
              f"Str={row['Strength']:.0f}")

    # ---- 级联失效模拟 ----
    cascade_result = compare_cascade_vs_nocascade(G, df_importance, capacity_factor=0.2)

    # ---- 可视化 ----
    plot_cascade_process(cascade_result)

    # ---- 关键保护建议 ----
    print(f"\n{'='*60}")
    print(f"  关键保护建议")
    print(f"{'='*60}")
    top5_ids = df_importance.head(5).index.tolist()
    top5_names = [G.nodes[i].get('name', str(i)) for i in top5_ids]
    print(f"  建议重点保护的 Top-5 站点: {top5_names}")
    print(f"  级联失效可能波及 {cascade_result['total_removed'] - 1} 个额外站点")
    print(f"  若不进行级联防护，网络效率将从正常水平骤降至"
          f" {nx.global_efficiency(cascade_result['final_graph']):.6f}")

    elapsed = time.time() - t0
    print(f"\n[完成] 总耗时: {elapsed/60:.1f} 分钟")
