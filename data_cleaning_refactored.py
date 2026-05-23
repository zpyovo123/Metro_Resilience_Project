"""
城市轨道交通 AFC 刷卡数据清洗脚本 (内存优化重构版)
=====================================================
原版问题:
  1. pd.concat([累计表, 新块]) 在循环内反复创建新 DataFrame —— O(n²) 内存拷贝
  2. 每次 concat 后又做全量 groupby —— 重复扫描
  3. 列名检测逻辑用硬编码索引 chunk.columns[4/5/6]，脆弱

重构策略 (双方案):
  方案 A: defaultdict 累积 (推荐用于本场景)
    - 每 chunk 只做 groupby, 结果累加进 defaultdict
    - 无 DataFrame 累积, 内存 O(OD对数) ≈ 常量
    - 最后一次性转为 dict/DataFrame

  方案 B: Dask 并行 (适合更复杂的聚合逻辑)
    - 延迟计算, 自动分块
    - 内存可控
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import time
import gc
import os


# ============================================================
# 方案 A: defaultdict 累积 (推荐)
# ============================================================

def clean_od_data_optimized(od_file_path, bad_dates=None, chunksize=2000000,
                             output_mode='dict', max_rows=None):
    """
    高效清洗 OD 客流数据 (不重复复制 DataFrame)

    参数
    ----
    od_file_path : str
        OD 数据 CSV 文件路径
    bad_dates : list
        需要排除的异常日期
    chunksize : int
        每块读取行数
    output_mode : 'dict' | 'df'
        'dict' 返回 {(O, D): flow} 字典
        'df'   返回聚合好的 DataFrame
    max_rows : int or None
        测试用：限制总读取行数 (None = 读取全部)

    返回
    ----
    dict 或 DataFrame, 列: ['O-Station', 'D-Station', 'Flow']
    """
    if bad_dates is None:
        bad_dates = [20170504, 20170508, 20170509, 20170616, 20170627, 20170628]

    print(f"开始处理 OD 数据: {od_file_path}")
    print(f"  chunksize = {chunksize:,} 行/块")
    print(f"  异常日期 = {bad_dates}")
    t0 = time.time()

    # ---- Step 1: 探测真实列名 ----
    first_chunk = pd.read_csv(od_file_path, nrows=5)
    first_chunk.columns = first_chunk.columns.str.strip()
    cols = first_chunk.columns.tolist()
    print(f"  列名: {cols}")

    # 自动匹配列名 (不依赖硬编码索引)
    date_col = _find_column(cols, ['date', 'Date', 'DATE'])
    o_col = _find_column(cols, ['originStation', 'O-Station', 'o_station', 'origin_station'])
    d_col = _find_column(cols, ['destinationStation', 'D-Station', 'd_station', 'destination_station'])
    f_col = _find_column(cols, ['Flow', 'flow', 'FLOW'])

    print(f"  匹配列: date={date_col}, O={o_col}, D={d_col}, Flow={f_col}")

    # ---- Step 2: 分块读取 + defaultdict 累积 ----
    flow_accumulator = defaultdict(int)
    chunk_count = 0
    total_rows_read = 0

    for chunk in pd.read_csv(od_file_path, chunksize=chunksize):
        chunk.columns = chunk.columns.str.strip()
        chunk_count += 1
        total_rows_read += len(chunk)

        # 过滤异常日期
        chunk = chunk[~chunk[date_col].isin(bad_dates)]

        # 本 chunk 聚合
        grouped = chunk.groupby([o_col, d_col])[f_col].sum()

        # 累积到 dict (只做 O(1) 的加法, 零拷贝)
        for (o, d), flow_val in grouped.items():
            flow_accumulator[(int(o), int(d))] += flow_val

        if chunk_count % 10 == 0:
            print(f"  已处理 {total_rows_read / 1e6:.0f}M 行, "
                  f"{len(flow_accumulator):,} 对 OD")

        # 测试模式: 限制行数
        if max_rows is not None and total_rows_read >= max_rows:
            print(f"  [测试模式] 达到限制 {max_rows:,} 行, 停止读取")
            break

    elapsed = time.time() - t0
    print(f"  读取完成: {total_rows_read:,} 行 → {len(flow_accumulator):,} 对 OD")
    print(f"  耗时: {elapsed:.1f} 秒")

    if output_mode == 'df':
        # 转为 DataFrame
        records = [{'O-Station': o, 'D-Station': d, 'Flow': f}
                   for (o, d), f in flow_accumulator.items()]
        df = pd.DataFrame(records)
        print(f"  DataFrame 大小: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
        return df
    else:
        return dict(flow_accumulator)


def _find_column(cols, candidates):
    """从列名列表中匹配第一个存在的候选列名"""
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"无法匹配列名, 现有列: {cols}, 候选: {candidates}")


# ============================================================
# 方案 B: Dask (适合需要更复杂操作的场景)
# ============================================================

def clean_od_data_dask(od_file_path, bad_dates=None, blocksize='200MB'):
    """
    使用 Dask 清洗 OD 数据 (延迟计算, 自动分块)

    需要安装: pip install dask[complete]
    """
    import dask.dataframe as dd

    if bad_dates is None:
        bad_dates = [20170504, 20170508, 20170509, 20170616, 20170627, 20170628]

    print(f"开始 Dask 处理: {od_file_path}")
    t0 = time.time()

    # 延迟读取 (不立即加载数据)
    ddf = dd.read_csv(od_file_path, blocksize=blocksize,
                      dtype={'Flow': 'float64'})
    ddf.columns = ddf.columns.str.strip()

    # 探测列名
    cols = ddf.columns.tolist()
    date_col = _find_column(cols, ['date', 'Date', 'DATE'])
    o_col = _find_column(cols, ['originStation', 'O-Station'])
    d_col = _find_column(cols, ['destinationStation', 'D-Station'])
    f_col = _find_column(cols, ['Flow', 'flow'])

    # 过滤
    ddf_filtered = ddf[~ddf[date_col].isin(bad_dates)]

    # 聚合 (延迟, 当前无计算)
    ddf_agg = ddf_filtered.groupby([o_col, d_col])[f_col].sum()

    # 触发计算
    result = ddf_agg.compute()

    # 转为 dict
    flow_dict = {(int(o), int(d)): float(f)
                 for (o, d), f in result.items()}

    elapsed = time.time() - t0
    print(f"Dask 处理完成: {len(flow_dict):,} 对 OD, 耗时 {elapsed:.1f} 秒")
    return flow_dict


# ============================================================
# 测试用: 创建 1000 行样本
# ============================================================

def create_test_sample(input_path, output_path, nrows=1000, seed=42):
    """
    从 11GB 大文件中抽取前 nrows 行作为测试样本
    """
    print(f"从 {input_path} 抽取 {nrows} 行 → {output_path}")
    df_sample = pd.read_csv(input_path, nrows=nrows)
    df_sample.to_csv(output_path, index=False)
    print(f"  测试文件已创建: {output_path} ({os.path.getsize(output_path):,} bytes)")
    return output_path


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='OD数据清洗脚本 (重构版)')
    parser.add_argument('--input', default='metroData_ODFlow.csv',
                        help='输入 CSV 文件路径')
    parser.add_argument('--chunksize', type=int, default=2000000,
                        help='每块行数 (默认 200万)')
    parser.add_argument('--max-rows', type=int, default=None,
                        help='限制读取行数 (测试用)')
    parser.add_argument('--test', action='store_true',
                        help='使用内置 1000 行测试模式')
    parser.add_argument('--method', choices=['defaultdict', 'dask'],
                        default='defaultdict', help='处理方法 (默认 defaultdict)')
    parser.add_argument('--output-mode', choices=['dict', 'df'],
                        default='dict', help='输出格式')
    args = parser.parse_args()

    # 切换工作目录
    os.chdir(r'C:\Users\93966\Metro_Resilience_Project')

    if args.test:
        # 内置测试: 用 1000 行样本跑
        test_file = 'metroData_test_1000rows.csv'
        if not os.path.exists(test_file):
            create_test_sample(args.input, test_file, nrows=1000)
        od_file = test_file
        args.max_rows = 1000
        print("\n========== 测试模式 (1000 行) ==========\n")
    else:
        od_file = args.input

    # 运行清洗
    if args.method == 'dask':
        result = clean_od_data_dask(od_file)
    else:
        result = clean_od_data_optimized(
            od_file,
            chunksize=args.chunksize,
            max_rows=args.max_rows,
            output_mode=args.output_mode
        )

    # 打印摘要
    if args.output_mode == 'dict' or isinstance(result, dict):
        print(f"\n最终结果: {len(result)} 对 OD")
        # 显示前5条
        items = list(result.items())[:5]
        for (o, d), f in items:
            print(f"  ({o:>6}, {d:>6}): {f:>10.0f}")
    else:
        print(f"\n最终结果: {len(result)} 行")
        print(result.head())

    print("\n[完成] 清洗结束!")
