#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_parquet.py

1_lake/turn.parquet 또는 다른 Parquet 디렉터리를 인자로 받아
메타데이터와 샘플을 요약 출력합니다.
usage: python inspect_parquet.py 1_lake/turn.parquet
"""
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd
import dask.dataframe as dd

def inspect_parquet(parquet_dir: Path, n_sample: int = 5):
    print(f"\n▶ Inspecting Parquet directory: {parquet_dir}\n")
    # --- 1) PyArrow Dataset & metadata ---
    dataset = pq.ParquetDataset(str(parquet_dir))
    # compute total rows across all parquet files
    total_rows = 0
    for f in Path(parquet_dir).rglob("*.parquet"):
        pf = pq.ParquetFile(str(f))
        total_rows += pf.metadata.num_rows
    print(f"총   레코드 수: {total_rows:,}")
    schema = dataset.schema

    # 컬럼 스키마
    print("\n컬럼 스키마:")
    for field in schema:
        print(f"  • {field.name:20} {field.type}")

    # 파티션 키 & 값 (filesystem-based)
    partition_values = {}
    for path in Path(parquet_dir).rglob("*"):
        if path.is_dir():
            parts = path.parts[len(parquet_dir.parts):]
            for part in parts:
                if '=' in part:
                    key, val = part.split('=', 1)
                    partition_values.setdefault(key, set()).add(val)
    partition_keys = sorted(partition_values.keys())
    print(f"\n파티션 키: {partition_keys}")
    print("파티션별 고유값 예시:")
    for key in partition_keys:
        vals = sorted(partition_values[key])
        print(f"  - {key}: {vals[:5]}{' …' if len(vals) > 5 else ''}")

    # 로우그룹 정보
    # (metadata removed: cannot show row-group info without a single metadata object)

    # --- 2) Dask DataFrame 로딩 & 샘플 ---
    print("\n▶ Dask DataFrame 으로 로드하여 dtypes 및 샘플 출력")
    ddf = dd.read_parquet(str(parquet_dir))
    print("\n컬럼별 dtypes:")
    print(ddf.dtypes.to_string())

    print(f"\n상위 {n_sample}개 레코드 샘플:")
    pdf = ddf.head(n_sample)
    # pandas 옵션: 모든 컬럼 다 보이게
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(pdf)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_parquet.py <parquet_dir>")
        sys.exit(1)
    parquet_dir = Path(sys.argv[1])
    if not parquet_dir.exists():
        print("Error: path not found:", parquet_dir)
        sys.exit(1)
    inspect_parquet(parquet_dir)