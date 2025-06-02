#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
etl_od_parquet.py

부천시 기종점 교통량 CSV → Parquet
 Mac 리소스포크 자동 필터링
 제어문자 제거
 원본 칼럼 100% 보존 + 메타칼럼 추가
 적절한 dtype 캐스팅
usage: python etl_od_parquet.py
"""
from __future__ import annotations
import re, glob, json
from pathlib import Path

import pandas as pd
import dask.dataframe as dd
from dask import delayed
from dask.diagnostics import ProgressBar

RAW_DIR   = "0_raw/교통량데이터(기종점교통량기반)"
OUT_DIR   = "1_lake/od.parquet"
GLOB_PAT  = f"{RAW_DIR}/**/*.csv"

# 제어문자/널 제거용 정규식
PAT_CTRL = re.compile(r"[\x00-\x1F]+")

# 1) 파일 목록 수집 & 리소스포크(. _*) 필터링
all_csvs = glob.glob(GLOB_PAT, recursive=True)
csv_paths = [
    p for p in all_csvs
    if not Path(p).name.startswith("._")
       and p.lower().endswith(".csv")
]
print(f"🔍 처리 대상 CSV 파일 수: {len(csv_paths):,}")

# 2) 단일 CSV → pandas.DataFrame
def parse_one_csv(path: str) -> pd.DataFrame:
    path = Path(path)
    # ── 메타 정보 추출 ──────────────────
    route = path.parents[1].name      # ex) 교차로_26
    date  = path.parents[0].name      # ex) 20220810
    # ── CSV 로드 ───────────────────────
    df = pd.read_csv(path, dtype=str)  # 일단 전부 문자열로
    # ── 제어문자 제거 ────────────────────
    for col in df.columns:
        # NaN도 문자열로 읽히므로 str → 제거 → 다시 NA 처리
        df[col] = (
            df[col]
            .fillna("")                      # NA → ""
            .astype(str)
            .str.replace(PAT_CTRL, "", regex=True)
            .replace({"": pd.NA})           # "" → NA
        )
    # ── 메타칼럼 추가 ────────────────────
    df["route"] = route
    df["date"]  = date
    df["path"]  = str(path)
    # ── 자료형 캐스팅 ────────────────────
    # 수치형으로 변환 가능한 칼럼들만 변환
    to_int = [
        "unix_time","stdr_ymd","stdr_hm","stdr_ss",
        "vhcl_typ","que_all","que_200","que_200_500","que_500"
    ]
    to_float = [
        "lon","lat","alt","spd","allowed_spd",
        "agl","slp","dist2to_inter","dist2nxt_inter","tl"
    ]
    for c in to_int:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in to_float:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Float64")
    # (나머지 칼럼은 pandas.StringDtype)
    str_cols = [c for c in df.columns if df[c].dtype == object]
    df[str_cols] = df[str_cols].astype("string")
    return df

# 3) Dask로 병합 & dt 생성
delayed_dfs = [delayed(parse_one_csv)(p) for p in csv_paths]
# 메타 정의
meta = parse_one_csv(csv_paths[0]).iloc[0:0]
ddf = dd.from_delayed(delayed_dfs, meta=meta)

# dt: unix_time → datetime → 15분 버킷
ddf["dt"] = (
    dd.to_datetime(ddf["unix_time"].astype("int64"), unit="s", errors="coerce")
      .dt.floor("15min")
)

# 4) Parquet 저장
print("⏳ Parquet 저장 시작…")
ProgressBar().register()
ddf.to_parquet(
    OUT_DIR,
    engine="pyarrow",
    compression="zstd",
    partition_on=["route", "date"],
    write_index=False,
)
print(f"✓ 저장 완료 → {OUT_DIR}")

# 5) 최종 칼럼 목록 출력
cols = list(ddf.columns)
print("\n■ 저장되는 칼럼:")
print(json.dumps(cols, ensure_ascii=False, indent=2))