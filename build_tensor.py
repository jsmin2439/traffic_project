#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_tensors.py

여러 날짜에 대해 입력 텐서(input_tensor)와 정규화된 텐서(normalized_tensor)를
자동으로 생성하여 '3_tensor/' 폴더에 저장하는 스크립트입니다.

사용법:
    python generate_tensors.py

필요 경로:
    1) Parquet 원본 데이터: '1_lake/od.parquet'
    2) 전체 lane ID 목록: '3_tensor/adjacency/lanes.txt'
    3) 결과 저장 폴더: '3_tensor/tensors/' (없으면 자동 생성)

출력 파일 예시:
    - 3_tensor/tensors/input_tensor_20220810.npy
    - 3_tensor/tensors/input_tensor_20220810.pkl
    - 3_tensor/tensors/normalized_tensor_20220810.npy
    - 3_tensor/tensors/normalized_tensor_20220810.pkl
"""

import os
import pandas as pd
import numpy as np
import pickle
import warnings

# 경고 메시지 감추기
warnings.filterwarnings('ignore')

# 1) 대상 날짜 리스트
DATE_LIST = [
    20220810, 20220811, 20220812, 20220813, 20220814, 20220815,
    20220819, 20220820, 20220821, 20220822, 20220823,
    20220906, 20220907, 20220909, 20220910, 20220911,
    20220912, 20220913, 20220914, 20220915, 20220916,
    20220926, 20220928, 20220930, 20221001, 20221002,
    20221026, 20221027, 20221028, 20221030, 20221031
]

# 2) 경로 설정
OD_PARQUET_PATH = '1_lake/od.parquet'
LANE_IDS_PATH   = '3_tensor/adjacency/lanes.txt'
OUTPUT_DIR      = '3_tensor/tensors/'
# # 출력 폴더가 없으면 생성
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3) lane_ids 로드 (한 줄당 하나의 lane_id)
with open(LANE_IDS_PATH, 'r', encoding='utf-8') as f:
    lane_ids = [line.strip() for line in f.readlines()]

NUM_LANES      = len(lane_ids)   # 보통 1370
NUM_SLOTS      = 288             # 하루치 슬롯(5분 단위 × 24시간)
NUM_CATEGORIES = 4               # 승용, 버스, 트럭, 오토바이
CHANNELS       = NUM_CATEGORIES * 2  # queue+spd → 8

# 4) 차량 종류(vhcl_typ) → 카테고리 맵핑
VHC_TYP_MAPPING = {
    1: 0,  # 승용차 → 0
    2: 1,  # 버스   → 1
    3: 2,  # 트럭   → 2
    4: 2,  # 대형트럭→ 트럭(2)
    5: None,  # 특수차량 → 제거
    6: 3   # 오토바이 → 3
}

def process_date(date: int):
    date_dir = os.path.join('3_tensor', str(date))
    os.makedirs(date_dir, exist_ok=True)
    """
    주어진 날짜(date, YYYYMMDD)별로:
      1) Parquet에서 해당 날짜 데이터 로드
      2) 전처리 → 입력 텐서(input_tensor) 생성
      3) Z-Score 정규화 → normalized_tensor 생성
      4) input_tensor 및 normalized_tensor를 .npy/.pkl로 저장
    """
    print(f"\n========== Processing Date: {date} ==========")
    # ──────────────────────────────────────────────────────────────────────────
    # 1) Parquet에서 해당 날짜 데이터 로드
    # ──────────────────────────────────────────────────────────────────────────
    print("1) Reading Parquet for date =", date)
    try:
        od_df = pd.read_parquet(
            OD_PARQUET_PATH,
            engine='pyarrow',
            filters=[('date', '=', date)]
        )
    except Exception as e:
        print(f"  ▶ Error: {date} 관련 Parquet 로드 실패: {e}")
        return

    # 검증: date 컬럼이 모두 동일한지 확인
    unique_dates = od_df['date'].astype(int).unique()
    if not (len(unique_dates) == 1 and unique_dates[0] == date):
        print(f"  ▶ Warning: date 필터링 결과가 예상과 다릅니다: {unique_dates}")
    print("  ▶ od_df shape:", od_df.shape)

    # ──────────────────────────────────────────────────────────────────────────
    # 2) 한국시간(dt_kst) 생성 및 원본 dt 열 삭제
    # ──────────────────────────────────────────────────────────────────────────
    print("2) Generating dt_kst (KST)")
    od_df['dt_utc'] = pd.to_datetime(od_df['unix_time'], unit='s', utc=True)
    od_df['dt_kst'] = od_df['dt_utc'].dt.tz_convert('Asia/Seoul')
    if 'dt' in od_df.columns:
        od_df.drop(columns=['dt'], inplace=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 3) 차량 종류(vhcl_typ) 재분류 및 특수차량(5) 제거
    # ──────────────────────────────────────────────────────────────────────────
    print("3) Mapping vhcl_typ → vhcl_cat and dropping special vehicles")
    od_df = od_df[od_df['vhcl_typ'] != 5].copy()  # 특수차량 제거
    od_df['vhcl_cat'] = od_df['vhcl_typ'].map(VHC_TYP_MAPPING)
    od_df.dropna(subset=['vhcl_cat'], inplace=True)
    od_df['vhcl_cat'] = od_df['vhcl_cat'].astype(int)

    # ──────────────────────────────────────────────────────────────────────────
    # 4) 5분 단위 슬롯(slot) 생성: 0 ~ 287
    # ──────────────────────────────────────────────────────────────────────────
    print("4) Generating 5-minute slot")
    od_df['slot'] = (
        (od_df['dt_kst'].dt.hour * 60 + od_df['dt_kst'].dt.minute) // 5
    ).astype(int)

    # ──────────────────────────────────────────────────────────────────────────
    # 5) Queue Count 계산 (que_all == 1인 경우)
    # ──────────────────────────────────────────────────────────────────────────
    print("5) Calculating queue_count")
    queue_df = od_df[od_df['que_all'] == 1]
    queue_count = (
        queue_df
        .groupby(['slot', 'lane_id', 'vhcl_cat'])['vhcl_id']
        .nunique()
        .reset_index(name='queue_count')
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 6) 평균 속도(avg_spd) 계산 (spd > 0인 경우)
    # ──────────────────────────────────────────────────────────────────────────
    print("6) Calculating avg_spd")
    moving_df = od_df[od_df['spd'] > 0]
    spd_avg = (
        moving_df
        .groupby(['slot', 'lane_id', 'vhcl_cat'])['spd']
        .mean()
        .reset_index(name='avg_spd')
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 7) 입력 텐서(input_tensor) 초기화
    # ──────────────────────────────────────────────────────────────────────────
    print("7) Initializing input_tensor array")
    input_tensor = np.zeros(
        (NUM_SLOTS, NUM_LANES, CHANNELS),
        dtype=np.float32
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 8) queue_count 데이터를 input_tensor 채널(0~3)에 배치
    # ──────────────────────────────────────────────────────────────────────────
    print("8) Filling queue channels (0~3)")
    for _, row in queue_count.iterrows():
        slot = int(row['slot'])
        lane = row['lane_id']
        cat = int(row['vhcl_cat'])
        cnt = int(row['queue_count'])
        if lane in lane_ids:
            lane_idx = lane_ids.index(lane)
            input_tensor[slot, lane_idx, cat] = cnt

    # ──────────────────────────────────────────────────────────────────────────
    # 9) avg_spd 데이터를 input_tensor 채널(4~7)에 배치
    # ──────────────────────────────────────────────────────────────────────────
    print("9) Filling speed channels (4~7)")
    for _, row in spd_avg.iterrows():
        slot = int(row['slot'])
        lane = row['lane_id']
        cat = int(row['vhcl_cat'])
        spd_val = float(row['avg_spd'])
        if lane in lane_ids:
            lane_idx = lane_ids.index(lane)
            input_tensor[slot, lane_idx, NUM_CATEGORIES + cat] = spd_val

    # ──────────────────────────────────────────────────────────────────────────
    # 10) input_tensor NaN/음수/Shape 검증
    # ──────────────────────────────────────────────────────────────────────────
    print("10) Validating input_tensor")
    if np.isnan(input_tensor).any():
        raise ValueError(f"[{date}] input_tensor에 NaN이 존재합니다.")
    if (input_tensor < 0).any():
        raise ValueError(f"[{date}] input_tensor에 음수 값이 존재합니다.")
    if input_tensor.shape != (NUM_SLOTS, NUM_LANES, CHANNELS):
        raise ValueError(f"[{date}] input_tensor shape 오류: 현재 {input_tensor.shape}")

    print("    ✔ input_tensor validation passed")

    # ──────────────────────────────────────────────────────────────────────────
    # 11) input_tensor 저장 (.npy, .pkl)
    # ──────────────────────────────────────────────────────────────────────────
    print("11) Saving input_tensor")
    input_npy_path = os.path.join(date_dir, f'input_tensor_{date}.npy')
    input_pkl_path = os.path.join(date_dir, f'input_tensor_{date}.pkl')
    np.save(input_npy_path, input_tensor)
    with open(input_pkl_path, 'wb') as f:
        pickle.dump(input_tensor, f)
    print(f"    ▶ Saved: {input_npy_path}, {input_pkl_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # 12) 채널별 통계(평균·표준편차) 계산
    # ──────────────────────────────────────────────────────────────────────────
    print("12) Calculating channel-wise mean & std for normalization")
    means = []
    stds  = []
    for ch in range(CHANNELS):
        vals = input_tensor[:, :, ch].ravel()
        mu    = np.mean(vals)
        sigma = np.std(vals)
        means.append(mu)
        stds.append(sigma)

    means = np.array(means, dtype=np.float32)
    stds  = np.array(stds, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # 13) Z-Score 정규화 수행 → normalized_tensor 생성
    # ──────────────────────────────────────────────────────────────────────────
    print("13) Performing Z-Score normalization")
    normalized_tensor = np.zeros_like(input_tensor, dtype=np.float32)
    for ch in range(CHANNELS):
        mu    = means[ch]
        sigma = stds[ch]
        if sigma == 0:
            normalized_tensor[:, :, ch] = 0.0
        else:
            normalized_tensor[:, :, ch] = (input_tensor[:, :, ch] - mu) / sigma

    # 정규화 후 검증 (평균 0, 표준편차 ≈ 1)
    for ch in range(CHANNELS):
        vals_norm = normalized_tensor[:, :, ch].ravel()
        mean_ch   = np.mean(vals_norm)
        std_ch    = np.std(vals_norm)
        if not np.isclose(mean_ch, 0.0, atol=1e-4) or not np.isclose(std_ch, 1.0, atol=1e-4):
            print(f"    ▶ Warning: 채널 {ch} 정규화 통계 이상 (mean={mean_ch:.4f}, std={std_ch:.4f})")

    # ──────────────────────────────────────────────────────────────────────────
    # 14) normalized_tensor 저장 (.npy, .pkl)
    # ──────────────────────────────────────────────────────────────────────────
    print("14) Saving normalized_tensor")
    norm_npy_path = os.path.join(date_dir, f'normalized_tensor_{date}.npy')
    norm_pkl_path = os.path.join(date_dir, f'normalized_tensor_{date}.pkl')
    np.save(norm_npy_path, normalized_tensor)
    with open(norm_pkl_path, 'wb') as f:
        pickle.dump(normalized_tensor, f)
    print(f"    ▶ Saved: {norm_npy_path}, {norm_pkl_path}")

    print(f"========== Completed Date: {date} ==========\n")


if __name__ == '__main__':
    # 모든 날짜에 대해 순차 처리
    for dt in DATE_LIST:
        process_date(dt)

    print("\nAll dates processed. Tensors saved in per-date folders under 3_tensor/")