#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_windows.py

1) 'normalized_tensor_{DATE}_with_weekend.npy' 파일(크기: 288×1370×9)을 로드
2) 과거 12스텝(1시간) → 다음 1스텝(5분) 슬라이딩 윈도우 (X, Y) 생성
   - X: shape = (12, 1370, 9)  ← queue+spd(8) + weekdend_onehot(1)
   - Y: shape = (1370, 8)       ← 다음 스텝의 queue+spd 8채널
3) 전체 윈도우를 NumPy 배열(all_X, all_Y)로 정리
4) 아래 검증 코드 실행
   - 배열 크기·dtype·NaN/Inf 여부 확인
   - “연속성(continuity)” 검증: X 마지막 스텝 값과 Y 값 일치 여부 확인
5) all_X.npy, all_Y.npy 파일로 저장

사용 방법:
    python make_windows.py
"""

import os
import numpy as np

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 0. 설정 및 파라미터                                                       │
# └──────────────────────────────────────────────────────────────────────────┘

# ① 처리할 날짜 리스트 (YYYYMMDD 정수 형식)
DATE_LIST = [
    20220810, 20220811, 20220812, 20220813, 20220814, 20220815,
    20220819, 20220820, 20220821, 20220822, 20220823,
    20220906, 20220907, 20220909, 20220910, 20220911,
    20220912, 20220913, 20220914, 20220915, 20220916,
    20220926, 20220928, 20220930, 20221001, 20221002,
    20221026, 20221027, 20221028, 20221030, 20221031
]

# ② 기본 디렉토리 (날짜별 폴더가 위치)
BASE_DIR = '3_tensor'

# ③ 슬라이딩 윈도우 파라미터
WINDOW_SIZE      = 12   # 과거 12스텝 (1시간)
NUM_SLOTS        = 288  # 하루 슬롯 수 (5분 단위 × 24시간)
INPUT_CHANNELS   = 9    # input 채널 수 (queue+spd=8 + weekend_flag=1)
OUTPUT_CHANNELS  = 8    # 예측 대상 채널 수 (queue+spd 8개)

# ④ 결과 저장 디렉토리 (없으면 생성)
SAVE_DIR = os.path.join(BASE_DIR, 'windows')
os.makedirs(SAVE_DIR, exist_ok=True)

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 1. 슬라이딩 윈도우 생성                                                   │
# └──────────────────────────────────────────────────────────────────────────┘

all_X = []  # List of input windows (각 원소 shape=(12,1370,15))
all_Y = []  # List of target windows (각 원소 shape=(1370,8))
all_DATE = []   # ← 윈도우별 DATE 메타정보를 담을 리스트

for DATE in DATE_LIST:
    date_str = str(DATE)
    tensor_path = os.path.join(BASE_DIR, date_str, f'normalized_tensor_{DATE}_with_weekend.npy')
    
    # 파일 유무 확인
    if not os.path.exists(tensor_path):
        print(f"⚠️ Skipping {DATE}: 파일이 없습니다 ({tensor_path})")
        continue
    
    # (1) 정규화 + 요일 정보가 붙은 텐서 로드
    #     - 파일명: normalized_tensor_{DATE}_with_weekday.npy
    #     - shape = (288, 1370, 15)
    daily = np.load(tensor_path)
    
    # (2) 텐서 shape 검증
    # (2) 텐서 shape 검증 (예상: (288,1370,9))
    if daily.shape[0] != NUM_SLOTS or daily.shape[2] != INPUT_CHANNELS:
        raise ValueError(f"[{DATE}] 텐서 shape이 예상과 다릅니다: {daily.shape} (예상: (288,1370,9))")
    
    # (3) 슬라이딩 윈도우: start = 0 ~ (288-12-1) = 275까지
    #     - X 윈도우: daily[start : start+12, :, :] → shape=(12,1370,15)
    #     - Y 타깃:  daily[start+12, :, :8]       → shape=(1370,8)
    for start in range(NUM_SLOTS - WINDOW_SIZE):
        x_win = daily[start : start + WINDOW_SIZE]              # (12,1370,15)
        y_next = daily[start + WINDOW_SIZE, :, :OUTPUT_CHANNELS] # (1370,8)
        
        all_X.append(x_win)
        all_Y.append(y_next)
        all_DATE.append(DATE)

# (4) 리스트 → NumPy 배열로 변환
#     - all_X: shape = (총 윈도우 개수, 12, 1370, 15)
#     - all_Y: shape = (총 윈도우 개수, 1370, 8)
all_X = np.stack(all_X, axis=0)
all_Y = np.stack(all_Y, axis=0)
all_DATE = np.array(all_DATE)         # shape=(총윈도우,)


# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 2. 검증 코드                                                               │
# └──────────────────────────────────────────────────────────────────────────┘

print("\n=== 슬라이딩 윈도우 생성 후 검증 ===")

# (1) 배열 크기·개수 확인
num_windows = all_X.shape[0]
print(f"▶ 총 윈도우 개수: {num_windows}")
print(f"▶ all_X shape: {all_X.shape}  (윈도우 수, 12, 1370, 15)")
print(f"▶ all_Y shape: {all_Y.shape}  (윈도우 수, 1370, 8)")

# (2) dtype 검사
print("▶ all_X dtype:", all_X.dtype)
print("▶ all_Y dtype:", all_Y.dtype)
assert all_X.dtype in [np.float32, np.float64], "all_X는 실수형이어야 합니다."
assert all_Y.dtype in [np.float32, np.float64], "all_Y는 실수형이어야 합니다."

# (3) NaN/Inf 검사
nan_X = np.isnan(all_X).any()
inf_X = np.isinf(all_X).any()
nan_Y = np.isnan(all_Y).any()
inf_Y = np.isinf(all_Y).any()
print("▶ all_X에 NaN?", nan_X, "  Inf?", inf_X)
print("▶ all_Y에 NaN?", nan_Y, "  Inf?", inf_Y)
assert not nan_X and not inf_X, "all_X에 NaN 또는 Inf가 존재합니다."
assert not nan_Y and not inf_Y, "all_Y에 NaN 또는 Inf가 존재합니다."

# (4) 연속성 검증 (첫 번째 윈도우 기준)
print("\n▶ 첫 번째 윈도우 예시 검증:")
print(" - X[0] shape:", all_X[0].shape)  # (12,1370,15)
print(" - Y[0] shape:", all_Y[0].shape)  # (1370,8)

# 임의로 노드 인덱스 0으로 샘플링
node_idx = 0

# X의 마지막 스텝(t=11) 채널 0~7 vs Y의 채널 0~7 비교
x_last = all_X[0, -1, node_idx, :OUTPUT_CHANNELS]  # (8,)
y_true = all_Y[0, node_idx, :]                       # (8,)

print(" - X 마지막 스텝 채널0~7:", np.round(x_last, 4))
print(" - Y 실제 값 채널0~7:   ", np.round(y_true, 4))
assert np.allclose(x_last, y_true, atol=1e-5), "연속성 검증 실패: X 마지막 스텝과 Y가 일치하지 않습니다."

print("\n✅ 슬라이딩 윈도우 (X, Y) 생성 및 검증 완료 ✓")


# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 3. 결과물 저장                                                             │
# └──────────────────────────────────────────────────────────────────────────┘

# (1) NumPy 파일로 저장
np.save(os.path.join(SAVE_DIR, 'all_X.npy'), all_X)
np.save(os.path.join(SAVE_DIR, 'all_Y.npy'), all_Y)
np.save(os.path.join(SAVE_DIR, 'all_DATE.npy'), all_DATE)  # ← DATE 배열도 저장

print(f"\n▶ all_X.npy, all_Y.npy, all_DATE.npy 파일이 '{SAVE_DIR}' 폴더에 저장되었습니다.")