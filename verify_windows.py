#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_windows.py

‘make_windows.py’ 실행 후 저장된 all_X.npy, all_Y.npy 파일이
정상적으로 저장되었는지 확인하는 스크립트입니다.

검증 내용:
1) 파일이 존재하는지
2) 배열 shape, dtype, NaN/Inf 유무
3) 저장된 값이 일관된지 (‘연속성’ 검증)
"""

import os
import numpy as np

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 0. 설정                                                                   │
# └──────────────────────────────────────────────────────────────────────────┘

# 저장된 윈도우 디렉토리(‘make_windows.py’에서 지정한 경로와 동일)
SAVE_DIR = '3_tensor/windows'
X_PATH   = os.path.join(SAVE_DIR, 'all_X.npy')
Y_PATH   = os.path.join(SAVE_DIR, 'all_Y.npy')

# 슬라이딩 윈도우 파라미터 (make_windows.py와 동일)
WINDOW_SIZE     = 12     # 과거 12스텝
NUM_SLOTS       = 288    # 하루 슬롯 수
INPUT_CHANNELS  = 9     # 입력 채널(9)
OUTPUT_CHANNELS = 8      # 예측 채널(8)

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 1. 파일 존재 확인                                                         │
# └──────────────────────────────────────────────────────────────────────────┘

print("\n=== 1) 파일 존재 여부 확인 ===")

if not os.path.exists(X_PATH):
    raise FileNotFoundError(f"all_X.npy 파일이 없습니다: {X_PATH}")
else:
    print(f"✔ all_X.npy 파일이 존재합니다 ({X_PATH})")

if not os.path.exists(Y_PATH):
    raise FileNotFoundError(f"all_Y.npy 파일이 없습니다: {Y_PATH}")
else:
    print(f"✔ all_Y.npy 파일이 존재합니다 ({Y_PATH})")

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 2. 배열 로드 및 기본 검증                                                  │
# └──────────────────────────────────────────────────────────────────────────┘

print("\n=== 2) 배열 로드 및 기본 검증 ===")

all_X = np.load(X_PATH)
all_Y = np.load(Y_PATH)

# (1) shape 확인
print(f"▶ all_X shape: {all_X.shape}")
print(f"▶ all_Y shape: {all_Y.shape}")

# 창(윈도우) 수 유도
num_windows_X = all_X.shape[0]
num_windows_Y = all_Y.shape[0]

assert num_windows_X == num_windows_Y, \
       f"윈도우 개수가 일치하지 않습니다: all_X={num_windows_X}, all_Y={num_windows_Y}"

print(f"▶ 윈도우 개수 (num_windows): {num_windows_X}")

# (2) dtype 확인
print(f"▶ all_X dtype: {all_X.dtype}")
print(f"▶ all_Y dtype: {all_Y.dtype}")

assert all_X.dtype in [np.float32, np.float64], "all_X는 실수형이어야 합니다."
assert all_Y.dtype in [np.float32, np.float64], "all_Y는 실수형이어야 합니다."

# (3) NaN/Inf 확인
nan_X = np.isnan(all_X).any()
inf_X = np.isinf(all_X).any()
nan_Y = np.isnan(all_Y).any()
inf_Y = np.isinf(all_Y).any()

print(f"▶ all_X에 NaN? {nan_X}    Inf? {inf_X}")
print(f"▶ all_Y에 NaN? {nan_Y}    Inf? {inf_Y}")

assert not nan_X and not inf_X, "all_X에 NaN 또는 Inf가 존재합니다."
assert not nan_Y and not inf_Y, "all_Y에 NaN 또는 Inf가 존재합니다."

# (4) 배열 차원 확인 (첫 번째 윈도우 기준)
print("\n▶ 첫 번째 윈도우 차원 검증:")
x0 = all_X[0]  # shape=(12,1370,9)
y0 = all_Y[0]  # shape=(1370,8)

print(f"  - X[0] shape: {x0.shape}  (예상: (12, 1370, 9))")
print(f"  - Y[0] shape: {y0.shape}  (예상: (1370, 8))")

assert x0.ndim == 3 and x0.shape[0] == WINDOW_SIZE and x0.shape[2] == INPUT_CHANNELS, \
       f"X[0] shape 오류: {x0.shape}"
assert y0.ndim == 2 and y0.shape[1] == OUTPUT_CHANNELS, \
       f"Y[0] shape 오류: {y0.shape}"

print("▶ 배열 차원 검증 통과 ✓")

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 3. 연속성(Continuity) 검증                                                  │
# └──────────────────────────────────────────────────────────────────────────┘

print("\n=== 3) 연속성 검증 (첫 번째 윈도우) ===")

# 노드 인덱스 예시: 0
node_idx = 0

# X 마지막 스텝(t=11) 채널 0~7
x_last = x0[-1, node_idx, :OUTPUT_CHANNELS]  # shape=(8,)
# Y 실제 값 채널 0~7
y_true = y0[node_idx, :]                     # shape=(8,)

print("▶ X 마지막 스텝 채널0~7:", np.round(x_last, 4))
print("▶ Y 실제 값 채널0~7:   ", np.round(y_true, 4))

assert np.allclose(x_last, y_true, atol=1e-5), \
       "연속성 검증 실패: X 마지막 스텝과 Y가 일치하지 않습니다."

print("▶ 연속성 검증 통과 ✓")

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 4. 샘플 값 예시 출력                                                        │
# └──────────────────────────────────────────────────────────────────────────┘

print("\n=== 4) 샘플 값 예시 출력 ===")

# (1) X[0]의 첫 슬롯(slot=0), 첫 노드(node=0) 채널 15개
print("▶ X[0][0,0,:] (첫 윈도우, 첫 슬롯, 첫 노드, 15채널):\n", np.round(x0[0, 0, :], 4))

# (2) Y[0]의 첫 노드(node=0) 채널 8개
print("\n▶ Y[0][0,:] (첫 윈도우, 첫 노드, 8채널):\n", np.round(y0[0, :], 4))

# (3) X[0]의 슬롯 t=5, 임의 노드 node=10, 채널 0~7 (queue+spd)
slot_t = 5
node_j = 10
print(f"\n▶ X[0][{slot_t},{node_j},:8] (첫 윈도우, 슬롯={slot_t}, 노드={node_j}, 8채널):\n",
      np.round(x0[slot_t, node_j, :OUTPUT_CHANNELS], 4))

# (4) X[0]의 일부 요일원핫(채널 8~14)
print(f"\n▶ X[0][{slot_t},{node_j},8:15] (요일 원-핫 채널):\n",
      x0[slot_t, node_j, OUTPUT_CHANNELS:])

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 5. 저장된 배열 다시 불러와 비교 (일관성 확인)                                 │
# └──────────────────────────────────────────────────────────────────────────┘

print("\n=== 5) 저장된 배열 재로드 및 일관성 확인 ===")

reloaded_X = np.load(X_PATH)
reloaded_Y = np.load(Y_PATH)

# (1) all_X와 reloaded_X가 동일한지
equal_X = np.array_equal(all_X, reloaded_X)
equal_Y = np.array_equal(all_Y, reloaded_Y)

print(f"▶ all_X vs reloaded_X identical? {equal_X}")
print(f"▶ all_Y vs reloaded_Y identical? {equal_Y}")

assert equal_X, "all_X.npy를 다시 불러온 배열이 원본과 일치하지 않습니다."
assert equal_Y, "all_Y.npy를 다시 불러온 배열이 원본과 일치하지 않습니다."

print("\n✅ 저장된 배열 일관성 확인 완료 ✓")