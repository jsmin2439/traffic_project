#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tensor_with_weekend.py

input_tensor 및 normalized_tensor(.npy)에 “주말/주중(weekday vs weekend)” 플래그 채널을 추가하여
(288,1370,8) → (288,1370,9)로 확장하는 스크립트입니다.
"""

import numpy as np
import pandas as pd
import os
import pickle

# 처리할 날짜 리스트
DATE_LIST = [
    20220810,20220811,20220812,20220813,20220814,20220815,
    20220819,20220820,20220821,20220822,20220823,
    20220906,20220907,20220909,20220910,20220911,
    20220912,20220913,20220914,20220915,20220916,
    20220926,20220928,20220930,20221001,20221002,
    20221026,20221027,20221028,20221030,20221031
]
BASE_DIR = '3_tensor'
FIELDS = ['input_tensor', 'normalized_tensor']

for DATE in DATE_LIST:
    date_str = str(DATE)
    date_dir = os.path.join(BASE_DIR, date_str)
    # 원본 .npy 경로
    input_path = os.path.join(date_dir, f'input_tensor_{DATE}.npy')
    norm_path  = os.path.join(date_dir, f'normalized_tensor_{DATE}.npy')
    
    if not os.path.exists(input_path) or not os.path.exists(norm_path):
        print(f"⚠️ Skipping {DATE}: 파일 누락")
        continue

    # 1) 텐서 로드
    input_tensor      = np.load(input_path)   # (288,1370,8)
    normalized_tensor = np.load(norm_path)    # (288,1370,8)

    # 2) DATE의 요일 계산
    ts = pd.to_datetime(str(DATE), format='%Y%m%d')
    weekday_idx = ts.weekday()  # 0=월,1=화,...,5=토,6=일

    # ──────────────────────────────────────────────────────────────────────────
    # [변경된 부분 시작] “7차원 원-핫” 대신 “주말 여부 플래그” 하나만 만들기
    # ──────────────────────────────────────────────────────────────────────────
    # if weekday_idx in [5, 6] → weekend_flag=1.0, else weekday_flag=0.0
    weekend_flag_value = 1.0 if (weekday_idx >= 5) else 0.0

    # (2-1) 슬롯(288) 축으로 브로드캐스트 → (288,)
    num_slots = input_tensor.shape[0]  # 288
    weekend_slots = np.full((num_slots,), weekend_flag_value, dtype=np.float32)  # (288,)

    # (2-2) 노드(1370) 축으로 브로드캐스트 → (288,1370,1)
    num_lanes = input_tensor.shape[1]  # 1370
    weekend_feat = weekend_slots.reshape(num_slots, 1, 1)                # (288,1,1)
    weekend_feat = np.broadcast_to(weekend_feat, (num_slots, num_lanes, 1))  # (288,1370,1)
    # ──────────────────────────────────────────────────────────────────────────
    # [변경된 부분 끝]
    # ──────────────────────────────────────────────────────────────────────────

    # 3) 기존 텐서 뒤에 붙이기 → 채널 8 → 9로 확장
    #    - aug_input: (288,1370,8) + (288,1370,1) → (288,1370,9)
    aug_input = np.concatenate([input_tensor,      weekend_feat], axis=2)
    aug_norm  = np.concatenate([normalized_tensor, weekend_feat], axis=2)

    # 4) 저장 경로 설정
    out_input_path = os.path.join(date_dir, f'input_tensor_{DATE}_with_weekend.npy')
    out_norm_path  = os.path.join(date_dir, f'normalized_tensor_{DATE}_with_weekend.npy')
    
    # 5) .npy로 저장
    np.save(out_input_path, aug_input)
    np.save(out_norm_path, aug_norm)
    
    # 6) (선택) Pickle 형식으로도 저장
    with open(os.path.join(date_dir, f'input_tensor_{DATE}_with_weekend.pkl'), 'wb') as f:
        pickle.dump(aug_input, f)
    with open(os.path.join(date_dir, f'normalized_tensor_{DATE}_with_weekend.pkl'), 'wb') as f:
        pickle.dump(aug_norm, f)

    print(f"● {DATE} → shapes: aug_input {aug_input.shape}, aug_norm {aug_norm.shape}")

print("▶ 모든 날짜 처리 완료")