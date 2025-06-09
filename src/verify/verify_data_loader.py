#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_data_loader.py

목적: data_loader.py에서 정의한 get_dataloaders()가 제대로 작동하는지 검증
사용법:
  cd traffic_project
  python verify_data_loader.py
"""

import torch
import numpy as np
from data_loader import get_dataloaders

def main():
    batch_size = 4
    print(f"▶ get_dataloaders(batch_size={batch_size}) 호출 중...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # --- Train Loader에서 첫 배치 가져오기 ---
    xb_train, yb_train, train_idx, train_date = next(iter(train_loader))
    print(f"X_train batch shape: {xb_train.shape}  (예상: [{batch_size}, C_in, 12, 1370])")
    print(f"Y_train batch shape: {yb_train.shape}  (예상: [{batch_size}, 8, 1370])")

    # --- Val Loader에서 첫 배치 가져오기 ---
    xb_val, yb_val, val_idx, val_date = next(iter(val_loader))
    print(f"X_val batch shape: {xb_val.shape}")
    print(f"Y_val batch shape: {yb_val.shape}")

    # --- Test Loader에서 첫 배치 가져오기 ---
    xb_test, yb_test, test_idx, test_date = next(iter(test_loader))
    print(f"X_test batch shape: {xb_test.shape}")
    print(f"Y_test batch shape: {yb_test.shape}")

    # --- 첫 번째 윈도우 예시 값 일부 출력 ---
    sample_x, sample_y = xb_train[0], yb_train[0]
    print("첫 윈도우 X (채널=0, 시간=0, 노드=0~3):", sample_x[0, 0, :4].numpy())
    print("첫 윈도우 Y (채널=0, 노드=0~3):", sample_y[0, :4].numpy())

    # --- 정규화값 분포 예시 (평균/표준편차) 확인 ---
    all_X_train = np.concatenate([batch[0].cpu().numpy() for batch in train_loader], axis=0)
    num_samples, C, T, N = all_X_train.shape
    flat = all_X_train.reshape(num_samples, C, T * N)
    # flat.shape == (num_samples, C, 12*1370)

    train_means = flat.mean(axis=2)   # (num_samples, C)
    train_stds  = flat.std(axis=2)    # (num_samples, C)

    for ch in range(all_X_train.shape[1]):
        print(f"  채널 {ch:2d}: 평균 = {train_means[:,ch].mean():.4f}, 표준편차 = {train_stds[:,ch].mean():.4f}")

    print("\n▶ 데이터 로더 검증 완료!")

if __name__ == '__main__':
    main()