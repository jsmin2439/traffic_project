# ┌──────────────────────────────────────────────────────────────────────────┐
# │ verify_data_loader.py                                                    │
# │                                                                          │
# │ 목적: data_loader.py에서 정의한 get_dataloaders()가 제대로 작동하는지  │
# │ 검증하는 스크립트입니다.                                                  │
# │                                                                          │
# │ 사용 방법:                                                               │
# │   cd traffic_project                                                      │
# │   python verify_data_loader.py                                            │
# │                                                                          │
# │ 파일 위치: traffic_project/verify_data_loader.py                           │
# └──────────────────────────────────────────────────────────────────────────┘

import torch
import numpy as np
from data_loader import get_dataloaders

def main():
    # ─────────────────────────────────────────────────────────────────────────
    # 1) DataLoader 생성
    # ─────────────────────────────────────────────────────────────────────────
    batch_size = 4
    print(f"▶ get_dataloaders(batch_size={batch_size}) 호출 중...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # ─────────────────────────────────────────────────────────────────────────
    # 2) 간단히 배치 하나만 꺼내서 형태 확인
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Train Loader에서 첫 배치 가져오기 ---")
    xb_train, yb_train = next(iter(train_loader))
    print(f"X_train batch shape: {xb_train.shape}  (예상: [batch_size, C_in, 12, 1370])")
    print(f"Y_train batch shape: {yb_train.shape}  (예상: [batch_size, 8, 1370])")

    print("\n--- Val Loader에서 첫 배치 가져오기 ---")
    xb_val, yb_val = next(iter(val_loader))
    print(f"X_val batch shape: {xb_val.shape}")
    print(f"Y_val batch shape: {yb_val.shape}")

    print("\n--- Test Loader에서 첫 배치 가져오기 ---")
    xb_test, yb_test = next(iter(test_loader))
    print(f"X_test batch shape: {xb_test.shape}")
    print(f"Y_test batch shape: {yb_test.shape}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3) 실제 데이터 예시 (첫 샘플) 값 확인
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 첫 번째 윈도우(인덱스=0) 예시 값 일부 출력 ---")
    # train_loader의 첫 배치, 첫 윈도우
    sample_x, sample_y = xb_train[0], yb_train[0]
    print("첫 윈도우 X (채널=0, 시간=0, 노드=0~3):", sample_x[0, 0, :4].numpy())
    # sample_x shape: (C_in, 12, 1370), 채널 0=Queue_승용
    print("첫 윈도우 Y (채널=0, 노드=0~3):", sample_y[0, :4].numpy())
    # sample_y shape: (8,1370), 채널 0=Queue_승용

    # ─────────────────────────────────────────────────────────────────────────
    # 4) 정규화값 분포 예시 (평균/표준편차) 확인
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- X (전체 Train 데이터) 채널별 평균·표준편차 예시 ---")
    # Tensor를 NumPy로 바꿔서 계산
    all_X_train = np.concatenate([batch[0].cpu().numpy() for batch in train_loader], axis=0)
    # all_X_train shape: (train_windows, C_in, 12, 1370)
    train_means = all_X_train.reshape(-1, all_X_train.shape[1], all_X_train.shape[2]*all_X_train.shape[3]).mean(axis=2)
    train_stds  = all_X_train.reshape(-1, all_X_train.shape[1], all_X_train.shape[2]*all_X_train.shape[3]).std(axis=2)
    for ch in range(all_X_train.shape[1]):
        print(f"  채널 {ch:2d}: 평균 = {train_means[:,ch].mean():.4f}, 표준편차 = {train_stds[:,ch].mean():.4f}")

    print("\n▶ 데이터 로더 검증 완료!")

if __name__ == '__main__':
    main()