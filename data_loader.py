# ┌──────────────────────────────────────────────────────────────────────────┐
# │ data_loader.py                                                           │
# │                                                                          │
# │ 목적: 전통적 LSTM, 원본 ST‐GCN, T‐GCN, DCRNN, 제안 모델(Residual 포함) 등 │
# │ 모든 모델이 동일한 방식으로 데이터를 읽어올 수 있도록                   │
# │ Dataset 클래스 및 DataLoader 생성 코드만 모아 두었습니다.               │
# │                                                                          │
# │ 사용 방법:                                                               │
# │   from data_loader import get_dataloaders                                  │
# │   train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)   │
# │                                                                          │
# │ 파일 위치: traffic_project/data_loader.py                                 │
# └──────────────────────────────────────────────────────────────────────────┘

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional

class TrafficDataset(Dataset):
    """
    ────────────────────────────────────────────────────────────────────────────
    PyTorch Dataset: TrafficWindowDataset 기능을 일반화한 버전입니다.
    각 (X, Y) 윈도우를 불러와, 모델에 넘길 수 있는 형태로 변환합니다.

    - X_raw shape: (윈도우 개수, 12, 1370, 채널수) = (M, 12, N, C)
      → PyTorch 모델 입력 형식: (채널수, 시간, 노드) = (C, 12, N)
    - Y_raw shape: (윈도우 개수, 1370, 8) = (M, N, 8)
      → PyTorch 모델 타깃 형식: (출력 채널수, 노드) = (8, 1370)
    ────────────────────────────────────────────────────────────────────────────
    """
    def __init__(self, X_np: np.ndarray, Y_np: np.ndarray):
        """
        Args:
            X_np (np.ndarray): NumPy 배열, shape = (M, 12, 1370, C_in)
            Y_np (np.ndarray): NumPy 배열, shape = (M, 1370, 8)
        """
        super().__init__()
        assert isinstance(X_np, np.ndarray) and isinstance(Y_np, np.ndarray), \
            "X_np와 Y_np는 numpy.ndarray 여야 합니다."
        assert X_np.shape[0] == Y_np.shape[0], "X와 Y의 윈도우 수(M)가 같아야 합니다."

        self.X = torch.from_numpy(X_np).float()  # (M, 12, 1370, C_in)
        self.Y = torch.from_numpy(Y_np).float()  # (M, 1370, 8)

        # 한 번만 체크
        M, T, N, C = self.X.shape
        assert T == 12, f"X의 시간축 길이(T)가 12여야 합니다. 현재: {T}"
        assert N == 1370, f"X의 노드 개수(N)가 1370이어야 합니다. 현재: {N}"
        assert C in [8, 9, 15], f"X 입력 채널 수(C_in)가 {8,9,15} 중 하나여야 합니다. 현재: {C}"

        # Y 채널은 반드시 8개 (queue4 + speed4) 여야 함
        assert self.Y.shape[1:] == (1370, 8), \
            f"Y의 shape가 (1370,8)이어야 합니다. 현재: {self.Y.shape}"

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        """
        Returns:
            x_tensor (torch.Tensor): shape = (C_in, 12, 1370)
            y_tensor (torch.Tensor): shape = (8, 1370)
            idx (int): 원본 윈도우 인덱스
        """
        x = self.X[idx]       # (12,1370,C_in)
        y = self.Y[idx]       # (1370,8)

        # [주의] 모델별로 입력 형태가 다를 수 있습니다.
        #  - 전통적 LSTM: (B, T, N*C) 혹은 (B, N*C, T) → 내부에서 reshape
        #  - ST-GCN: (B, C, T, N) → 직접 (C,12,1370)
        # LSTM 및 GRU용 모델: reshape/permute 로 추가 처리
        # 여기서는 “모든 모델 공통”으로 (C,T,N) 형태로 리턴

        # (12,1370,C_in) → (C_in,12,1370)
        x_tensor = x.permute(2, 0, 1).contiguous()
        # (1370,8) → (8,1370)
        y_tensor = y.permute(1, 0).contiguous()

        return x_tensor, y_tensor, idx

def get_dataloaders(batch_size: int = 4, num_workers: Optional[int] = None, pin_memory: bool = True):
    """
    ────────────────────────────────────────────────────────────────────────────
    모든 모델(전통적 LSTM, ST-GCN, T-GCN, DCRNN, ResLSTM 등)이 공통으로 사용하는
    DataLoader를 생성해 주는 함수입니다.

    Args:
        batch_size (int): Mini-batch 크기. 모델 학습 시 batch_size=4 정도 권장.
        num_workers (int): DataLoader에 사용할 CPU 코어 수.
                           None인 경우, os.cpu_count()//2 혹은 1로 자동 설정.
        pin_memory (bool): True면, GPU(pin_memory=True)용 고속 전송을 위해 사용.

    Returns:
        train_loader, val_loader, test_loader (tuple of DataLoader)
    ────────────────────────────────────────────────────────────────────────────
    """
    # ─────────────────────────────────────────────────────────────────────────
    # 0) numpy 파일 경로 설정 (프로젝트 루트 기준)
    # ─────────────────────────────────────────────────────────────────────────
    # 1) 윈도우 데이터: all_X.npy (shape=(8556,12,1370,C)) 
    #                   all_Y.npy (shape=(8556,1370,8))
    data_windows_dir = os.path.join('3_tensor', 'windows')
    x_path = os.path.join(data_windows_dir, 'all_X.npy')
    y_path = os.path.join(data_windows_dir, 'all_Y.npy')

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"all_X.npy 또는 all_Y.npy 파일을 찾을 수 없습니다:\n"
                                f"  Expected: {x_path}\n"
                                f"  Expected: {y_path}")

    # 2) NumPy로 로드
    X_all = np.load(x_path)  # (8556,12,1370,C_in)
    Y_all = np.load(y_path)  # (8556,1370,8)

    # ─────────────────────────────────────────────────────────────────────────
    # 3) Train/Val/Test 분할 (70% / 15% / 15%)
    # ─────────────────────────────────────────────────────────────────────────
    N = X_all.shape[0]
    train_end = int(N * 0.7)
    val_end   = int(N * 0.85)

    X_train, Y_train = X_all[:train_end],   Y_all[:train_end]
    X_val,   Y_val   = X_all[train_end:val_end], Y_all[train_end:val_end]
    X_test,  Y_test  = X_all[val_end:],     Y_all[val_end:]

    # ─────────────────────────────────────────────────────────────────────────
    # 4) Dataset 및 DataLoader 생성
    # ─────────────────────────────────────────────────────────────────────────
    train_ds = TrafficDataset(X_train, Y_train)
    val_ds   = TrafficDataset(X_val,   Y_val)
    test_ds  = TrafficDataset(X_test,  Y_test)

    # num_workers 설정
    if num_workers is None:
        cpu_cnt = os.cpu_count() or 1
        num_workers = min(4, cpu_cnt // 2)

    # prefetch_factor (파이프라인 성능 관련)
    prefetch = 4 if batch_size < 8 else 2

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=prefetch,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=prefetch,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=prefetch,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 스크립트를 직접 실행했을 때, 간단한 확인 메시지 출력
    print("data_loader.py가 성공적으로 로드되었습니다.")
    print("get_dataloaders() 함수를 통해 train/val/test DataLoader를 생성하세요.")