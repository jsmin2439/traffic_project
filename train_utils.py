# ┌──────────────────────────────────────────────────────────────────────────┐
# │ train_utils.py                                                            │
# │                                                                          │
# │ 학습 스크립트에서 공통으로 사용하는 유틸 함수/클래스 모음                     │
# └──────────────────────────────────────────────────────────────────────────┘

import os
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

class EarlyStopping:
    """
    Validation Loss가 'patience' 에폭 동안 개선되지 않으면 학습을 조기 중단합니다.
    """
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        """
        Args:
            patience (int): 개선이 없다고 판단할 최대 Epoch 수
            min_delta (float): '개선'으로 볼 최소 Loss 감소 폭
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.num_bad_epochs = 0
        self.stop = False

    def step(self, val_loss: float):
        """
        매 validation마다 콜하여 내부 상태를 업데이트합니다.

        Args:
            val_loss (float): 현재 epoch의 validation loss
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return

        # 이전 best_loss보다 'val_loss'가 충분히 감소했으면 갱신
        if (self.best_loss - val_loss) > self.min_delta:
            self.best_loss = val_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.stop = True

    def reset(self):
        """ 상태를 초기화할 때 호출 """
        self.best_loss = None
        self.num_bad_epochs = 0
        self.stop = False


def ensure_dir(path: str):
    """
    경로가 존재하지 않으면 디렉토리를 생성합니다.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    """
    모델의 학습 가능 파라미터 수를 반환합니다.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_memory_usage(device: torch.device):
    """
    CUDA 혹은 MPS 사용 시, 현재 GPU 메모리 사용량을 출력합니다.
    """
    if device.type == 'cuda':
        used = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        print(f"   ▶ GPU Memory - Used: {used:.2f} GB | Reserved: {reserved:.2f} GB")
    elif device.type == 'mps':
        # MPS는 정확한 메모리 사용량 확인이 어렵기 때문에 간단히 안내만
        print("   ▶ MPS backend 사용 중: 메모리 사용량은 자동 관리됩니다.")
    else:
        print("   ▶ CPU 모드: GPU 메모리 체크 불필요")