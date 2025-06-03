# ┌──────────────────────────────────────────────────────────────────────────┐
# │ model/stgcn_model.py                                                      │
# │                                                                          │
# │ 원본 ST‐GCN 모델 클래스 정의                                              │
# │                                                                          │
# │ - 입력: (B, C_in, T, N)                                                   │
# │ - ST‐GCN Layer 2개                                                           │
# │ - 마지막 1×1 시공간 컨볼루션으로 시간 차원을 1로 축소 후, (B, C_out, N) 반환   │
# └──────────────────────────────────────────────────────────────────────────┘

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    """
    기본 그래프 컨볼루션 (k=1 Chebyshev 대체):
    - 입력: x (B, C, T, N)
    - 인접행렬 A (N, N)
    연산:
      1) x → (B, T, N, C)
      2) A @ x_permuted → (B, T, N, C)
      3) 다시 (B, C, T, N)
      4) 1×1 Conv (theta) → (B, C_out, T, N)
    """
    def __init__(self, in_c: int, out_c: int, A: torch.Tensor):
        super().__init__()
        self.A = A            # (N, N) 인접행렬
        self.conv1x1 = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.conv1x1.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, N)
        B, C, T, N = x.shape

        # 1) (B, C, T, N) → (B, T, N, C)
        x_perm = x.permute(0, 2, 3, 1).contiguous()

        # 2) 그래프 컨볼루션: A @ x_perm (행렬 곱: (N,N) × (B,T,N,C) → (B,T,N,C))
        #    아래 연산에서는 broadcast를 피하기 위해 view와 matmul을 사용
        x_gc = torch.matmul(self.A, x_perm)  # (B, T, N, C)

        # 3) (B, T, N, C) → (B, C, T, N)
        x_gc = x_gc.permute(0, 3, 1, 2).contiguous()

        # 4) 1×1 Conv: 채널 변화 (in_c → out_c)
        out = self.conv1x1(x_gc)  # (B, out_c, T, N)
        return out


class STGCNLayer(nn.Module):
    """
    ST‐GCN 레이어: (TemporalConv → GraphConv → TemporalConv) + BatchNorm + ReLU
    """
    def __init__(self, in_c: int, out_c: int, A: torch.Tensor, t_kernel: int = 3):
        """
        Args:
            in_c (int): 입력 채널 수
            out_c (int): 출력 채널 수
            A (torch.Tensor): 인접행렬, shape=(N,N)
            t_kernel (int): Temporal conv 커널 크기 (논문에서는 3)
        """
        super().__init__()
        pad = (t_kernel - 1) // 2

        # ───────────────────────────────────────────────────────────────────
        # 1) 첫 번째 Temporal Conv (in_c → out_c)
        #    - 커널: (t_kernel, 1)
        #    - 패딩: (pad, 0) → 시간축만 패딩
        # ───────────────────────────────────────────────────────────────────
        self.temp1 = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=(t_kernel, 1),
            padding=(pad, 0),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_c)

        # ───────────────────────────────────────────────────────────────────
        # 2) GraphConv (out_c → out_c)
        # ───────────────────────────────────────────────────────────────────
        self.gconv = GraphConv(out_c, out_c, A)
        self.bn2 = nn.BatchNorm2d(out_c)

        # ───────────────────────────────────────────────────────────────────
        # 3) 두 번째 Temporal Conv (out_c → out_c)
        # ───────────────────────────────────────────────────────────────────
        self.temp2 = nn.Conv2d(
            in_channels=out_c,
            out_channels=out_c,
            kernel_size=(t_kernel, 1),
            padding=(pad, 0),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_c, T, N)
        Returns:
            out: (B, out_c, T, N)
        """
        # (1) TemporalConv → BN → ReLU
        h = self.temp1(x)      # (B, out_c, T, N)
        h = self.bn1(h)
        h = self.relu(h)

        # (2) GraphConv → BN → ReLU
        h = self.gconv(h)      # (B, out_c, T, N)
        h = self.bn2(h)
        h = self.relu(h)

        # (3) TemporalConv → BN → ReLU
        h = self.temp2(h)      # (B, out_c, T, N)
        h = self.bn3(h)
        h = self.relu(h)

        return h


class STGCN(nn.Module):
    """
    원본 ST‐GCN 모델:
    - 입력: (B, C_in, T, N)
    - STGCNLayer × 2
    - 최종 1×1 Convolution (time축을 1로 축소) → (B, C_out, 1, N)
    - squeeze time dim → (B, C_out, N)
    """
    def __init__(self, in_channels: int, out_channels: int, num_nodes: int, A: torch.Tensor):
        """
        Args:
            in_channels (int): 입력 채널 수 (큐+스피드+요일 플래그 등, 예: 9 or 15)
            out_channels (int): 출력 채널 수 (큐+스피드, 8)
            num_nodes (int): 노드(차로) 수 (예: 1370)
            A (torch.Tensor): 인접행렬, shape=(N,N)
        """
        super().__init__()
        self.num_nodes = num_nodes

        # ───────────────────────────────────────────────────────────────────
        # ST‐GCN Layer 1: in_channels → 64
        # ───────────────────────────────────────────────────────────────────
        self.layer1 = STGCNLayer(in_c=in_channels, out_c=64, A=A, t_kernel=3)

        # ───────────────────────────────────────────────────────────────────
        # ST‐GCN Layer 2: 64 → 64
        # ───────────────────────────────────────────────────────────────────
        self.layer2 = STGCNLayer(in_c=64, out_c=64, A=A, t_kernel=3)

        # ───────────────────────────────────────────────────────────────────
        # 마지막 Conv: (64, T=12, N) → (out_channels, 1, N)
        # ※ kernel_size=(12,1) 로 시간축 전체를 한 번에 컨볼루션
        #    패딩 없음 → 출력 time dim = 1
        # ───────────────────────────────────────────────────────────────────
        self.final_conv = nn.Conv2d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=(12, 1),
            padding=(0, 0),
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, 12, N)
        Returns:
            y_hat: (B, out_channels, N)
        """
        B, C_in, T, N = x.shape
        assert T == 12 and N == self.num_nodes, f"입력 크기 오류: {x.shape}"

        # (1) ST‐GCN 레이어 1
        h = self.layer1(x)         # (B, 64, 12, N)

        # (2) ST‐GCN 레이어 2
        h = self.layer2(h)         # (B, 64, 12, N)

        # (3) 마지막 Conv → (B, out_channels, 1, N)
        out = self.final_conv(h)

        # (4) time dim 제거 → (B, out_channels, N)
        return out.squeeze(2)