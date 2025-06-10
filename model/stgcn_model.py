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
        B, C, T, N = x.shape
        x_perm = x.permute(0, 2, 3, 1).contiguous()  # (B, T, N, C)
        x_gc = torch.matmul(self.A, x_perm)          # (B, T, N, C)
        x_gc = x_gc.permute(0, 3, 1, 2).contiguous() # (B, C, T, N)
        return self.conv1x1(x_gc)                    # (B, out_c, T, N)

class STGCNLayer(nn.Module):
    """
    ST‐GCN 레이어: (TemporalConv → GraphConv → TemporalConv) + BatchNorm + ReLU
    """
    def __init__(self, in_c: int, out_c: int, A: torch.Tensor, t_kernel: int = 3):
        super().__init__()
        pad = (t_kernel - 1) // 2
        self.temp1 = nn.Conv2d(in_c, out_c, kernel_size=(t_kernel,1), padding=(pad,0), bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.gconv = GraphConv(out_c, out_c, A)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.temp2 = nn.Conv2d(out_c, out_c, kernel_size=(t_kernel,1), padding=(pad,0), bias=False)
        self.bn3   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.bn1(self.temp1(x)))
        h = self.relu(self.bn2(self.gconv(h)))
        h = self.relu(self.bn3(self.temp2(h)))
        return h

class STGCN(nn.Module):
    """
    원본 ST‐GCN 모델 (확장 가능):
    - 입력: (B, C_in, T, N)
    - STGCNLayer × 2
    - 최종 1×1 Convolution (time축을 1로 축소) → (B, C_out, N)
    """
    def __init__(
        self,
        in_channels: int,
        hidden1: int,
        out_channels: int,
        num_nodes: int,
        A: torch.Tensor,
        horizon: int = 1
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        # STGCNLayer 1: in_channels → hidden1
        self.layer1 = STGCNLayer(in_channels, hidden1, A, t_kernel=3)
        # STGCNLayer 2: hidden1 → hidden1
        self.layer2 = STGCNLayer(hidden1, hidden1, A, t_kernel=3)
        # 마지막 Conv: (hidden1, T, N) → (out_channels*horizon, 1, N)
        self.final_conv = nn.Conv2d(
            hidden1,
            out_channels * horizon,
            kernel_size=(12,1),
            bias=True
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, N = x.shape
        assert T == 12 and N == self.num_nodes, f"입력 크기 오류: {x.shape}"
        h = self.layer1(x)            # (B, hidden1, 12, N)
        h = self.layer2(h)            # (B, hidden1, 12, N)
        out = self.final_conv(h)               # (B, out_c*horizon, 1, N)
        B, _, _, N = out.shape
        out = out.view(B, -1, self.horizon, N) # (B, out_channels, horizon, N)
        return out
