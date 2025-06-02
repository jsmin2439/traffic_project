# ▣ model.py ──────────────────────────────────────────────────────────────
# 목적 : ① ST-GCN 으로 다음 5 분의 queue+speed(8채널) 1-step 예측
#        ② 예측-실측 잔차를 주말 flag 와 함께 LSTM 에 넣어 장기 주기 보정
# ------------------------------------------------------------------------
# ⚠️  사용 전 확인
#  ▸ A (1370×1370) 는 반드시 row-normalize(D⁻¹A) 또는 symmetric(D⁻¹⁄²AD⁻¹⁄²)
#  ▸ 입력 x 의 shape : (batch, 9, 12, 1370)
#        - 9채널 = [queue4, speed4, weekend_flag]
#  ▸ 출력 y 의 shape : (batch, 8, 1370)
#        - 8채널 = [queue4, speed4]
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────
# 1) GraphConv  : 1-hop Chebyshev(=A) + 1×1 Conv
# ------------------------------------------------------------------------
class GraphConv(nn.Module):
    """
    ▸ 입력  x : (B, C_in, T, N)
    ▸ 출력 out: (B, C_out, T, N)

    계산식:  X' = θ · (A · X)   (# θ = 1×1 Conv, A 정규화 필요)
    """
    def __init__(self, in_c: int, out_c: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer('A', A)      # 그래프 정규화 행렬 저장(학습 X)
        self.theta = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.theta.weight)  # He 초기화

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T, N)  →  (B, T, N, C)
        x = x.permute(0, 2, 3, 1)
        # 그래프 합성곱 : A·X  (B,T,N,C)
        x = torch.matmul(self.A, x)
        # 다시 (B,C,T,N)
        x = x.permute(0, 3, 1, 2)
        # 채널 압축/확장 1×1 Conv
        out = self.theta(x)
        return out

# ─────────────────────────────────────────────────────────────────────────
# 2) ST-GCN Layer : Temporal → Graph → Temporal (T-G-T)
# ------------------------------------------------------------------------
class STGCNLayer(nn.Module):
    """
    in_c → out_c 로 채널 변경 / 시간축 receptive field = 3(기본)
    구조 : TConv(패딩) → BN → ReLU
         → GraphConv   → BN → ReLU
         → TConv(패딩) → BN → ReLU
    """
    def __init__(self, in_c: int, out_c: int, A: torch.Tensor, t_kernel: int = 3):
        super().__init__()
        pad = (t_kernel - 1) // 2  # causal X, 양쪽 동일 padding
        self.temp1 = nn.Conv2d(in_c,  out_c, (t_kernel, 1), padding=(pad, 0))
        self.bn1   = nn.BatchNorm2d(out_c)

        self.graph = GraphConv(out_c, out_c, A)
        self.bn2   = nn.BatchNorm2d(out_c)

        self.temp2 = nn.Conv2d(out_c, out_c, (t_kernel, 1), padding=(pad, 0))
        self.bn3   = nn.BatchNorm2d(out_c)

        self.relu  = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.bn1(self.temp1(x)))   # T-conv1
        h = self.relu(self.bn2(self.graph(h)))   # GraphConv
        h = self.relu(self.bn3(self.temp2(h)))   # T-conv2
        return h

# ─────────────────────────────────────────────────────────────────────────
# 3) ST-GCN 전체 모듈 (2층 + Time-glob Conv)
# ------------------------------------------------------------------------
class STGCN(nn.Module):
    """
    ▸ in_channels  = 9  (queue4 + speed4 + weekend1)
    ▸ out_channels = 8  (queue4 + speed4)  ← time dimension 1 로 압축
    """
    def __init__(self, in_channels: int, out_channels: int,
                 num_nodes: int, A: torch.Tensor):
        super().__init__()
        self.layer1 = STGCNLayer(in_channels, 64, A)   # 9→64
        self.layer2 = STGCNLayer(64, 64, A)            # 64→64

        # 시간축 12→1 로 줄여서 바로 8채널 출력
        self.final  = nn.Conv2d(
            in_channels=64, out_channels=out_channels,
            kernel_size=(12, 1), padding=(0, 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 9, 12, N)  →  out : (B, 8, N)
        """
        h = self.layer1(x)           # (B,64,12,N)
        h = self.layer2(h)           # (B,64,12,N)
        out = self.final(h)          # (B,8,1,N)
        return out.squeeze(2)        # (B,8,N)

# ─────────────────────────────────────────────────────────────────────────
# 4) Residual-LSTM  (장기 패턴 보정)
# ------------------------------------------------------------------------
class ResidualLSTM(nn.Module):
    """
    입력 : 잔차(B,8,N) 플랫 + weekend_flag(B,)  → LSTM(in_len=1)
    출력 : 보정값(B,8,N)
    """
    def __init__(self, num_nodes: int, hidden_size: int = 512):
        super().__init__()
        self.num_nodes  = num_nodes
        in_dim = num_nodes * 8 + 1          # 잔차 벡터 + 주말 플래그
        self.lstm = nn.LSTM(in_dim, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_nodes * 8)

    def forward(self,
                residual_flat: torch.Tensor,  # (B, N*8)
                weekend_flag : torch.Tensor   # (B,) float(0/1)
               ) -> torch.Tensor:
        B = residual_flat.size(0)
        # (B,) → (B,1)
        w = weekend_flag.unsqueeze(1)
        # (B, N*8 + 1) → (B,1, in_dim)
        x = torch.cat([residual_flat, w], dim=1).unsqueeze(1)
        # LSTM
        out, _ = self.lstm(x)          # (B,1,H)
        out = self.fc(out[:, -1, :])   # (B, N*8)
        return out.view(B, 8, self.num_nodes)  # (B,8,N)