# 파일 위치: project_root/model/gated_fusion_stgcn.py
# ──────────────────────────────────────────────────────────────────────────
# Gated Fusion 기반 하이브리드 ST-GCN + 경량 Temporal Branch 모델
#  - ST-GCN: 단기 시공간 특징 추출 (12스텝)
#  - Temporal Branch: LSTM 또는 TCN 선택 가능
#    → LSTM: 반복 주기 패턴 보정 (주중/주말)
#    → TCN: 경량화된 시계열 보정
#  최종 예측 = ST-GCN 예측 + 보정값
# ──────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
from model.stgcn_model import STGCNLayer


class STGCNBackbone(nn.Module):
    """
    ST-GCN Backbone 모듈
    - 입력: 과거 T=12 스텝의 9채널 시계열 (queue4 + speed4 + holiday_flag)
      shape = (B, in_channels, T, N)
    - 출력: 시공간 특징 텐서
      shape = (B, hidden1, T, N)
    """
    def __init__(self, in_channels: int, hidden1: int, A: torch.Tensor):
        super().__init__()
        self.layer1 = STGCNLayer(in_c=in_channels,
                                 out_c=hidden1,
                                 A=A,
                                 t_kernel=3)
        self.layer2 = STGCNLayer(in_c=hidden1,
                                 out_c=hidden1,
                                 A=A,
                                 t_kernel=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, T=12, N)
        h = self.layer1(x)  # (B, hidden1, T, N)
        h = self.layer2(h)  # (B, hidden1, T, N)
        return h            # (B, hidden1, T, N)


class PatternLSTM(nn.Module):
    """
    LSTM Branch 모듈
    - 입력: ST-GCN 특징 및 holiday_flag
      Z1 shape = (B, hidden1, T, N)
      holiday_flag shape = (B,)
    - 출력: 보정 특징 Z2
      shape = (B, N, hidden2)
    """
    def __init__(self, hidden1: int, hidden2: int, seq_len: int = 12):
        super().__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=hidden1 + 1,
            hidden_size=hidden2,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, Z1: torch.Tensor, holiday_flag: torch.Tensor) -> torch.Tensor:
        B, hidden1, T, N = Z1.shape
        # (B, hidden1, T, N) -> (B*N, T, hidden1)
        seq = Z1.permute(0, 3, 2, 1).reshape(B * N, T, hidden1)
        wf = holiday_flag.view(B, 1).expand(B, N).reshape(B * N, 1)
        wf_seq = wf.unsqueeze(1).expand(B * N, T, 1)
        lstm_in = torch.cat([seq, wf_seq], dim=2)  # (B*N, T, hidden1+1)
        out, _ = self.lstm(lstm_in)                # (B*N, T, hidden2)
        h_T = out[:, -1, :]                        # (B*N, hidden2)
        return h_T.view(B, N, self.hidden2)        # (B, N, hidden2)


class PatternTCN(nn.Module):
    """
    TCN Branch (경량화)
    - 입력: ST-GCN 특징 및 holiday_flag
      Z1 shape = (B, hidden1, T, N)
      holiday_flag shape = (B,)
    - 출력: 보정 특징 Z2
      shape = (B, N, hidden2)
    """
    def __init__(self, hidden1: int, hidden2: int, seq_len: int = 12):
        super().__init__()
        # 1차원 시간 합성: Conv2d(kernel=(3,1))
        self.conv = nn.Conv2d(
            in_channels=hidden1 + 1,
            out_channels=hidden2,
            kernel_size=(3, 1),
            padding=(1, 0)
        )

    def forward(self, Z1: torch.Tensor, holiday_flag: torch.Tensor) -> torch.Tensor:
        B, hidden1, T, N = Z1.shape
        # holiday_flag → (B,1,T,N)
        wf = holiday_flag.view(B, 1, 1, 1).expand(B, 1, T, N)
        # concat along 채널: (B, hidden1+1, T, N)
        x = torch.cat([Z1, wf], dim=1)
        # conv → (B, hidden2, T, N)
        out = self.conv(x)
        # 시간축 평균 풀링: (B, hidden2, N)
        pooled = out.mean(dim=2)
        # 순서 변경: (B, N, hidden2)
        return pooled.permute(0, 2, 1)


class GatedFusionSTGCN(nn.Module):
    """
    하이브리드 예측 모델
    - ST-GCN + (LSTM or TCN) 보정
    - 게이트에 Dropout 적용, 초기 게이트 편향으로 g≈0.5 유지
    - 예측 주기: 다음 5분 스텝 → 5분마다 실시간 추론
    """
    def __init__(
        self,
        in_channels: int,
        hidden1: int,
        hidden2: int,
        out_channels: int,
        num_nodes: int,
        A: torch.Tensor,
        wd_emb_dim: int = 16,
        use_tcn: bool = False,
        gate_dropout: float = 0.2
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.use_tcn = use_tcn
        # ST-GCN
        self.backbone = STGCNBackbone(in_channels, hidden1, A)
        # Temporal 보정 Branch
        if use_tcn:
            self.temp_branch = PatternTCN(hidden1, hidden2)
        else:
            self.temp_branch = PatternLSTM(hidden1, hidden2)
        # 주말 플래그 임베딩
        self.wd_lin = nn.Linear(1, wd_emb_dim)
        # 게이트 MLP + Dropout
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden1 + hidden2 + wd_emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(gate_dropout),
            nn.Linear(64, 1)
        )
        # 게이트 편향 초기화: sigmoid(0)=0.5
        nn.init.constant_(self.gate_mlp[-1].bias, 0.0)
        # ST-GCN 풀링 → hidden2 차원 매핑
        self.fuse_lin1 = nn.Linear(hidden1, hidden2)
        # 예측 헤드
        self.pred_head = nn.Linear(hidden2, out_channels)
        # 예측 스텝 주기 (초)
        self.pred_interval = 5 * 60  # 5분

    def forward(self, x: torch.Tensor, holiday_flag: torch.Tensor) -> torch.Tensor:
        B, C, T, N = x.shape
        # 1) 시공간 특징
        Z1 = self.backbone(x)                       # (B, hidden1, T, N)
        # 2) 시간 평균 풀링
        Z1_pool = Z1.mean(dim=2)                    # (B, hidden1, N)
        Z1_pool_t = Z1_pool.permute(0, 2, 1)         # (B, N, hidden1)
        # 3) Temporal Branch
        Z2 = self.temp_branch(Z1, holiday_flag)     # (B, N, hidden2)
        # 4) 성장 플래그 임베딩
        wf = holiday_flag.view(B, 1)
        E_wd = self.wd_lin(wf).unsqueeze(1).expand(B, N, -1)
        # 5) Gated Fusion
        cat = torch.cat([Z1_pool_t, Z2, E_wd], dim=2)      # (B,N,*)
        g = torch.sigmoid(self.gate_mlp(cat))              # (B,N,1)
        # 선택적 clamp to avoid extremes
        g = torch.clamp(g, min=0.1, max=0.9)
        H1 = self.fuse_lin1(Z1_pool_t)                     # (B,N,hidden2)
        H2 = Z2                                           # (B,N,hidden2)
        Zf = g * H1 + (1 - g) * H2                        # (B,N,hidden2)
        # 6) 예측
        y = self.pred_head(Zf)                            # (B,N,out_channels)
        return y.permute(0, 2, 1)                         # (B,out_channels,N)

    def predict_interval(self) -> int:
        """
        예측 결과 반환 주기 (초)
        """
        return self.pred_interval

# ──────────────────────────────────────────────────────────────────────────
# 개선 사항:
# 1. Temporal Branch 선택: TCN 옵션(use_tcn=True)으로 LSTM flatten(B*N) 제거 → 메모리 절감
# 2. Gate Dropout 및 bias 초기화 → 게이트 극단 치우침 방지
# 3. Gate clamp(min=0.1,max=0.9) → 안정적 융합 비율 확보
# 4. 예측 주기 pred_interval 속성 추가 → 5분마다 실시간 추론 지원
# 5. 데이터 불균형 보정은 데이터 로더 단계에서 오버샘플링/손실 가중치 적용 권장
# 6. 실시간 제어용 경량 버전은 hidden 차원 축소 및 use_tcn=True 옵션 활용
# ──────────────────────────────────────────────────────────────────────────
