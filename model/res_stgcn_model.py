# ┌──────────────────────────────────────────────────────────────────────────┐
# │ model/res_stgcn_model.py                                                  │
# │                                                                          │
# │ 우리 모델: ST‐GCN + Residual LSTM                                          │
# │                                                                          │
# │ 1) ST‐GCN 분기: 과거 12스텝의 (queue+spd+weekend) 9채널 시계열로부터        │
# │    “다음 1스텝 (queue+spd 8채널)” 을 예측                                 │
# │ 2) Residual LSTM 분기: ST‐GCN이 지난 12번 연속으로 예측한 값과 실제 Y의 잔차 │
# │    시퀀스를 받아서, 하루 주기성 및 주말 플래그를 반영한 보정값을 추가 학습   │
# │ 3) 최종 예측: ST‐GCN 예측 + Residual LSTM 보정값                            │
# └──────────────────────────────────────────────────────────────────────────┘

import torch
import torch.nn as nn

# (1) STGCN, GraphConv, STGCNLayer은 stgcn_model.py에서 그대로 가져다 쓰거나 재정의
from model.stgcn_model import GraphConv, STGCNLayer


class STGCN_Base(nn.Module):
    """
    ST‐GCN 단기 예측 분기 모듈:
    - ST‐GCN Layer 2개 → 마지막 1×1 Conv → (B, 8, N)
    - 실제로는 stgcn_model.STGCN과 동일
    """
    def __init__(self, in_channels: int, out_channels: int, num_nodes: int, A: torch.Tensor):
        super().__init__()
        self.num_nodes = num_nodes

        # ST‐GCN 레이어 2개 (in→64, 64→64)
        self.layer1 = STGCNLayer(in_c=in_channels, out_c=64, A=A, t_kernel=3)
        self.layer2 = STGCNLayer(in_c=64, out_c=64, A=A, t_kernel=3)

        # 최종 컨볼루션: (64,12,N) → (out_channels, 1, N) → squeeze → (out_channels, N)
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
            x: (B, in_channels, T=12, N)
        Returns:
            y_st: (B, out_channels=8, N)
        """
        h = self.layer1(x)         # (B, 64, 12, N)
        h = self.layer2(h)         # (B, 64, 12, N)
        out = self.final_conv(h)   # (B, 8, 1, N)
        return out.squeeze(2)      # (B, 8, N)


class ResidualLSTM(nn.Module):
    """
    잔차 보정 분기:

    ST‐GCN이 연속적으로 생성한 12개의 예측값(각각 (B,8,N))을 모아 시퀀스로 삼고,
    실제 Y와 ST‐GCN 예측의 차이(Residual)를 학습하여 ‘하루 주기 패턴 보정값’을 출력.

    - Residual 시퀀스: (B, seq_len=12, nodes=1370, channels=8)
      → seq_len=12 시퀀스를 LSTM에 투입하기 위해 다음과 같이 전처리:
        1) residual_flat: (B, seq_len, nodes*channels) → (B, 12, 1370*8)
        2) weekend_flag: (B,) 0 or 1

    - LSTM: input_size = (nodes*channels + 1), hidden_size = hidden_dim
      → 입력은 “(residual_flat[:, t, :] concatenated with weekend_flag)” 형태.
      → 출력: 최종 시점의 hidden → FC → (nodes*channels)  
      → reshape → (B, 8, nodes)

    - 최종 예측: y_ST + y_res_corr
    """
    def __init__(self, num_nodes: int, hidden_dim: int = 512):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = 12       # ST‐GCN이 연속 12스텝을 먼저 예측 → 12개 잔차 시퀀스
        self.num_feats = num_nodes * 8  # 잔차 채널 총 개수 (1370*8)

        # LSTM input dim = num_feats (잔차) + 1 (weekend_flag)
        self.lstm_in_dim = self.num_feats + 1
        self.hidden_dim = hidden_dim

        # ───────────────────────────────────────────────────────────────────
        # LSTM: 입력 차원 = lstm_in_dim, 은닉 차원 = hidden_dim
        #   batch_first=True: (B, seq_len, feature)
        #   단방향 LSTM
        # ───────────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=self.lstm_in_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # ───────────────────────────────────────────────────────────────────
        # FC: hidden_dim → num_feats (nodes*8)
        # ───────────────────────────────────────────────────────────────────
        self.fc = nn.Linear(self.hidden_dim, self.num_feats)

    def forward(self, res_seq: torch.Tensor, weekend_flag: torch.Tensor) -> torch.Tensor:
        """
        Args:
            res_seq (torch.Tensor): ST‐GCN 잔차 시퀀스, shape = (B, 12, N, 8)
              - N=1370, 8=queue+spd
            weekend_flag (torch.Tensor): 주말 여부, shape = (B,) 0 or 1

        Returns:
            res_corr (torch.Tensor): 보정값, shape = (B, 8, N)
        """
        B, seq_len, N, C = res_seq.shape
        assert seq_len == self.seq_len and N == self.num_nodes and C == 8, \
            f"res_seq shape 예상: (B,12,1370,8), 현재: {res_seq.shape}"
        assert weekend_flag.shape == (B,), f"weekend_flag shape는 (B,) 이어야 합니다. 현재: {weekend_flag.shape}"

        # ───────────────────────────────────────────────────────────────────
        # 1) 잔차 시퀀스를 Flatten: (B, 12, 1370, 8) → (B, 12, 1370*8)
        # ───────────────────────────────────────────────────────────────────
        res_flat = res_seq.view(B, seq_len, -1)   # (B,12,1370*8)

        # ───────────────────────────────────────────────────────────────────
        # 2) weekend_flag를 시퀀스 길이만큼 확장하여 LSTM 입력과 결합
        #    - weekend_flag: (B,) → (B,12,1) (모든 타임스텝에서 동일한 flag 사용)
        # ───────────────────────────────────────────────────────────────────
        # (B,) → (B,1) → (B,1,1) → expand → (B,12,1)
        w = weekend_flag.view(B, 1).unsqueeze(2).expand(B, seq_len, 1)  # (B,12,1)

        # (B,12,1370*8) + (B,12,1) concatenate → (B,12, 1370*8 +1)
        lstm_input = torch.cat([res_flat, w], dim=2)  # (B,12, num_feats+1)

        # ───────────────────────────────────────────────────────────────────
        # 3) LSTM forward: (B,12, num_feats+1) → outputs: (B,12, hidden_dim)
        # ───────────────────────────────────────────────────────────────────
        out, (h_n, c_n) = self.lstm(lstm_input)
        # out[:, -1, :] → (B, hidden_dim) = 마지막 time step hidden state
        last_h = out[:, -1, :]  # (B, hidden_dim)

        # ───────────────────────────────────────────────────────────────────
        # 4) FC: (B, hidden_dim) → (B, num_feats) = (B, 1370*8)
        # ───────────────────────────────────────────────────────────────────
        y_flat = self.fc(last_h)  # (B, 1370*8)

        # ───────────────────────────────────────────────────────────────────
        # 5) 복원: (B, 1370*8) → (B, 8, 1370)
        # ───────────────────────────────────────────────────────────────────
        res_corr = y_flat.view(B, 8, N)  # (B,8,1370)

        return res_corr


class ResSTGCN(nn.Module):
    """
    최종 하이브리드 모델: ST‐GCN 단기 예측 + Residual LSTM 장기 보정
    """
    def __init__(self, in_channels: int, out_channels: int, num_nodes: int, A: torch.Tensor, hidden_dim: int = 512):
        """
        Args:
            in_channels (int): ST‐GCN 입력 채널 수 (큐+스피드+weekend, 예: 9 or 15)
            out_channels (int): ST‐GCN 출력 채널 수(큐+스피드=8)
            num_nodes (int): 노드(차로) 수 (1370)
            A (torch.Tensor): 인접행렬, shape=(N,N)
            hidden_dim (int): Residual LSTM hidden dimension
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.out_channels = out_channels

        # ───────────────────────────────────────────────────────────────────
        # (1) ST‐GCN 분기: 단기 예측
        # ───────────────────────────────────────────────────────────────────
        self.stgcn = STGCN_Base(in_channels=in_channels, out_channels=out_channels, num_nodes=num_nodes, A=A)

        # ───────────────────────────────────────────────────────────────────
        # (2) Residual LSTM 분기: 잔차 보정
        # ───────────────────────────────────────────────────────────────────
        self.reslstm = ResidualLSTM(num_nodes=num_nodes, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor, res_seq: torch.Tensor = None, weekend_flag: torch.Tensor = None) -> torch.Tensor:
        B, C_in, T, N = x.shape
        # ───────────────────────────────────────────────────────────────────
        # (1) ST‐GCN 단기 예측
        # ───────────────────────────────────────────────────────────────────
        y_pred_st = self.stgcn(x)  # (B, 8, N)

        # 만약 res_seq 또는 weekend_flag가 주어지지 않으면(추론 단계) → y_pred만 반환
        if res_seq is None or weekend_flag is None:
            return y_pred_st

        # ───────────────────────────────────────────────────────────────────
        # (2) Residual LSTM 보정 (학습 시 사용)
        # ───────────────────────────────────────────────────────────────────
        res_corr = self.reslstm(res_seq, weekend_flag)  # (B, 8, N)

        # ───────────────────────────────────────────────────────────────────
        # (3) 최종 예측: y_pred_st + res_corr (학습 시) 
        # ───────────────────────────────────────────────────────────────────
        y_final = y_pred_st + res_corr  # (B, 8, N)
        return y_final