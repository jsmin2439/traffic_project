# ┌──────────────────────────────────────────────────────────────────────────┐
# │ model/lstm_model.py                                                      │
# │                                                                          │
# │ 전통적 LSTM 기반 교통량 예측 모델 클래스 정의                              │
# │                                                                          │
# │ - 입력: 과거 12스텝 동안 각 노드(1370)별 차종별 Queue+Speed(8채널) 시계열    │
# │ - 처리: 노드별 시계열을 하나의 파이프라인(batch_size×1370)로 합쳐서 LSTM을  │
# │   공유 가중치로 통과시키고, 최종 시점 결과를 FC로 8채널 출력              │
# │ - 출력: 다음 스텝(5분 후)의 각 노드별 Queue+Speed 8채널 예측               │
# └──────────────────────────────────────────────────────────────────────────┘

import torch
import torch.nn as nn

class BasicLSTM(nn.Module):
    """
    전통적 LSTM 모델:
    - num_nodes개의 노드가 있고, 각 노드마다 채널 수(feature_dim=8)의 시계열을 가짐.
    - 과거 T=12 스텝의 시계열을 LSTM으로 처리하여, 마지막 타임스텝 hidden state를 가져와
      FC 레이어로 8개 출력(Queue+Speed)을 예측.
    - 모든 노드에는 '공유된 가중치' LSTM을 적용함으로써, 파라미터 수를 절약.
    """

    def __init__(self, num_nodes: int = 1370, input_dim: int = 8, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0):
        """
        Args:
            num_nodes (int): 예측할 노드(차로) 수, (예: 1370)
            input_dim (int): LSTM 입력 feature 차원 (Queue4 + Speed4 = 8)
            hidden_dim (int): LSTM hidden state 차원
            num_layers (int): LSTM 레이어 개수
            dropout (float): LSTM 내부 드롭아웃 비율 (레이어 >1일 때 적용)
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ───────────────────────────────────────────────────────────────────
        # 1) LSTM: shared across all nodes
        #    - input size: input_dim (8)
        #    - hidden size: hidden_dim (64)
        #    - num_layers: num_layers
        #    - batch_first=True 로 (batch, seq_len, feature) 순서
        #    - dropout: if num_layers >1, 내부 레이어 사이에 적용
        # ───────────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False  # 단방향 LSTM
        )

        # ───────────────────────────────────────────────────────────────────
        # 2) FC: 마지막 time step의 hidden state (크기 = hidden_dim) → 8개 채널 예측
        #    - 노드마다 독립적인 출력이지만, 가중치는 공유됨
        #    - 입력 차원: hidden_dim
        #    - 출력 차원: input_dim (8) → Queue4 + Speed4
        # ───────────────────────────────────────────────────────────────────
        self.fc = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 입력 텐서, shape = (B, 12, num_nodes, 8)
              - B: batch size
              - 12: 과거 12스텝
              - num_nodes: 1370
              - 8: Queue4 + Speed4
        Returns:
            y_hat (torch.Tensor): 예측값, shape = (B, num_nodes, 8)
        """
        B, T, N, C = x.shape
        # 입력 shape 체크 (디버깅용)
        assert T == 12 and N == self.num_nodes and C == self.input_dim, \
            f"입력 크기 오류: expected (B,12,{self.num_nodes},8), got {x.shape}"

        # ───────────────────────────────────────────────────────────────────
        # (1) LSTM 처리: 모든 노드·배치 차원을 하나로 합쳐서 시계열 처리
        # ───────────────────────────────────────────────────────────────────
        #   - x.view → (B * N, T, C)
        #   - shared LSTM → 출력 hidden states: (B*N, T, hidden_dim)
        #   - 우리는 마지막 time step(hidden state)만 사용할 것임
        x_reshaped = x.permute(0, 2, 1, 3).contiguous()   # (B, N, 12, 8)
        x_reshaped = x_reshaped.view(B * N, T, C)         # (B*N, 12, 8)

        # LSTM forward
        #   out: (B*N, 12, hidden_dim)
        #   (h_n, c_n): 각각 (num_layers, B*N, hidden_dim) 
        out, (h_n, c_n) = self.lstm(x_reshaped)

        # 우리는 마지막 time step의 out[:, -1, :]를 사용
        last_hidden = out[:, -1, :]   # (B*N, hidden_dim)

        # ───────────────────────────────────────────────────────────────────
        # (2) FC 처리: 마지막 hidden state → 8개 채널 예측
        # ───────────────────────────────────────────────────────────────────
        y_flat = self.fc(last_hidden)  # (B*N, 8)

        # (3) 원래 배치 형태로 복원: (B, N, 8)
        y_hat = y_flat.view(B, N, C)    # (B, 1370, 8)

        return y_hat