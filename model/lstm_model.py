# ┌──────────────────────────────────────────────────────────────────────────┐
# │ model/lstm_model.py                                                      │
# │                                                                          │
# │ 전통적 LSTM 기반 교통량 예측 모델 클래스 정의                              │
# │                                                                          │
# │ - 입력: 과거 12스텝 동안 각 노드(1370)별 Queue4+Speed4(8채널) 시계열과       │
# │   주말 플래그(1채널) 포함 (총 9채널)                                        │
# │ - 처리: 채널 0~7 데이터를 노드별로 모아 LSTM 공유 가중치로 처리,            │
# │   마지막 히든 스테이트를 FC로 변환하여 8채널 예측                          │
# │ - 출력: 다음 5분 스텝 후 각 노드별 Queue4+Speed4 예측 (8채널)              │
# └──────────────────────────────────────────────────────────────────────────┘

import torch
import torch.nn as nn

class BasicLSTM(nn.Module):
    """
    전통적 LSTM 모델:
    - 입력 채널: Queue4+Speed4(8채널) + WeekendFlag(1채널) = 9채널
    - 과거 T=12 스텝의 시계열을 LSTM으로 처리하여, 마지막 타임스텝 hidden state를 FC로 변환
    - 모든 노드에 동일한 LSTM 가중치를 공유
    - 예측: 다음 스텝의 Queue4+Speed4 (8채널)
    """

    def __init__(self,
                 num_nodes: int = 1370,
                 input_dim: int = 8,
                 hidden_dim: int = 64,
                 num_layers: int = 1,
                 dropout: float = 0.0):
        """
        Args:
            num_nodes (int): 노드(차로) 수 (예: 1370)
            input_dim (int): 입력 feature 수 (Queue4+Speed4=8)
            hidden_dim (int): LSTM hidden 차원
            num_layers (int): LSTM 레이어 수
            dropout (float): 레이어 간 드롭아웃 비율 (num_layers>1일 때 적용)
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM: 입력 feature = 8, hidden = hidden_dim
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        # FC: 마지막 hidden -> 8채널 예측
        self.fc = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, x: torch.Tensor, weekend_flag: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 입력 텐서, shape=(B,9,12,N)
              - 채널 0~7: Queue+Speed, 채널 8: weekend_flag
            weekend_flag: 사용하지 않음(호환성 유지)

        Returns:
            y_hat (torch.Tensor): 예측값, shape=(B,8,N)
        """
        B, C, T, N = x.shape
        assert C == self.input_dim + 1 and T == 12 and N == self.num_nodes, \
            f"입력 크기 오류: expected (B,9,12,{self.num_nodes}), got {x.shape}"

        # 채널 0~7 선택
        seq = x[:, :self.input_dim]             # (B,8,12,N)
        # (B,8,12,N) -> (B,12,N,8)
        seq = seq.permute(0, 2, 3, 1).contiguous()
        # (B*N,12,8)
        seq = seq.view(B * N, T, self.input_dim)

        # LSTM forward
        out, _ = self.lstm(seq)                 # (B*N,12,hidden_dim)
        last_h = out[:, -1, :]                  # (B*N,hidden_dim)
        flat = self.fc(last_h)                  # (B*N,8)

        # (B,N,8) -> (B,8,N)
        y = flat.view(B, N, self.input_dim).permute(0, 2, 1).contiguous()
        return y
