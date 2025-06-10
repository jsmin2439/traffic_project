import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from model.stgcn_model import STGCNLayer, GraphConv
from typing import Optional

class STGCN_Base(nn.Module):
    """
    ST‐GCN 단기 예측 분기 모듈: 2-layer STGCN ⇒ Conv1×1
    """
    def __init__(self, in_channels, out_channels, num_nodes, A):
        super().__init__()
        self.num_nodes = num_nodes
        self.layer1 = STGCNLayer(in_c=in_channels, out_c=64, A=A, t_kernel=3)
        self.layer2 = STGCNLayer(in_c=64, out_c=64, A=A, t_kernel=3)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=(12,1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, N = x.shape
        assert T == 12, f"Expected sequence length T == 12, got T={T}"
        h = self.layer1(x)
        h = self.layer2(h)
        out = self.final_conv(h)        # (B, out_c, 1, N)
        return out.squeeze(2)

class NetVLADPool(nn.Module):
    """
    Soft-pooling 모듈 (NetVLAD 스타일)
    입력: R_seq (B, T, N, C)
    출력: R_c   (B, T, K, C)
    """
    def __init__(self, num_clusters: int, dim: int, initial_alpha: float = 1.0, max_alpha: float = 100.0):
        super().__init__()
        self.K = num_clusters
        self.C = dim
        # Xavier 초기화 후 L2 정규화된 centroids
        self.centroids = nn.Parameter(torch.empty(self.K, self.C))
        nn.init.xavier_uniform_(self.centroids)
        with torch.no_grad():
            self.centroids.data = F.normalize(self.centroids.data, dim=1)
        self.cluster_weight = nn.Linear(self.C, self.K)
        self.initial_alpha = initial_alpha
        self.max_alpha = max_alpha
        self.alpha = initial_alpha

    def forward(self, R_seq: torch.Tensor) -> torch.Tensor:
        B, T, N, C = R_seq.shape
        assert C == self.C, f"Expected feature dim={self.C}, got {C}"

        # (B, T, N, C) → (B*T, N, C)
        r_bt = R_seq.reshape(B * T, N, C)
        # (B*T*N, C)
        r_flat = r_bt.reshape(-1, C)
        # logits 및 softmax
        logits = self.cluster_weight(r_flat) * self.alpha  # (B*T*N, K)
        a = F.softmax(logits, dim=-1).reshape(B * T, N, self.K)  # (B*T, N, K)
        # residuals to centroids
        c = self.centroids.view(1, 1, self.K, C)  # (1,1,K,C)
        res = r_bt.unsqueeze(2) - c                # (B*T, N, K, C)
        # weighted sum over nodes
        v = (a.unsqueeze(-1) * res).sum(dim=1)     # (B*T, K, C)
        # 복원
        R_c = v.reshape(B, T, self.K, C)          # (B, T, K, C)
        return R_c

class ClusterLSTM(nn.Module):
    """
    입력: R_c (B, T, K, C)
    출력: Δ_c (B, K, C)
    개선: per-cluster 시퀀스를 (B*K, T, C) 형태로 보고,
         1) nn.Linear(C→H) → 2) LSTM(H→H) → 3) nn.Linear(H→C) 복원
    """
    def __init__(self, num_clusters: int, channels: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.K, self.C = num_clusters, channels
        self.hidden = hidden
        # 1) 각 시점별 feature C→hidden 투영
        self.pre_fc  = nn.Linear(self.C, hidden)
        # 2) 시퀀스 모델링
        self.lstm    = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        # 3) 마지막 hidden→C 복원
        self.post_fc = nn.Linear(hidden, self.C)
        # Initialize post_fc bias to zero for stable residual start
        nn.init.zeros_(self.post_fc.bias)

    def forward(self, R_c: torch.Tensor) -> torch.Tensor:
        """
        R_c: (B, T, K, C)
        returns delta_c: (B, K, C)
        """
        B, T, K, C = R_c.shape
        # (B, T, K, C) → (B*K, T, C)
        seq = R_c.permute(0,2,1,3).reshape(B*K, T, C)
        # step1: C→hidden
        h_in = self.pre_fc(seq)               # (B*K, T, hidden)
        # step2: LSTM
        h_seq, _ = self.lstm(h_in)            # (B*K, T, hidden)
        h_seq = self.dropout(h_seq)
        last  = h_seq[:, -1, :]               # (B*K, hidden)
        # step3: hidden→C
        out   = self.post_fc(last)            # (B*K, C)
        # 복원: (B, K, C)
        delta_c = out.view(B, K, C)
        return delta_c

def scatter_pool(delta_c: torch.Tensor, cluster_masks: torch.Tensor) -> torch.Tensor:
    """
    delta_c: (B, K, C), cluster_masks: (K, N) boolean mask
    return Δ (B, C, N)
    """
    B, K, C = delta_c.shape
    K2, N = cluster_masks.shape
    assert K == K2, f"cluster_masks must have shape (K, N), got {cluster_masks.shape}"
    # Initialize output
    out = delta_c.new_zeros(B, C, N)
    # Scatter per-cluster delta to nodes
    for k in range(K):
        mask = cluster_masks[k]  # (N,) boolean
        if mask.any():
            # Broadcast delta for cluster k to its nodes
            out[:, :, mask] = delta_c[:, k, :].unsqueeze(-1)
            # Actually: delta_c[:,k,:] shape (B,C), unsqueeze to (B,C,1) then broadcast to mask.sum()
    return out

class PooledResSTGCN(nn.Module):
    """
    ST-GCN + NetVLAD Soft Pooling + Cluster-LSTM 보정

    전체 파이프라인:
    1) ST-GCN Backbone으로 기본 단기 예측 수행 (단일 스텝)
    2) 과거 잔차 시퀀스에 대해 NetVLAD 기반 Soft Pooling 수행하여 클러스터별 특성 생성
    3) Cluster-LSTM으로 클러스터별 시퀀스 모델링 및 잔차 보정값 예측
    4) 클러스터 보정값을 노드별로 분산(scatter)시켜 보정 후, 다중 스텝 예측을 위한 Conv1x1 변환 수행
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        num_nodes: int,
        A: torch.Tensor,
        cluster_id: torch.Tensor,
        K: int = 32,
        hidden_lstm: int = 64,
        horizon: int = 1
    ):
        super().__init__()
        assert cluster_id.numel() == num_nodes, "cluster_id 길이가 num_nodes와 달라요"
        self.num_nodes = num_nodes
        self.out_c = out_c
        self.horizon = horizon

        # 1) ST-GCN Backbone
        self.stgcn = STGCN_Base(in_c, out_c, num_nodes, A)
        # 2) Soft-Pooling + Cluster-LSTM (채널 수 out_c→out_c+1)
        self.pool  = NetVLADPool(K, dim=out_c + 1)
        self.clstm = ClusterLSTM(K, channels=out_c + 1, hidden=hidden_lstm)
        self.multi_conv = nn.Conv2d(out_c, out_c * horizon, kernel_size=(1,1), bias=True)
        # 3) 클러스터 매핑
        self.register_buffer("cluster_id", cluster_id)

        # Precompute cluster-to-node boolean masks
        masks = []
        for k in range(K):
            masks.append(cluster_id == k)  # boolean mask of shape (N,)
        self.register_buffer("cluster_masks", torch.stack(masks, dim=0))  # (K, N)

    def forward(self, x: torch.Tensor, residual_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Separate holiday_flag from input channels (last channel)
        # x: (B, C_in, T, N), where last channel is flag
        feat = x[:, :-1, :, :]                      # (B, C_in-1, T, N)
        holiday_flag = x[:, -1, 0, 0]               # (B,)
        B, C, T, N = x.shape
        horizon = self.horizon
        # ST-GCN 예측 (단일 스텝)
        y_s = self.stgcn(feat)    # (B, out_c, N)

        # Residual sequence 준비
        assert residual_seq is not None, \
            "forward 호출 시 residual_seq를 넘겨야 합니다."
        # residual_seq: (B, T, C, N)
        R_seq = residual_seq - y_s.unsqueeze(1)  # (B, T, C, N)
        # (B, T, N, C)
        R_seq = R_seq.permute(0, 1, 3, 2)

        # 플래그 채널 추가 → (B, T, N, C+1)
        num_steps = R_seq.size(1)  # 실제 시퀀스 길이 (horizon)
        wf = holiday_flag.view(B,1,1,1).expand(B, num_steps, N, 1)
        R_seq = torch.cat([R_seq, wf], dim=3)  # (B, T, N, C+1)

        # 보정 경로: Soft Pooling + Cluster-LSTM
        R_c     = self.pool(R_seq)          # (B, T, K, C+1)
        delta_c = self.clstm(R_c)           # (B, K, C+1)
        delta_all = scatter_pool(delta_c, self.cluster_masks)  # (B, C+1, N)
        delta = delta_all[:, :self.out_c, :]                 # (B, out_c, N)
        # Add delta correction and produce multi-step output
        corrected = (y_s + delta).unsqueeze(2)            # (B, C, 1, N)
        multi = self.multi_conv(corrected)                # (B, C*horizon, 1, N)
        return multi.view(B, self.out_c, self.horizon, N)
    def update_alpha(self, epoch: int, total_epochs: int):
        """
        Linearly (or exponentially) schedule alpha from initial_alpha to max_alpha.
        """
        # linear schedule
        self.alpha = self.initial_alpha + (self.max_alpha - self.initial_alpha) * (epoch / (total_epochs - 1))