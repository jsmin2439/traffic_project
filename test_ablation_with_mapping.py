#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_ablation_with_mapping.py

lane_to_segment_id_by_edge.npy 를 실제 cluster_id 로 사용하여
PooledResSTGCN 모델의 Forward PASS가 정상 동작하는지 검증합니다.
"""

import numpy as np
import torch
from data_loader import get_dataloaders
from model.pooled_residual_stgcn import PooledResSTGCN

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, _, _ = get_dataloaders(batch_size=2)
    A = torch.from_numpy(np.load('3_tensor/adjacency/A_lane.npy')).float().to(device)

    mapping    = np.load('lane_to_segment_id_by_edge.npy')
    cluster_id = torch.from_numpy(mapping).long().to(device)

    num_nodes  = mapping.shape[0]
    n_clusters = int(cluster_id.max()) + 1
    model = PooledResSTGCN(
        in_c=9, out_c=8, num_nodes=num_nodes,
        A=A, cluster_id=cluster_id, K=n_clusters,
        hidden_lstm=256
    ).to(device)

    print(f"✔ 모델 생성 완료: nodes={num_nodes}, clusters={n_clusters}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters())}")

    x, y, idx, date = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    # 1) Residual-only Forward
    model.eval()
    with torch.no_grad():
        y_st = model.stgcn(x)
        B = x.size(0)
        res_seq = torch.zeros((B, 12, num_nodes, 8), device=device)
        res_seq[:, 11] = y.permute(0, 2, 1) - y_st.permute(0, 2, 1)
        weekend = x[:, 8, 0, 0].view(-1, 1, 1, 1).expand(-1, 12, num_nodes, 1)
        R_seq = torch.cat([res_seq, weekend], dim=3)
        R_c   = model.pool(R_seq)
        delta_c = model.clstm(R_c)
        from model.pooled_residual_stgcn import scatter_pool
        delta_all = scatter_pool(delta_c, cluster_id)
        res_corr = delta_all[:, :8, :]
        y_hat_ro = y_st + res_corr
    print("✔ Residual-only Forward PASS 성공!  출력 shape:", y_hat_ro.shape)

    # 2) Full Forward with both ema_r and weekend_flag as keywords
    ema_r = y.unsqueeze(1)                # (B,1,8,N)
    weekend_flag = x[:, 8, 0, 0]          # (B,)
    y_hat_f = model(x, ema_r=ema_r, weekend_flag=weekend_flag)
    print("✔ Full Forward PASS 성공!         출력 shape:", y_hat_f.shape)

if __name__ == '__main__':
    main()