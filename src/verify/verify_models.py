#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_models.py

목적: 기본 LSTM, ST-GCN, GatedFusionSTGCN, PooledResSTGCN 모델의 Forward PASS를
더미 입력으로 검증합니다.
"""

import torch
import numpy as np

# ─────────────────────────────────────────────────────────────────────────
# 공통 변수
# ─────────────────────────────────────────────────────────────────────────
B = 2           # 배치 크기
T = 12          # 시퀀스 길이
N = 1370        # 노드 수
C_traffic = 8   # Queue4+Speed4 채널 수
C_flag = 1      # 주말/공휴일 플래그 채널 수

# ─────────────────────────────────────────────────────────────────────────
# 1) 전통적 LSTM 모델
# ─────────────────────────────────────────────────────────────────────────
from model.lstm_model import BasicLSTM

C_lstm = C_traffic + C_flag
# (B, C_in, T, N)
x_lstm = torch.randn(B, C_lstm, T, N)

lstm_model = BasicLSTM(num_nodes=N, input_dim=C_traffic)
lstm_model.eval()
with torch.no_grad():
    y_lstm = lstm_model(x_lstm)
print("BasicLSTM Forward PASS 성공!  출력 shape:", y_lstm.shape)  # 예상: (B, 8, N)

# ─────────────────────────────────────────────────────────────────────────
# 2) 원본 ST-GCN 모델
# ─────────────────────────────────────────────────────────────────────────
from model.stgcn_model import STGCN

C_stgcn = C_traffic + C_flag
x_stgcn = torch.randn(B, C_stgcn, T, N)
A_dummy = torch.eye(N)

stgcn_model = STGCN(
        in_channels=C_stgcn,
        hidden1=64,
        out_channels=C_traffic,
        num_nodes=N,
        A=A_dummy
    )
stgcn_model.eval()
with torch.no_grad():
    y_stgcn = stgcn_model(x_stgcn)
print("ST-GCN Forward PASS 성공!  출력 shape:", y_stgcn.shape)  # 예상: (B, 8, N)

# ─────────────────────────────────────────────────────────────────────────
# 3) Gated Fusion ST-GCN 모델
# ─────────────────────────────────────────────────────────────────────────
from model.gated_fusion_stgcn import GatedFusionSTGCN

C_gated = C_traffic + C_flag
x_gated = torch.randn(B, C_gated, T, N)
holiday_flag = torch.randint(0, 2, (B,))
ema_r = torch.randn(B, T, C_traffic, N)

gated_model = GatedFusionSTGCN(
    in_channels=C_gated,
    hidden1=64,
    hidden2=32,
    out_channels=C_traffic,
    num_nodes=N,
    A=A_dummy
)
gated_model.eval()
with torch.no_grad():
    # GatedFusionSTGCN.forward()는 ema_r 인자를 받지 않으므로, 
        # 주말 플래그만 넘겨 호출합니다.
        # LongTensor → FloatTensor 변환
        y_gated = gated_model(x_gated, holiday_flag.float())
print("GatedFusionSTGCN Forward PASS 성공!  출력 shape:", y_gated.shape)  # 예상: (B, 8, N)

# ─────────────────────────────────────────────────────────────────────────
# 4) Pooled Residual ST-GCN 모델
# ─────────────────────────────────────────────────────────────────────────
from model.pooled_residual_stgcn import PooledResSTGCN

C_pool = C_traffic + C_flag
x_pool = torch.randn(B, C_pool, T, N)
cluster_id = torch.zeros(N, dtype=torch.long)

pooled_model = PooledResSTGCN(
    in_c=C_pool,
    out_c=C_traffic,
    num_nodes=N,
    A=A_dummy,
    cluster_id=cluster_id
)
pooled_model.eval()
with torch.no_grad():
    y_pooled = pooled_model(x_pool, ema_r=ema_r, weekend_flag=holiday_flag)
print("PooledResSTGCN Forward PASS 성공!  출력 shape:", y_pooled.shape)  # 예상: (B, 8, N)
