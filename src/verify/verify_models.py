# ┌──────────────────────────────────────────────────────────────────────────┐
# │ verify_models.py                                                          │
# │                                                                          │
# │ 목적: 세 모델(전통적 LSTM, 원본 ST-GCN, 우리 모델)의 Forward PASS를      │
# │ 간단한 가짜 입력으로 각자 올바르게 연산되는지 검증합니다.                │
# └──────────────────────────────────────────────────────────────────────────┘

import torch
import numpy as np

# STEP 1: 데이터 로더(생략) 대신, 랜덤 입력으로 확인
# 예: B=2 (batch size), T=12, N=1370, C_in=8(큐+스피드)
B = 2
T = 12
N = 1370
C_in = 8

# 랜덤 입력 생성 (정규분포 N(0,1) 가정)
x_rand = torch.randn(B, T, N, C_in)

# ─────────────────────────────────────────────────────────────────────────
# STEP 2: 전통적 LSTM 모델 불러와 Forward PASS
# ─────────────────────────────────────────────────────────────────────────
from model.lstm_model import BasicLSTM

hidden_dim = 64
num_layers = 1
lstm_model = BasicLSTM(num_nodes=N, input_dim=C_in, hidden_dim=hidden_dim, num_layers=num_layers)
lstm_model.eval()

with torch.no_grad():
    y_pred_lstm = lstm_model(x_rand)  # (B, N, 8)

print("LSTM Forward PASS 성공!")
print(f"  입력 x_rand shape: {x_rand.shape} → 출력 y_pred_lstm shape: {y_pred_lstm.shape}")
#   예상 출력: (2, 1370, 8)

# ─────────────────────────────────────────────────────────────────────────
# (비교를 위해) ST‐GCN, 우리 모델도 순서대로 체크 (다음 섹션 참조)
# ─────────────────────────────────────────────────────────────────────────
try:
    from model.stgcn_model import STGCN
    # ST‐GCN 입력 shape: (B, C, T, N) ; C=9 채널(큐+스피드+weekend) 등
    C_stgcn = 9
    x_stgcn = torch.randn(B, C_stgcn, T, N)
    # 인접행렬 A 임시: 단위행렬(N,N)
    A_dummy = torch.eye(N)
    stgcn_model = STGCN(in_channels=C_stgcn, out_channels=8, num_nodes=N, A=A_dummy)
    with torch.no_grad():
        y_pred_stgcn = stgcn_model(x_stgcn)
    print("ST‐GCN Forward PASS 성공!")
    print(f"  입력 x_stgcn shape: {x_stgcn.shape} → 출력 y_pred_stgcn shape: {y_pred_stgcn.shape}")
    # 예상 출력: (2, 8, 1370)
except Exception as e:
    print("ST‐GCN Forward PASS 오류:", e)

try:
    from model.res_stgcn_model import ResSTGCN
    # ResSTGCN 입력: (B, C, T, N) ; C=9 채널
    x_res = x_stgcn.clone()
    res_model = ResSTGCN(in_channels=C_stgcn, out_channels=8, num_nodes=N, A=A_dummy)
    with torch.no_grad():
        y_pred_res = res_model(x_res)
    print("우리 모델(Res+ST‐GCN) Forward PASS 성공!")
    print(f"  입력 x_res shape: {x_res.shape} → 출력 y_pred_res shape: {y_pred_res.shape}")
    # 예상 출력: (2, 8, 1370)
except Exception as e:
    print("우리 모델 Forward PASS 오류:", e)