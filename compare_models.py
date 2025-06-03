#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_models.py

세 모델 비교 분석 스크립트
─────────────────────────────────────────────────────────────────────────────────────
전통적 LSTM, 원본 ST‐GCN, ResSTGCN 세 모델을 동일한 테스트셋(Window 단위)에서 평가하고,
다음 항목들을 자동으로 생성해 줍니다:

  1) 정량적 지표
     - 전역(Global) 오차: MSE, RMSE, MAE
     - 채널(Channel) 오차: 채널별 MSE, RMSE, MAE
     - 모델별 총 추론(Inference) 시간
  2) 정성적 시각화
     - 채널별 RMSE 비교 바 차트 (모델별 색상 대비)
     - 모델별 오차 분포 히스토그램
     - 특정 노드(예: node_idx=42)의 샘플 채널 시계열 비교
  3) 결과 파일 저장
     - metrics_global.csv  : 전역 지표
     - metrics_channel.csv : 채널별 지표
     - channel_rmse.png     : 채널별 RMSE 비교 그래프
     - error_histogram.png  : 모델별 오차 분포 히스토그램
     - sample_node_42_timeseries.png : 노드 42번 채널0 시계열 비교
     - inference_time.txt   : 모델별 총 추론 시간

【설계 근거】
─────────────────────────────────────────────────────────────────────────────────────
1) **데이터 일관성 확보**  
   - 세 모델 모두 동일한 테스트셋 윈도우를 사용하여 평가해야만 공정한 비교가 가능.  
   - `data_loader.get_dataloaders()` 로드 시 Train/Val/Test이 순서대로 분리되어 있으므로,
     Test Loader에서 추출된 `(x_batch, y_batch, idx_batch)` 을 세 모델이 같은 순서로 평가에 사용.

2) **입력 형태 통일**  
   - **LSTM**: 학습 단계에서 “queue+speed” 8개 채널만 사용. 따라서 `x_batch[:, :8, :, :]` 을 `(B,12,1370,8)` 형태로 변환 후 모델 인퍼런스.  
   - **ST‐GCN**: “queue+speed+weekend(1)” = 총 9개 채널을 `(B,9,12,1370)` 형태로 모델에 바로 전달.  
   - **ResSTGCN**: ST‐GCN 분기를 사용해 `(B,9,12,1370)` → `(B,8,1370)` 예측을 얻은 뒤, “잔차 시퀀스(Residual Sequence)”를 구현하여 `(B,12,1370,8)` 형태로 ResLSTM에 전달하고 보정값을 더해 최종 예측.  
   - **왜 이렇게?**:  
     - LSTM은 그래프 연결 정보를 전혀 사용하지 않는 전통적 시퀀셜 RNN. 오직 노드별 시계열만 사용.  
     - ST‐GCN은 그래프 구조를 활용해 “공간+시간 패턴”을 동시에 학습.  
     - ResSTGCN은 단기 예측(ST‐GCN) + 장기 추세 보정(ResLSTM)을 결합 → 단독 ST‐GCN이 놓치는 “하루 주기 패턴” 등을 추가로 잡아낼 수 있음.  

3) **지표 선택**  
   - MSE(Mean Squared Error), RMSE(Root MSE), MAE(Mean Absolute Error)는 시계열 예측 논문에서 기본적으로 쓰이는 지표.  
   - 전역(Global) 지표를 통해 전체 예측력을 한눈에 파악하고, 채널(Channel) 지표를 통해 “어떤 채널(예: 승용차 queue, 버스 speed 등)에서 성능 차이가 큰지” 분석.  
   - 오차 분포 히스토그램은 “분포가 좁을수록 안정적인 예측”임을 시각적으로 보여 주기 위함.  
   - 특정 노드(예: 42번) 시계열 비교는 “실제 흐름 대략 얼마나 근접한지”를 직관적으로 보여 주기 위함.  

4) **Inference 시간 측정**  
   - 모델 성능과 함께 “실행 효율성(inference speed)”도 비교해야 실무 적용 시 적합성을 논문에서 어필 가능.  
   - 배치 단위로 `time.time()` 을 사용해 총 테스트셋에 대해 누적된 추론 시간을 기록.  

5) **파일 저장 구조**  
   - 모든 결과 파일을 자동으로 `compare_results/` 하위에 저장  
   - CSV와 PNG, TXT 파일을 일괄 생성 → 논문에 곧바로 활용 가능한 형태  

6) **주의사항**  
   - 체크포인트에서 사용한 **모델 구조와 하이퍼파라미터**(채널 수, 히든 크기 등)가 이 스크립트와 완전히 일치해야 함.  
   - GPU 메모리가 부족하다면 `--batch` 값을 줄여야 함.  
   - macOS 환경에서 **한글 시각화**가 필요하면 `plt.rcParams['font.family']='AppleGothic'` 등을 설정할 것.  
   - **입출력 채널**을 잘라내거나 포함하는 부분을 절대로 실수하면 안 됨(LSTM: `x[:, :8]` / STGCN·ResSTGCN: `x[:, :9]`).  

─────────────────────────────────────────────────────────────────────────────────────

"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 한글 폰트 설정 (필요시)
# plt.rcParams['font.family'] = 'AppleGothic'   # macOS 예시
# plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 예시

from data_loader import get_dataloaders

# 모델 임포트
from model.lstm_model import BasicLSTM
from model.stgcn_model import STGCN
from model.res_stgcn_model import ResSTGCN

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 0. ArgumentParser 설정                                                    │
# └──────────────────────────────────────────────────────────────────────────┘
parser = argparse.ArgumentParser(description="Compare LSTM, ST-GCN, ResSTGCN on Test Set")
parser.add_argument('--batch', type=int, default=4, help='Batch size for evaluation')
parser.add_argument('--ckpt_lstm', type=str, required=True, help='Path to BasicLSTM checkpoint (.pt)')
parser.add_argument('--ckpt_stgcn', type=str, required=True, help='Path to STGCN checkpoint (.pt)')
parser.add_argument('--ckpt_resstgcn', type=str, required=True, help='Path to ResSTGCN checkpoint (.pt)')
parser.add_argument('--output_dir', type=str, default='compare_results', help='Directory to save results')
args = parser.parse_args()

# 저장 폴더 생성
SAVE_DIR = args.output_dir
os.makedirs(SAVE_DIR, exist_ok=True)

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 1. Device 설정                                                             │
# └──────────────────────────────────────────────────────────────────────────┘
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"▶ Using device: {device}\n")

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 2. DataLoader 생성                                                         │
# └──────────────────────────────────────────────────────────────────────────┘
# get_dataloaders() → train/val/test DataLoader 반환. 여기서는 test_loader만 사용
# num_workers=2: 멀티프로세스 읽기, pin_memory=True: GPU 고속전송 지원
train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch,
                                                        num_workers=2,
                                                        pin_memory=True)

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 3. 모델 정의 및 체크포인트 로드                                             │
# └──────────────────────────────────────────────────────────────────────────┘
num_nodes = 1370      # lane 수
# ──────────────────────────────────────────────────────────────────────────
# (1) Basic LSTM 모델
# ──────────────────────────────────────────────────────────────────────────
# 입력 차원: 8 (queue4 + speed4)
input_dim_lstm = 8
# 은닉 차원과 레이어 수는 학습할 때 사용한 코드와 동일하게 설정해야 함
lstm_model = BasicLSTM(num_nodes=num_nodes, input_dim=input_dim_lstm,
                       hidden_dim=64, num_layers=1, dropout=0.0).to(device)

# Checkpoint 로드
if not os.path.exists(args.ckpt_lstm):
    raise FileNotFoundError(f"BasicLSTM 체크포인트를 찾을 수 없습니다: {args.ckpt_lstm}")
ckpt_lstm = torch.load(args.ckpt_lstm, map_location=device)
lstm_model.load_state_dict(ckpt_lstm['model_state_dict'])
lstm_model.eval()
print(f"▶ Loaded BasicLSTM checkpoint from {args.ckpt_lstm}")

# ──────────────────────────────────────────────────────────────────────────
# (2) ST-GCN 모델
# ──────────────────────────────────────────────────────────────────────────
# 입력 채널: 9 (queue4 + speed4 + weekend_flag)
in_ch_stgcn = 9
out_ch_stgcn = 8
# 정규화된 인접행렬 A_lane.npy 로드
adj_norm_path = os.path.join('3_tensor', 'adjacency', 'A_lane.npy')
if not os.path.exists(adj_norm_path):
    raise FileNotFoundError(f"A_lane.npy를 찾을 수 없습니다: {adj_norm_path}")
A_norm = np.load(adj_norm_path)            # shape=(1370,1370), 이미 정규화된 상태
A_norm = torch.from_numpy(A_norm).float().to(device)
stgcn_model = STGCN(in_channels=in_ch_stgcn, out_channels=out_ch_stgcn,
                    num_nodes=num_nodes, A=A_norm).to(device)

if not os.path.exists(args.ckpt_stgcn):
    raise FileNotFoundError(f"STGCN 체크포인트를 찾을 수 없습니다: {args.ckpt_stgcn}")
ckpt_stgcn = torch.load(args.ckpt_stgcn, map_location=device)
stgcn_model.load_state_dict(ckpt_stgcn['model_state_dict'])
stgcn_model.eval()
print(f"▶ Loaded ST-GCN checkpoint from {args.ckpt_stgcn}")

# ──────────────────────────────────────────────────────────────────────────
# (3) ResSTGCN 모델
# ──────────────────────────────────────────────────────────────────────────
# 입력 채널: 9 (queue4 + speed4 + weekend_flag)
in_ch_res = 9
out_ch_res = 8
# ResSTGCN 내부에서 ST-GCN 분기와 Residual LSTM 분기를 모두 포함
res_model = ResSTGCN(in_channels=in_ch_res, out_channels=out_ch_res,
                     num_nodes=num_nodes, A=A_norm, hidden_dim=256).to(device)

if not os.path.exists(args.ckpt_resstgcn):
    raise FileNotFoundError(f"ResSTGCN 체크포인트를 찾을 수 없습니다: {args.ckpt_resstgcn}")
ckpt_res = torch.load(args.ckpt_resstgcn, map_location=device)
# ResSTGCN의 파라미터는 두 개(state_dict) 로 분리 저장됨
res_model.stgcn.load_state_dict(ckpt_res['stgcn_state_dict'])
res_model.reslstm.load_state_dict(ckpt_res['reslstm_state_dict'])
res_model.eval()
print(f"▶ Loaded ResSTGCN checkpoint from {args.ckpt_resstgcn}\n")

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 4. 평가 함수(evaluate_model) 정의                                            │
# └──────────────────────────────────────────────────────────────────────────┘
def evaluate_model(model: nn.Module, model_type: str):
    """
    model_type: 'lstm' | 'stgcn' | 'resstgcn'
    
    Args:
      model: PyTorch 모델 (BasicLSTM, STGCN, 또는 ResSTGCN)
      model_type: 문자열로 모델 타입 지정
    Returns:
      딕셔너리 containing:
        - global_mse, global_rmse, global_mae
        - channel: pandas.DataFrame (columns=['channel','MSE','RMSE','MAE'])
        - preds: np.ndarray, shape=(M,1370,8)  (M=윈도우 수)
        - trues: np.ndarray, shape=(M,1370,8)
        - inf_time: 총 inference 시간(초)
    """
    all_preds = []
    all_trues = []
    total_time = 0.0
    mse_loss = nn.MSELoss(reduction='mean')

    # 평가 모드
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch, idx_batch in test_loader:
            x_batch = x_batch.to(device)  # shape=(B,C,12,1370)
            y_batch = y_batch.to(device)  # shape=(B,8,1370)
            B = x_batch.size(0)

            if model_type == 'lstm':
                # 전통적 LSTM: 입력 채널 8개만 사용 → (B,12,1370,8)
                x_in = x_batch[:, :8, :, :].permute(0, 2, 3, 1).contiguous()
                # 평가 시간 측정 시작
                start = time.time()
                y_pred = model(x_in)             # shape=(B,1370,8)
                elapsed = time.time() - start
                # LSTM 출력은 (B,1370,8) → ST/RES는 (B,8,1370) 형태이므로 통일
                y_pred = y_pred.permute(0, 2, 1)  # (B,8,1370)

            elif model_type == 'stgcn':
                # ST-GCN: (B,9,12,1370) 그대로 전달
                start = time.time()
                y_pred = model(x_batch)         # (B,8,1370)
                elapsed = time.time() - start

            else:  # model_type == 'resstgcn'
                # ResSTGCN: ST-GCN 분기 → 잔차 보정
                start = time.time()
                # (1) ST-GCN 단기 예측 (마지막 스텝)
                y_pred_st = model.stgcn(x_batch)  # (B,8,1370)

                # (2) 잔차 생성: 간소화 버전으로 “마지막 스텝 잔차”만 사용
                #    ① res_seq 초기화
                res_seq = torch.zeros((B, 12, num_nodes, 8), device=device, dtype=torch.float32)
                #    ② y_true_last: (B,1370,8), y_pred_last: (B,1370,8)
                y_true_last = y_batch.permute(0, 2, 1)
                y_pred_last = y_pred_st.permute(0, 2, 1)
                #    ③ 마지막(time=11) 잔차 삽입
                res_seq[:, 11, :, :] = y_true_last - y_pred_last

                # (3) 주말 플래그 추출: x_batch 채널 8번은 weekend_flag
                weekend_flag = x_batch[:, 8, 0, 0]  # shape=(B,)
                # (4) Residual LSTM 보정
                res_corr = model.reslstm(res_seq, weekend_flag)  # shape=(B,8,1370)
                # (5) 최종 예측
                y_pred = y_pred_st + res_corr  # (B,8,1370)

                elapsed = time.time() - start

            total_time += elapsed

            # GPU→CPU→NumPy로 변환 후 누적
            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y_batch.cpu().numpy())

    # 리스트 → NumPy 배열
    preds = np.concatenate(all_preds, axis=0)  # shape=(M,8,1370)
    trues = np.concatenate(all_trues, axis=0)  # shape=(M,8,1370)
    diff = preds - trues                     # shape=(M,8,1370)

    # ─────────────────────────────────────────────────────────────────────────
    # 전역(Global) 지표
    # ─────────────────────────────────────────────────────────────────────────
    mse_global = np.mean(diff**2)
    rmse_global = np.sqrt(mse_global)
    mae_global = np.mean(np.abs(diff))

    # ─────────────────────────────────────────────────────────────────────────
    # 채널(Channel)별 지표
    #   axis=(0,2): 윈도우 축(0)과 노드 축(2) 평균
    # ─────────────────────────────────────────────────────────────────────────
    ch_mse = np.mean(diff**2, axis=(0, 2))   # (8,)
    ch_rmse = np.sqrt(ch_mse)
    ch_mae = np.mean(np.abs(diff), axis=(0, 2))

    channel_df = pd.DataFrame({
        'channel': [f'ch{c}' for c in range(8)],
        'MSE': ch_mse,
        'RMSE': ch_rmse,
        'MAE': ch_mae
    })

    return {
        'global_mse': mse_global,
        'global_rmse': rmse_global,
        'global_mae': mae_global,
        'channel': channel_df,
        # preds/trues 반환 형태: (M,8,1370) → 시각화 시 노드별(1370) 접근 편의 위해 (M,1370,8)으로 transpose
        'preds': preds.transpose(0, 2, 1),
        'trues': trues.transpose(0, 2, 1),
        'inf_time': total_time
    }

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 5. 세 모델 평가                                                       │
# └──────────────────────────────────────────────────────────────────────────┘
results = {}

print("▶ Evaluating Basic LSTM ...")
results['lstm'] = evaluate_model(lstm_model, 'lstm')
print(f"    ▶ Basic LSTM total inference time: {results['lstm']['inf_time']:.3f}s\n")

print("▶ Evaluating ST-GCN ...")
results['stgcn'] = evaluate_model(stgcn_model, 'stgcn')
print(f"    ▶ ST-GCN total inference time: {results['stgcn']['inf_time']:.3f}s\n")

print("▶ Evaluating ResSTGCN ...")
results['resstgcn'] = evaluate_model(res_model, 'resstgcn')
print(f"    ▶ ResSTGCN total inference time: {results['resstgcn']['inf_time']:.3f}s\n")

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 6. 정량적 지표 CSV 저장                                                   │
# └──────────────────────────────────────────────────────────────────────────┘
# (a) 전역 지표 DataFrame 생성
rows = []
for key in ['lstm','stgcn','resstgcn']:
    r = results[key]
    rows.append({
        'model': key,
        'global_MSE': r['global_mse'],
        'global_RMSE': r['global_rmse'],
        'global_MAE': r['global_mae']
    })
global_df = pd.DataFrame(rows)

# (b) 채널별 지표 DataFrame (모델 컬럼 추가)
chan_dfs = []
for key in ['lstm','stgcn','resstgcn']:
    df = results[key]['channel'].copy()
    df.insert(0, 'model', key)
    chan_dfs.append(df)
channel_df = pd.concat(chan_dfs, ignore_index=True)

# (c) CSV 저장
global_df.to_csv(os.path.join(SAVE_DIR,'metrics_global.csv'), index=False)
channel_df.to_csv(os.path.join(SAVE_DIR,'metrics_channel.csv'), index=False)
print(f"▶ Saved metrics CSV files → {SAVE_DIR}/metrics_global.csv, metrics_channel.csv\n")

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 7. 시각화                                                               │
# └──────────────────────────────────────────────────────────────────────────┘
# 7.1 채널별 RMSE 바 차트
plt.figure(figsize=(8, 5))
x = np.arange(8)
width = 0.25
for i, key in enumerate(['lstm','stgcn','resstgcn']):
    rmse = results[key]['channel']['RMSE'].values
    plt.bar(x + i*width, rmse, width=width, label=key.upper())
plt.xticks(x + width, [f'ch{c}' for c in range(8)])
plt.xlabel('Channel (queue4 + speed4)')
plt.ylabel('RMSE')
plt.title('Channel-wise RMSE Comparison')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,'channel_rmse.png'))
plt.close()
print(f"▶ Saved: {SAVE_DIR}/channel_rmse.png")

# 7.2 모델별 오차 분포 히스토그램
# preds-trues를 flatten 한 뒤 axis=1으로 합쳐서 (num_samples, 3) 배열 생성
all_diff = np.stack([
    (results['lstm']['preds'] - results['lstm']['trues']).ravel(),
    (results['stgcn']['preds'] - results['stgcn']['trues']).ravel(),
    (results['resstgcn']['preds'] - results['resstgcn']['trues']).ravel()
], axis=1)  # shape = (M*1370*8, 3)

plt.figure(figsize=(8, 5))
plt.hist(all_diff, bins=100, label=['LSTM','STGCN','ResSTGCN'], alpha=0.6, density=True)
plt.xlabel('Prediction Error')
plt.ylabel('Density')
plt.title('Error Distribution Comparison')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,'error_histogram.png'))
plt.close()
print(f"▶ Saved: {SAVE_DIR}/error_histogram.png")

# 7.3 샘플 노드 시계열 비교 (예: node_idx=42, 채널0)
node_idx = 42
# 전체 테스트 윈도우 수 M
M = results['lstm']['preds'].shape[0]
# 시각화할 샘플 개수 (최대 200)
num_examples = min(200, M)
t = np.arange(num_examples)

plt.figure(figsize=(10, 6))
# 실제값
plt.plot(t, results['lstm']['trues'][:num_examples, node_idx, 0], 'k-', label='True')
# LSTM 예측
plt.plot(t, results['lstm']['preds'][:num_examples, node_idx, 0], 'r--', label='LSTM')
# ST-GCN 예측
plt.plot(t, results['stgcn']['preds'][:num_examples, node_idx, 0], 'b--', label='STGCN')
# ResSTGCN 예측
plt.plot(t, results['resstgcn']['preds'][:num_examples, node_idx, 0], 'g--', label='ResSTGCN')
plt.xlabel('Test Window Index')
plt.ylabel('Channel 0 Value (예: 승용차 Queue)')
plt.title(f'Node {node_idx} Channel0 Time Series (First {num_examples} Windows)')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,f'sample_node_{node_idx}_timeseries.png'))
plt.close()
print(f"▶ Saved: {SAVE_DIR}/sample_node_{node_idx}_timeseries.png\n")

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 8. Inference 시간 기록                                                     │
# └──────────────────────────────────────────────────────────────────────────┘
with open(os.path.join(SAVE_DIR,'inference_time.txt'), 'w') as f:
    for key in ['lstm','stgcn','resstgcn']:
        f.write(f"{key.upper()} total inference time (s): {results[key]['inf_time']:.3f}\n")
print(f"▶ Saved: {SAVE_DIR}/inference_time.txt\n")

print("▶ All done. Compare results are saved in '{}'".format(SAVE_DIR))