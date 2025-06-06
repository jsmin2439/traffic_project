#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_models_denorm_from_pkl.py

세 모델 비교 분석 스크립트 (normalized_tensor_{DATE}_with_weekend.pkl을 사용한 역정규화)

─────────────────────────────────────────────────────────────────────────────────────
전통적 LSTM, 원본 ST‐GCN, ResSTGCN 세 모델을 동일한 테스트셋(Window 단위)에서 평가하고,
“normalized_tensor_{DATE}_with_weekend.pkl” 파일과 “input_tensor_{DATE}.pkl”을 활용하여
모든 지표를 **원래 스케일(실제 대기 큐 & 속도)**로 역정규화한 결과를 계산·시각화합니다.

사용법:
    python compare_models_denorm_from_pkl.py \
      --batch 4 \
      --ckpt_lstm path/to/lstm_checkpoint.pt \
      --ckpt_stgcn path/to/stgcn_checkpoint.pt \
      --ckpt_resstgcn path/to/resstgcn_checkpoint.pt \
      --output_dir compare_results_denorm_pkl \
      --target_date 20220810  # 특정 날짜만 평가하려면 지정, 전체 평가하려면 생략 \
      --tensor_dir 3_tensor
─────────────────────────────────────────────────────────────────────────────────────
"""
import os
import time
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_loader import get_dataloaders
import glob

# 한글 폰트 설정 (필요시)
# plt.rcParams['font.family'] = 'AppleGothic'   # macOS 예시
# plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 예시


# 모델 임포트
from model.lstm_model import BasicLSTM
from model.stgcn_model import STGCN
from model.res_stgcn_model import ResSTGCN

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 0. ArgumentParser 설정                                                    │
# └──────────────────────────────────────────────────────────────────────────┘
parser = argparse.ArgumentParser(
    description="Compare LSTM, ST-GCN, ResSTGCN on Test Set (Denormalize from PKL)"
)
parser.add_argument('--batch', type=int, default=4, help='Batch size for evaluation')
parser.add_argument('--ckpt_lstm', type=str, required=True, help='Path to BasicLSTM checkpoint (.pt)')
parser.add_argument('--ckpt_stgcn', type=str, required=True, help='Path to STGCN checkpoint (.pt)')
parser.add_argument('--ckpt_resstgcn', type=str, required=True, help='Path to ResSTGCN checkpoint (.pt)')
parser.add_argument('--output_dir', type=str, default='compare_results_denorm_pkl', help='Directory to save results')
parser.add_argument('--target_date', type=int, default=None,
                    help='특정 날짜(YYYYMMDD)만 평가하려면 지정 (예: 20220810). 지정하지 않으면 전체 윈도우 평가')
parser.add_argument('--tensor_dir', type=str, default='3_tensor',
                    help='normalized_tensor 및 input_tensor가 위치한 상위 디렉토리 (기본: 3_tensor)')
# ---- extra arguments for comparison/epoch evolution ----
parser.add_argument('--compare_node', type=int, default=42,
                    help='노드 ID for per‑channel time‑series comparison')
parser.add_argument('--epoch_list', type=str, default=None,
                    help='Comma‑separated epoch numbers for “metric‑vs‑epoch” plots. '
                         'If omitted, epoch evolution plots are skipped.')
parser.add_argument('--ckpt_tpl_lstm', type=str, default=None,
                    help='Template for LSTM checkpoints, e.g. checkpoints_lstm/lstm_epoch{epoch:04d}.pt')
parser.add_argument('--ckpt_tpl_stgcn', type=str, default=None,
                    help='Template for ST‑GCN checkpoints')
parser.add_argument('--ckpt_tpl_resstgcn', type=str, default=None,
                    help='Template for ResSTGCN checkpoints')
args = parser.parse_args()

# Parse epoch list if provided
epoch_nums = []
if args.epoch_list:
    epoch_nums = [int(e.strip()) for e in args.epoch_list.split(',') if e.strip().isdigit()]

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 1. 출력 폴더 생성                                                         │
# └──────────────────────────────────────────────────────────────────────────┘
SAVE_DIR = args.output_dir
os.makedirs(SAVE_DIR, exist_ok=True)

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 2. Device 설정                                                             │
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
# │ 3. DataLoader 생성                                                         │
# └──────────────────────────────────────────────────────────────────────────┘
train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch,
                                                        num_workers=2,
                                                        pin_memory=True)

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 4. 모델 정의 및 체크포인트 로드                                             │
# └──────────────────────────────────────────────────────────────────────────┘
num_nodes = 1370  # lane 수

# (1) Basic LSTM 모델
input_dim_lstm = 8  # LSTM 입력은 queue4+speed4 (정규화된 채널 8개)
lstm_model = BasicLSTM(num_nodes=num_nodes, input_dim=input_dim_lstm,
                       hidden_dim=64, num_layers=1, dropout=0.0).to(device)
if not os.path.exists(args.ckpt_lstm):
    raise FileNotFoundError(f"BasicLSTM 체크포인트를 찾을 수 없습니다: {args.ckpt_lstm}")
ckpt_lstm = torch.load(args.ckpt_lstm, map_location=device)
lstm_model.load_state_dict(ckpt_lstm['model_state_dict'])
lstm_model.eval()
print(f"▶ Loaded BasicLSTM checkpoint from {args.ckpt_lstm}")

# (2) ST-GCN 모델
in_ch_stgcn = 9   # queue4 + speed4 + weekend_flag
out_ch_stgcn = 8  # queue4 + speed4
adj_norm_path = os.path.join(args.tensor_dir, 'adjacency', 'A_lane.npy')
if not os.path.exists(adj_norm_path):
    raise FileNotFoundError(f"A_lane.npy를 찾을 수 없습니다: {adj_norm_path}")
A_norm = np.load(adj_norm_path)
A_norm = torch.from_numpy(A_norm).float().to(device)
stgcn_model = STGCN(in_channels=in_ch_stgcn, out_channels=out_ch_stgcn,
                    num_nodes=num_nodes, A=A_norm).to(device)
if not os.path.exists(args.ckpt_stgcn):
    raise FileNotFoundError(f"STGCN 체크포인트를 찾을 수 없습니다: {args.ckpt_stgcn}")
ckpt_stgcn = torch.load(args.ckpt_stgcn, map_location=device)
stgcn_model.load_state_dict(ckpt_stgcn['model_state_dict'])
stgcn_model.eval()
print(f"▶ Loaded ST-GCN checkpoint from {args.ckpt_stgcn}")

# (3) ResSTGCN 모델
in_ch_res = 9    # queue4 + speed4 + weekend_flag
out_ch_res = 8   # queue4 + speed4
res_model = ResSTGCN(in_channels=in_ch_res, out_channels=out_ch_res,
                     num_nodes=num_nodes, A=A_norm, hidden_dim=256).to(device)
if not os.path.exists(args.ckpt_resstgcn):
    raise FileNotFoundError(f"ResSTGCN 체크포인트를 찾을 수 없습니다: {args.ckpt_resstgcn}")
ckpt_res = torch.load(args.ckpt_resstgcn, map_location=device)
res_model.stgcn.load_state_dict(ckpt_res['stgcn_state_dict'])
res_model.reslstm.load_state_dict(ckpt_res['reslstm_state_dict'])
res_model.eval()
print(f"▶ Loaded ResSTGCN checkpoint from {args.ckpt_resstgcn}\n")

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 5. 날짜별 평균/표준편차(채널 0~7) 사전 계산                                    │
# └──────────────────────────────────────────────────────────────────────────┘
"""
우리가 가진 역정규화를 위한 통계는 두 가지 소스 중 하나에서 읽어온다.

1) **normalized_tensor_{DATE}_with_weekend.pkl**  
   - 전처리 단계에서 채널‑별 mean / std 를 dict 형태로 같이 저장해 두었다면
     pickle 로드 후 `dict["mean"]`, `dict["std"]` 로 바로 사용한다.

2) 위 파일이 없거나 통계가 없다면 **input_tensor_{DATE}.pkl** 을 불러와
   직접 채널 0~7 의 mean / std 를 계산한다.

채널 0~7 은 queue4 + speed4 를 뜻한다.
"""
date_to_stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}

WINDOWS_DIR = os.path.join(args.tensor_dir, 'windows')
date_npy_path = os.path.join(WINDOWS_DIR, 'all_DATE.npy')
if not os.path.exists(date_npy_path):
    raise FileNotFoundError(f"all_DATE.npy not found at {date_npy_path}")
all_date_array = np.load(date_npy_path)            # (M,)
for d in np.unique(all_date_array):
    d = int(d)
    day_dir = os.path.join(args.tensor_dir, str(d))

    # 1) 시도: normalized_tensor pkl 안에 통계가 있는 경우
    norm_pkl = os.path.join(day_dir, f'normalized_tensor_{d}_with_weekend.pkl')
    mean_arr = std_arr = None
    if os.path.exists(norm_pkl):
        with open(norm_pkl, 'rb') as f:
            obj = pickle.load(f)
        # 통계를 dict 로 저장했다면
        if isinstance(obj, dict) and 'mean' in obj and 'std' in obj:
            mean_arr = np.asarray(obj['mean'][:8], dtype=np.float32)   # (8,)
            std_arr  = np.asarray(obj['std'][:8],  dtype=np.float32)   # (8,)

    # 2) fallback: input_tensor에서 직접 계산
    if mean_arr is None or std_arr is None:
        inp_pkl = os.path.join(day_dir, f'input_tensor_{d}.pkl')
        if not os.path.exists(inp_pkl):
            raise FileNotFoundError(f"Neither stats in norm‑pkl nor {inp_pkl} exist")
        with open(inp_pkl, 'rb') as f:
            inp = pickle.load(f)                         # (288,1370,>=8)
        mean_arr = np.mean(inp[:, :, :8], axis=(0, 1)).astype(np.float32)
        std_arr  = np.std(inp[:, :, :8],  axis=(0, 1)).astype(np.float32)

    date_to_stats[d] = (mean_arr, std_arr)

print("▶ Loaded per‑date mean/std for", len(date_to_stats), "dates\n")

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 6. 모델 평가 함수 정의                                                     │
# └──────────────────────────────────────────────────────────────────────────┘
def evaluate_model(model: nn.Module, model_type: str, target_date: int = None):
    """
    model_type: 'lstm' | 'stgcn' | 'resstgcn'
    
    Args:
      model: PyTorch 모델
      model_type: 문자열로 모델 타입 지정
      target_date: 특정 날짜(YYYYMMDD)만 평가하려면 지정. None이면 전체 평가.
    Returns:
      dict containing:
        - preds_norm: np.ndarray, shape=(M_sel, 1370, 8)  (정규화된 예측)
        - trues_norm: np.ndarray, shape=(M_sel, 1370, 8)
        - dates_sel:   np.ndarray, shape=(M_sel,)       (윈도우별 DATE)
        - inf_time: 총 inference 시간(초)
    """
    all_preds = []
    all_trues = []
    all_dates = []
    total_time = 0.0

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch, idx_batch, date_batch in test_loader:
            # date_batch: shape=(B,), dtype=int
            if target_date is not None:
                mask = (date_batch == target_date)
                if mask.sum().item() == 0:
                    continue
                x_batch = x_batch[mask]
                y_batch = y_batch[mask]
                date_batch = date_batch[mask]

            B = x_batch.size(0)
            if B == 0:
                continue

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            if model_type == 'lstm':
                # 전통적 LSTM: normalized 입력 채널 8개만 사용
                x_in = x_batch[:, :8, :, :].permute(0, 2, 3, 1).contiguous()
                x_in = x_in.to(device)
                start = time.time()
                y_pred = model(x_in)             # (B,1370,8)
                elapsed = time.time() - start
                y_pred = y_pred.permute(0, 2, 1)  # (B,8,1370)

            elif model_type == 'stgcn':
                start = time.time()
                y_pred = model(x_batch)          # (B,8,1370)
                elapsed = time.time() - start

            else:  # 'resstgcn'
                start = time.time()
                # (1) ST-GCN 단기 예측
                y_pred_st = model.stgcn(x_batch)  # (B,8,1370)

                # (2) 잔차 생성 (마지막 스텝만 사용)
                res_seq = torch.zeros((B, 12, num_nodes, 8), device=device, dtype=torch.float32)
                y_true_last = y_batch.permute(0, 2, 1)   # (B,1370,8)
                y_pred_last = y_pred_st.permute(0, 2, 1)  # (B,1370,8)
                res_seq[:, 11, :, :] = y_true_last - y_pred_last

                # (3) weekend flag
                weekend_flag = x_batch[:, 8, 0, 0]  # (B,)
                # (4) Residual LSTM 보정
                res_corr = model.reslstm(res_seq, weekend_flag)  # (B,8,1370)
                # (5) 최종 예측
                y_pred = y_pred_st + res_corr  # (B,8,1370)
                elapsed = time.time() - start

            total_time += elapsed
            all_preds.append(y_pred.cpu().numpy())       # (B,8,1370)
            all_trues.append(y_batch.cpu().numpy())      # (B,8,1370)
            all_dates.append(date_batch.numpy())         # (B,)

    # concatenate
    if len(all_preds) == 0:
        return {
            'preds_norm': np.empty((0, 1370, 8), dtype=np.float32),
            'trues_norm': np.empty((0, 1370, 8), dtype=np.float32),
            'dates_sel':  np.empty((0,), dtype=int),
            'inf_time': total_time
        }

    preds_norm = np.concatenate(all_preds, axis=0)  # shape=(M_sel, 8, 1370)
    trues_norm = np.concatenate(all_trues, axis=0)  # shape=(M_sel, 8, 1370)
    dates_sel  = np.concatenate(all_dates,  axis=0)  # shape=(M_sel,)

    # (8,1370) → (1370,8)
    preds_norm = preds_norm.transpose(0, 2, 1)  # (M_sel,1370,8)
    trues_norm = trues_norm.transpose(0, 2, 1)  # (M_sel,1370,8)

    return {
        'preds_norm': preds_norm,
        'trues_norm': trues_norm,
        'dates_sel': dates_sel,
        'inf_time': total_time
    }

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 7. 세 모델 평가                                                           │
# └──────────────────────────────────────────────────────────────────────────┘
results = {}

print("▶ Evaluating Basic LSTM ...")
results['lstm'] = evaluate_model(lstm_model, 'lstm', args.target_date)
print(f"    ▶ Basic LSTM total inference time: {results['lstm']['inf_time']:.3f}s\n")

print("▶ Evaluating ST-GCN ...")
results['stgcn'] = evaluate_model(stgcn_model, 'stgcn', args.target_date)
print(f"    ▶ ST-GCN total inference time: {results['stgcn']['inf_time']:.3f}s\n")

print("▶ Evaluating ResSTGCN ...")
results['resstgcn'] = evaluate_model(res_model, 'resstgcn', args.target_date)
print(f"    ▶ ResSTGCN total inference time: {results['resstgcn']['inf_time']:.3f}s\n")

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 8. 역정규화(Denormalize) 처리                                               │
# └──────────────────────────────────────────────────────────────────────────┘
for key in ['lstm', 'stgcn', 'resstgcn']:
    preds_n = results[key]['preds_norm']  # (M_sel,1370,8)
    trues_n = results[key]['trues_norm']  # (M_sel,1370,8)
    dates   = results[key]['dates_sel']   # (M_sel,)

    M_sel = preds_n.shape[0]
    preds_o = np.zeros_like(preds_n, dtype=np.float32)  # (M_sel,1370,8)
    trues_o = np.zeros_like(trues_n, dtype=np.float32)

    # 날짜별로 묶어서 연산
    for date in np.unique(dates):
        idxs = np.where(dates == date)[0]  # 해당 날짜에 속한 윈도우 인덱스 배열
        means, stds = date_to_stats[int(date)]  # (8,), (8,)

        # preds_n[idxs]: (N_date,1370,8)
        preds_o[idxs] = preds_n[idxs] * stds.reshape(1, 1, 8) + means.reshape(1, 1, 8)
        trues_o[idxs] = trues_n[idxs] * stds.reshape(1, 1, 8) + means.reshape(1, 1, 8)

    results[key]['preds_orig'] = preds_o
    results[key]['trues_orig'] = trues_o

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 9. 정량적 지표 CSV 저장 (Denormalized 기준)                                   │
# └──────────────────────────────────────────────────────────────────────────┘
rows = []
for key in ['lstm','stgcn','resstgcn']:
    pred_o = results[key]['preds_orig']  # (M_sel,1370,8)
    true_o = results[key]['trues_orig']  # (M_sel,1370,8)
    diff_o = pred_o - true_o             # (M_sel,1370,8)

    mse_global_o  = np.mean(diff_o**2)
    rmse_global_o = np.sqrt(mse_global_o)
    mae_global_o  = np.mean(np.abs(diff_o))
    # 추가 지표
    mape_global_o = np.mean(np.abs(diff_o) / (np.abs(true_o) + 1e-6)) * 100.0  # %
    ss_tot = np.sum((true_o - true_o.mean())**2)
    ss_res = np.sum(diff_o**2)
    r2_global_o   = 1.0 - ss_res / (ss_tot + 1e-12)
    rows.append({
        'model': key,
        'global_MSE': mse_global_o,
        'global_RMSE': rmse_global_o,
        'global_MAE': mae_global_o,
        'global_MAPE(%)': mape_global_o,
        'global_R2':      r2_global_o,
    })
global_df = pd.DataFrame(rows)

chan_dfs = []
for key in ['lstm','stgcn','resstgcn']:
    pred_o = results[key]['preds_orig']
    true_o = results[key]['trues_orig']
    diff_o = pred_o - true_o  # (M_sel,1370,8)

    ch_mse_o  = np.mean(diff_o**2, axis=(0,1))  # (8,)
    ch_rmse_o = np.sqrt(ch_mse_o)
    ch_mae_o  = np.mean(np.abs(diff_o), axis=(0,1))
    ch_mape_o = np.mean(np.abs(diff_o) / (np.abs(true_o) + 1e-6), axis=(0,1)) * 100.0
    # R2 per channel
    ss_tot_ch = np.sum((true_o - true_o.mean(axis=(0,1), keepdims=True))**2, axis=(0,1))
    ss_res_ch = np.sum(diff_o**2, axis=(0,1))
    ch_r2_o   = 1.0 - ss_res_ch / (ss_tot_ch + 1e-12)

    df = pd.DataFrame({
        'channel': [f'ch{c}' for c in range(8)],
        'MSE':  ch_mse_o,
        'RMSE': ch_rmse_o,
        'MAE':  ch_mae_o,
        'MAPE(%)': ch_mape_o,
        'R2':      ch_r2_o
    })
    df.insert(0, 'model', key)
    chan_dfs.append(df)
channel_df = pd.concat(chan_dfs, ignore_index=True)

global_df.to_csv(os.path.join(SAVE_DIR,'metrics_global_denorm.csv'), index=False)
channel_df.to_csv(os.path.join(SAVE_DIR,'metrics_channel_denorm.csv'), index=False)
print(f"▶ Saved metrics CSV (Denormalized) → {SAVE_DIR}/metrics_global_denorm.csv, metrics_channel_denorm.csv\n")

# --- extra plot: node‑wise RMSE (or MAE) for the chosen date -------------
if args.target_date is not None:
    plt.figure(figsize=(12,4))
    for i, key in enumerate(['lstm','stgcn','resstgcn']):
        # diff for selected date only
        mask = (results[key]['dates_sel'] == args.target_date)
        if mask.sum() == 0:           # no window for that date
            continue
        diff_date = (results[key]['preds_orig'][mask] -
                     results[key]['trues_orig'][mask])
        rmse_nodes = np.sqrt(np.mean(diff_date**2, axis=(0,2)))   # (1370,)
        plt.plot(rmse_nodes, label=key.upper(), linewidth=1.5)
    plt.xlabel('Node index')
    plt.ylabel('RMSE (denorm)')
    plt.title(f'Node‑wise RMSE on {args.target_date}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'node_rmse_models_{args.target_date}.png'))
    plt.close()
    print(f"▶ Saved: {SAVE_DIR}/node_rmse_models_{args.target_date}.png")

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 10. 정성적 시각화 (Denormalized 기준)                                        │
# └──────────────────────────────────────────────────────────────────────────┘

# 10.1 채널별 RMSE 바 차트
plt.figure(figsize=(8, 5))
x = np.arange(8)
width = 0.25
for i, key in enumerate(['lstm','stgcn','resstgcn']):
    rmse_o = channel_df[channel_df['model']==key]['RMSE'].values
    plt.bar(x + i*width, rmse_o, width=width, label=key.upper())
plt.xticks(x + width, [f'ch{c}' for c in range(8)])
plt.xlabel('Channel (Queue4 + Speed4)')
plt.ylabel('RMSE (원래 단위)')
plt.title('Channel-wise RMSE Comparison (Denormalized)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,'channel_rmse_denorm.png'))
plt.close()
print(f"▶ Saved: {SAVE_DIR}/channel_rmse_denorm.png")

# 10.1‑b 채널별 MAPE 바 차트
plt.figure(figsize=(8,5))
for i, key in enumerate(['lstm','stgcn','resstgcn']):
    mape_vals = channel_df[channel_df['model']==key]['MAPE(%)'].values
    plt.bar(x + i*width, mape_vals, width=width, label=key.upper())
plt.xticks(x + width, [f'ch{c}' for c in range(8)])
plt.xlabel('Channel')
plt.ylabel('MAPE (%)')
plt.title('Channel‑wise MAPE Comparison (Denormalized)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'channel_mape_denorm.png'))
plt.close()
print(f"▶ Saved: {SAVE_DIR}/channel_mape_denorm.png")

# 10.2 모델별 오차 분포 히스토그램 (Denormalized)
all_diff_o = np.stack([
    (results['lstm']['preds_orig'] - results['lstm']['trues_orig']).ravel(),
    (results['stgcn']['preds_orig'] - results['stgcn']['trues_orig']).ravel(),
    (results['resstgcn']['preds_orig'] - results['resstgcn']['trues_orig']).ravel()
], axis=1)  # shape = (M_sel*1370*8, 3)

plt.figure(figsize=(8, 5))
plt.hist(all_diff_o, bins=100, label=['LSTM','STGCN','ResSTGCN'], alpha=0.6, density=True)
plt.xlabel('Prediction Error (원래 단위)')
plt.ylabel('Density')
plt.title('Error Distribution Comparison (Denormalized)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,'error_histogram_denorm.png'))
plt.close()
print(f"▶ Saved: {SAVE_DIR}/error_histogram_denorm.png")

# 10.3 샘플 노드 시계열 비교 (Denormalized, per-channel, for chosen node)
node_idx = args.compare_node
for ch in range(8):
    plt.figure(figsize=(10,4))
    plt.plot(results['lstm']['trues_orig'][:, node_idx, ch], 'k-',  label='True')
    plt.plot(results['lstm']['preds_orig'][:, node_idx, ch], 'r--', label='LSTM')
    plt.plot(results['stgcn']['preds_orig'][:, node_idx, ch], 'b--', label='STGCN')
    plt.plot(results['resstgcn']['preds_orig'][:, node_idx, ch], 'g--', label='ResSTGCN')
    plt.xlabel('Test window idx')
    plt.ylabel(f'Channel {ch}')
    plt.title(f'Node {node_idx} – Channel {ch} (denorm)')
    plt.legend(fontsize=8)
    plt.tight_layout()
    fname = f'node{node_idx}_ch{ch}_timeseries_denorm.png'
    plt.savefig(os.path.join(SAVE_DIR, fname))
    plt.close()
print(f"▶ Saved per‑channel plots for node {node_idx}")

# 10.4 노드별 평균 RMSE 막대그래프 (ResSTGCN 기준, Denormalized)
diff_res_o = (results['resstgcn']['preds_orig'] - results['resstgcn']['trues_orig'])  # (M_sel,1370,8)
mse_res_nodes_o = np.mean(diff_res_o**2, axis=(0,2))  # (1370,)
rmse_res_nodes_o = np.sqrt(mse_res_nodes_o)           # (1370,)
plt.figure(figsize=(12, 4))
plt.bar(np.arange(num_nodes), rmse_res_nodes_o, color='tab:orange')
plt.xlabel('Node Index (0~1369)')
plt.ylabel('Node-wise RMSE\n(모든 채널 평균, 원래 단위)')
plt.title('ResSTGCN 노드별 평균 RMSE (Denormalized)')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,'node_rmse_resstgcn_denorm.png'))
plt.close()
print(f"▶ Saved: {SAVE_DIR}/node_rmse_resstgcn_denorm.png\n")

# 10.5 윈도우(Window)별 RMSE 변화 추이 선그래프 (Denormalized, 분할 출력)
window_rmse_o = {}
for key in ['lstm','stgcn','resstgcn']:
    diff_o = results[key]['preds_orig'] - results[key]['trues_orig']  # (M_sel,1370,8)
    rmse_per_window_o = np.sqrt(np.mean(diff_o**2, axis=(1,2)))     # (M_sel,)
    window_rmse_o[key] = rmse_per_window_o

windows_len = window_rmse_o['lstm'].shape[0]
ranges = [(0, 100), (101, 200), (200, min(287, windows_len - 1))]
for start, end in ranges:
    end = min(end, windows_len - 1)
    x = np.arange(start, end + 1)
    plt.figure(figsize=(10, 4))
    for key in ['lstm','stgcn','resstgcn']:
        y_o = window_rmse_o[key][start:end + 1]
        plt.plot(x, y_o, label=key.upper(), linewidth=2)
    plt.xlabel('Window Index')
    plt.ylabel('RMSE per Window\n(원래 단위)')
    plt.title(f'모델별 윈도우 RMSE 변화 추이 (Windows {start} to {end}) (Denormalized)')
    plt.legend(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    filename = f'window_rmse_trend_denorm_{start}_{end}.png'
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close()
    print(f"▶ Saved: {SAVE_DIR}/{filename}\n")

# 10.6 채널×노드 평균 RMSE Heatmap (ResSTGCN 기준, Denormalized)
mse_res_o = np.mean((results['resstgcn']['preds_orig'] - results['resstgcn']['trues_orig'])**2, axis=0)  # (1370,8)
rmse_res_o = np.sqrt(mse_res_o)  # (1370,8)
plt.figure(figsize=(12, 3))
plt.imshow(rmse_res_o.T, aspect='auto', cmap='viridis', interpolation='nearest')
plt.colorbar(label='RMSE (원래 단위)')
plt.xlabel('Node Index (0~1369)')
plt.ylabel('Channel (0~7)')
plt.title('ResSTGCN 채널×노드 평균 RMSE Heatmap (Denormalized)')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,'heatmap_resstgcn_channel_node_denorm.png'))
plt.close()
print(f"▶ Saved: {SAVE_DIR}/heatmap_resstgcn_channel_node_denorm.png\n")

# ---------------- Epoch evolution plots ----------------------------------
if epoch_nums and args.ckpt_tpl_lstm and args.ckpt_tpl_stgcn and args.ckpt_tpl_resstgcn:
    def load_ckpt(path_tmpl, epoch):
        path = path_tmpl.format(epoch=epoch)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return torch.load(path, map_location=device)

    epoch_metrics = {'lstm':[], 'stgcn':[], 'resstgcn':[]}
    for ep in epoch_nums:
        # LSTM
        lstm_model.load_state_dict(load_ckpt(args.ckpt_tpl_lstm, ep)['model_state_dict'])
        r = evaluate_model(lstm_model,'lstm',args.target_date)
        epoch_metrics['lstm'].append(np.sqrt(np.mean((r['preds_norm']-r['trues_norm'])**2)))
        # ST‑GCN
        stgcn_model.load_state_dict(load_ckpt(args.ckpt_tpl_stgcn, ep)['model_state_dict'])
        r = evaluate_model(stgcn_model,'stgcn',args.target_date)
        epoch_metrics['stgcn'].append(np.sqrt(np.mean((r['preds_norm']-r['trues_norm'])**2)))
        # ResSTGCN
        state = load_ckpt(args.ckpt_tpl_resstgcn, ep)
        res_model.stgcn.load_state_dict(state['stgcn_state_dict'])
        res_model.reslstm.load_state_dict(state['reslstm_state_dict'])
        r = evaluate_model(res_model,'resstgcn',args.target_date)
        epoch_metrics['resstgcn'].append(np.sqrt(np.mean((r['preds_norm']-r['trues_norm'])**2)))

    # plot
    plt.figure(figsize=(8,5))
    for key in ['lstm','stgcn','resstgcn']:
        plt.plot(epoch_nums, epoch_metrics[key], marker='o', label=key.upper())
    plt.xlabel('Epoch')
    plt.ylabel('Global RMSE (norm space)')
    plt.title('Metric vs Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR,'epoch_vs_rmse.png'))
    plt.close()
    print(f"▶ Saved: {SAVE_DIR}/epoch_vs_rmse.png")

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 11. Inference 시간 기록                                                    │
# └──────────────────────────────────────────────────────────────────────────┘
with open(os.path.join(SAVE_DIR,'inference_time_denorm.txt'), 'w') as f:
    for key in ['lstm','stgcn','resstgcn']:
        f.write(f"{key.upper()} total inference time (s): {results[key]['inf_time']:.3f}\n")
print(f"▶ Saved: {SAVE_DIR}/inference_time_denorm.txt\n")

print(f"▶ All done. Denormalized compare results are saved in '{SAVE_DIR}'")