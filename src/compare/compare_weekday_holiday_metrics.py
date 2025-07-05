#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_weekday_holiday_metrics.py
----------------------------------
1) Load best checkpoints for LSTM, ST-GCN, PooledResSTGCN
2) Run each on the test set, tag each sample as Weekday or Holiday
3) Compute MAE, RMSE, MAPE, R² separately for the two groups
4) Save a CSV summary + grouped-bar charts

Usage
-----
python compare_weekday_holiday_metrics.py \
    --lstm_ckpt  ck_lstm/lstm_ep040.pt \
    --stgcn_ckpt ck_stgcn/stgcn_ep040.pt \
    --pooled_ckpt ck_pooled/pooled_ep040.pt \
    --batch 8 \
    --out_csv  weekday_holiday_metrics.csv \
    --out_png  weekday_holiday_plot.png \
    --dpi 150
"""
from __future__ import annotations
import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
import holidays

# load per-date normalization stats from raw tensors
DATE_LIST = [
    20220810,20220811,20220812,20220813,20220814,20220815,
    20220819,20220820,20220821,20220822,20220823,
    20220906,20220907,20220909,20220910,20220911,
    20220912,20220913,20220914,20220915,20220916,
    20220926,20220928,20220930,20221001,20221002,
    20221026,20221027,20221028,20221030,20221031
]
date_stats = {}
for dt in DATE_LIST:
    raw = np.load(f'3_tensor/{dt}/input_tensor_{dt}.npy')  # (288,1370,8)
    means = raw.mean(axis=(0,1))   # shape (8,)
    stds  = raw.std(axis=(0,1))    # shape (8,)
    date_stats[dt] = (means, stds)

# 프로젝트 코드
from model.lstm_model import BasicLSTM
from model.stgcn_model import STGCN
from model.pooled_residual_stgcn import PooledResSTGCN

# ────────────────────────────── CLI
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lstm_ckpt',   required=True)
    p.add_argument('--stgcn_ckpt',  required=True)
    p.add_argument('--pooled_ckpt', required=True)
    p.add_argument('--batch',       type=int, default=8)
    p.add_argument('--out_csv',     default='weekday_holiday_metrics.csv')
    p.add_argument('--out_png',     default='weekday_holiday_plot.png')
    p.add_argument('--dpi',         type=int, default=150)
    p.add_argument('--windows_dir', default='3_tensor/windows',
                   help='all_X.npy 등이 저장된 디렉터리')
    p.add_argument('--cluster_map', default='',
                   help="Optional path to 'lane_to_segment_id_by_edge.npy'. "
                        "If omitted, script searches default locations.")
    return p.parse_args()

# ────────────────────────────── 공통 유틸
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str,float]:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    r2   = r2_score(y_true, y_pred)
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}

# ────────────────────────────── 휴일/주말 판별 유틸
kr_holidays = holidays.KR()              # 한국 공휴일
def is_weekend_or_holiday(date_int: int) -> bool:
    """YYYYMMDD 정수 → 평일(False) / 주말·공휴일(True)"""
    ts = pd.to_datetime(str(int(date_int)), format='%Y%m%d')
    return (ts.weekday() >= 5) or (ts.date() in kr_holidays)

class WindowDataset(Dataset):
    """
    all_X.npy   : (N, 12, 1370, 9)
    all_Y.npy   : (N, 1370, 8)
    all_DATE.npy: (N,)  – YYYYMMDD
    """
    def __init__(self, windows_dir: str):
        self.X     = np.load(os.path.join(windows_dir, 'all_X.npy'), mmap_mode='r')
        self.Y     = np.load(os.path.join(windows_dir, 'all_Y.npy'), mmap_mode='r')
        self.DATE  = np.load(os.path.join(windows_dir, 'all_DATE.npy'), mmap_mode='r')
        assert len(self.X) == len(self.Y) == len(self.DATE), "X, Y, DATE 길이가 다릅니다."

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        # .copy() → writable NumPy array; avoids torch warning
        x = self.X[idx].copy()        # (12,1370,9)
        y = self.Y[idx].copy()        # (1370,8)
        d = int(self.DATE[idx])       # YYYYMMDD
        return torch.from_numpy(x).float(), torch.from_numpy(y).float(), d

def split_and_accumulate(
    preds: list[np.ndarray], trues: list[np.ndarray],
    holiday_flags: list[np.ndarray], dates_list: list[np.ndarray]
) -> tuple[dict, dict]:
    """
    preds, trues: 리스트마다 (B, target_dim) ndarray
    holiday_flags: 같은 길이의 (B,) bool ndarray
    dates_list: 같은 길이의 (B,) int ndarray
    Returns metrics dict for weekday / holiday.
    """
    dates_all = np.concatenate(dates_list)      # (samples,)
    # concatenate preds/trues
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)
    holidays = np.concatenate(holiday_flags)

    # Determine dimensions
    # Use the first date's stats to get sigma shape
    first_dt = int(dates_all[0])
    mu, sigma = date_stats[first_dt]
    num_channels = sigma.shape[0]  # number of features (8)
    num_nodes = y_true.shape[1] // num_channels
    # Reshape to (samples, nodes, channels)
    y_true2 = y_true.reshape(-1, num_nodes, num_channels)
    y_pred2 = y_pred.reshape(-1, num_nodes, num_channels)
    # Inverse-normalize per sample-node-channel
    y_true_raw2 = np.zeros_like(y_true2)
    y_pred_raw2 = np.zeros_like(y_pred2)
    for i, dt in enumerate(dates_all):
        mu, sigma = date_stats[int(dt)]
        y_true_raw2[i] = y_true2[i] * sigma + mu
        y_pred_raw2[i] = y_pred2[i] * sigma + mu
    # Flatten back to (samples*num_nodes*channels,)
    y_true_raw = y_true_raw2.reshape(-1)
    y_pred_raw = y_pred_raw2.reshape(-1)
    # Create masks for raw flattened data
    mask_weekday = np.repeat(~holidays, num_nodes * num_channels)
    mask_holiday = np.repeat(holidays, num_nodes * num_channels)
    # Compute metrics on raw
    weekday_metrics = compute_metrics(
        y_true_raw[mask_weekday],
        y_pred_raw[mask_weekday])
    holiday_metrics = compute_metrics(
        y_true_raw[mask_holiday],
        y_pred_raw[mask_holiday])
    return weekday_metrics, holiday_metrics

# ────────────────────────────── 모델별 평가 → (weekday_metrics, holiday_metrics)
def evaluate_lstm(ckpt_path, loader, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = BasicLSTM(
        num_nodes=loader.dataset.X.shape[2],
        input_dim=8,
        hidden_dim=ckpt['model_state']['lstm.weight_ih_l0'].shape[0] // 4,
        num_layers=1, dropout=0.0
    ).to(device)
    model.load_state_dict(ckpt['model_state'], strict=True); model.eval()

    preds, trues, flags = [], [], []
    dates_list = []
    with torch.no_grad():
        for x, y_last, dates in loader:          # dates: (B,)
            holi = np.array([is_weekend_or_holiday(d) for d in dates.numpy()])
            # LSTM expects (B, C, T, N): permute and include flag channel
            x_lstm = x.permute(0, 3, 1, 2).to(device)   # (B,9,12,1370)
            pred = model(x_lstm).cpu().numpy()          # (B,8,N)
            preds.append(pred.reshape(y_last.size(0), -1))
            trues.append(y_last.numpy().reshape(y_last.size(0), -1))
            flags.append(holi)
            dates_list.append(dates.numpy())
    return split_and_accumulate(preds, trues, flags, dates_list)

def evaluate_stgcn(ckpt_path, loader, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    A = torch.from_numpy(np.load('3_tensor/adjacency/A_lane.npy')).float().to(device)
    num_nodes = A.size(0)
    model = STGCN(in_channels=9,
                  hidden1=ckpt['model_state']['layer1.temp1.weight'].shape[0],
                  out_channels=8,
                  num_nodes=num_nodes, A=A, horizon=1).to(device)
    model.load_state_dict(ckpt['model_state'], strict=True); model.eval()

    preds, trues, flags = [], [], []
    dates_list = []
    with torch.no_grad():
        for x, y_last, dates in loader:
            holi = np.array([is_weekend_or_holiday(d) for d in dates.numpy()])
            # ST-GCN expects (B, C, T, N)
            x_stgcn = x.permute(0, 3, 1, 2).to(device)  # (B,9,12,1370)
            pred = model(x_stgcn).squeeze(2).cpu().numpy()  # (B,8,N)
            preds.append(pred.reshape(y_last.size(0), -1))
            trues.append(y_last.numpy().reshape(y_last.size(0), -1))
            flags.append(holi)
            dates_list.append(dates.numpy())
    return split_and_accumulate(preds, trues, flags, dates_list)

def evaluate_pooled(ckpt_path, loader, device, cluster_map_path: str = ''):
    ckpt = torch.load(ckpt_path, map_location=device)

    # ① Adjacency
    A = torch.from_numpy(np.load('3_tensor/adjacency/A_lane.npy')).float().to(device)

    # ② Cluster map (lane → segment)
    cluster_map_candidates = []
    if cluster_map_path:
        cluster_map_candidates.append(cluster_map_path)
    # default fallbacks
    cluster_map_candidates += [
        'segment/lane_to_segment_id_by_edge.npy',
        '3_tsegment/lane_to_segment_id_by_edge.npy'
    ]
    cluster_np = None
    for p in cluster_map_candidates:
        if os.path.exists(p):
            cluster_np = np.load(p)
            break
    if cluster_np is None:
        raise FileNotFoundError(
            "lane_to_segment_id_by_edge.npy not found in default locations.\n"
            "Please provide the correct path or move the file to "
            "'segment/' or '3_tensor/segment/'."
        )
    cluster_id = torch.from_numpy(cluster_np).long().to(device)
    num_nodes = A.size(0)
    model = PooledResSTGCN(
        in_c = ckpt['model_state']['stgcn.layer1.temp1.weight'].shape[1],
        out_c=8, num_nodes=num_nodes, A=A,
        cluster_id=cluster_id, K=cluster_id.max().item()+1,
        hidden_lstm=ckpt['model_state']['clstm.pre_fc.weight'].shape[0],
        horizon=1).to(device)
    model.load_state_dict(ckpt['model_state'], strict=True); model.eval()

    preds, trues, flags = [], [], []
    dates_list = []
    with torch.no_grad():
        for x, y_next, dates in loader:
            holi = np.array([is_weekend_or_holiday(d) for d in dates.numpy()])
            # PooledRes expects (B, C, T, N)
            x_pool = x.permute(0, 3, 1, 2).to(device)         # (B,9,12,1370)
            # residual_seq is first horizon step y_next: shape (B, T, C, N)
            res = y_next.unsqueeze(1).permute(0, 1, 3, 2).to(device)  # (B,1,8,1370)
            pred = model(x_pool, residual_seq=res).squeeze(2).cpu().numpy()
            preds.append(pred.reshape(y_next.size(0), -1))
            trues.append(y_next.numpy().reshape(y_next.size(0), -1))
            flags.append(holi)
            dates_list.append(dates.numpy())
    return split_and_accumulate(preds, trues, flags, dates_list)

# ────────────────────────────── 메인
def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = WindowDataset(args.windows_dir)
    # CUDA 환경일 때만 pin_memory 사용 (MPS/CPU에서는 False)
    pin_mem = (device.type == 'cuda')
    test_loader = DataLoader(dataset, batch_size=args.batch,
                             shuffle=False, num_workers=4,
                             pin_memory=pin_mem)
    print(f'▶ Loaded windows dataset: {len(dataset):,} samples')

    rows = []   # CSV 누적
    evals = [
        ('LSTM',      evaluate_lstm,   args.lstm_ckpt),
        ('ST-GCN',    evaluate_stgcn,  args.stgcn_ckpt),
        ('PooledRes', lambda ck, ld, dev: evaluate_pooled(ck, ld, dev, args.cluster_map), args.pooled_ckpt),
    ]

    for name, fn, ckpt in evals:
        print(f'▶ Evaluating {name} ...')
        weekday_m, holiday_m = fn(ckpt, test_loader, device)

        row_w = {'model': name, 'type': 'Weekday', **weekday_m}
        row_h = {'model': name, 'type': 'Holiday', **holiday_m}
        rows.extend([row_w, row_h])

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f'💾 Metrics saved to {args.out_csv}')

    # ────────── 시각화: 모델별로 Weekday vs Holiday 2-bar
    metrics = ['mae', 'rmse', 'mape', 'r2']
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7,4))
        width = 0.35
        models = df['model'].unique()
        idx = np.arange(len(models))

        weekday_vals = [df[(df.model==m)&(df.type=='Weekday')][metric].values[0] for m in models]
        holiday_vals = [df[(df.model==m)&(df.type=='Holiday')][metric].values[0] for m in models]

        ax.bar(idx - width/2, weekday_vals, width, label='Weekday')
        ax.bar(idx + width/2, holiday_vals, width, label='Holiday')

        ax.set_xticks(idx); ax.set_xticklabels(models, fontsize=11)
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} – Weekday vs Holiday')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()

        out_png_metric = args.out_png.replace('.png', f'_{metric}.png')
        plt.savefig(out_png_metric, dpi=args.dpi)
        print(f'📈 Saved {metric.upper()} plot → {out_png_metric}')
        plt.show()

if __name__ == '__main__':
    main()