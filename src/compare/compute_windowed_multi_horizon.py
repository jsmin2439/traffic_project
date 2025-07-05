#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_windowed_multi_horizon.py

1) all_X.npy, all_Y.npy, all_DATE.npy 로드
   - all_X: (W, T, N, C) 입력 윈도우
   - all_Y: (W, N, 8) 다음 스텝 실제값
   - all_DATE: (W,) 각 윈도우의 날짜 메타데이터
2) 각 윈도우 i에 대해 같은 날짜 구간만을 사용해 1~Horizon 단계 recursive forecasting
3) Horizon별 MAE 또는 RMSE 계산
4) 추세선 그래프 출력

python compute_windowed_multi_horizon.py \
  --lstm_ckpt ck_lstm/lstm_ep040.pt \
  --stgcn_ckpt ck_stgcn/stgcn_ep040.pt \
  --pooled_ckpt ck_pooled/pooled_ep040.pt \
  --out_plot combined_slots_rmse.png \
  --metric rmse \
  --dpi 150

"""

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from tqdm import tqdm

from model.lstm_model import BasicLSTM
from model.stgcn_model import STGCN
from model.pooled_residual_stgcn import PooledResSTGCN

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lstm_ckpt',   required=True, help='LSTM checkpoint path')
    p.add_argument('--stgcn_ckpt',  required=True, help='ST-GCN checkpoint path')
    p.add_argument('--pooled_ckpt', required=True, help='PooledResSTGCN checkpoint path')
    p.add_argument('--all_x',        default='3_tensor/windows/all_X.npy', help='all_X.npy file path')
    p.add_argument('--all_y',        default='3_tensor/windows/all_Y.npy', help='all_Y.npy file path')
    p.add_argument('--all_date',     default='3_tensor/windows/all_DATE.npy', help='all_DATE.npy file path')
    # Removed unused --horizon argument
    p.add_argument('--metric',       choices=['mae','rmse'], default='rmse')
    p.add_argument('--batch',        type=int, default=1024, help='평가 batch size')
    p.add_argument('--out_plot',     required=True)
    p.add_argument('--dpi',          type=int, default=150)
    return p.parse_args()

def load_model(model_type, ckpt_path, num_nodes, device):
    ckpt = torch.load(ckpt_path, map_location=device)['model_state']
    if model_type=='LSTM':
        hidden = ckpt['lstm.weight_ih_l0'].shape[0]//4
        model = BasicLSTM(num_nodes,8,hidden).to(device)
    elif model_type=='ST-GCN':
        A = torch.from_numpy(np.load('3_tensor/adjacency/A_lane.npy')).float().to(device)
        hidden = ckpt['layer1.temp1.weight'].shape[0]
        model = STGCN(9,hidden,8,num_nodes,A,1).to(device)
    else:
        A = torch.from_numpy(np.load('3_tensor/adjacency/A_lane.npy')).float().to(device)
        cid = torch.from_numpy(np.load('segment/lane_to_segment_id_by_edge.npy')).long().to(device)
        in_c = ckpt['stgcn.layer1.temp1.weight'].shape[1]
        hidden_lstm = ckpt['clstm.pre_fc.weight'].shape[0]
        model = PooledResSTGCN(in_c,8,num_nodes,A,cid, cid.max().item()+1, hidden_lstm,1).to(device)
    model.load_state_dict(ckpt)
    model.eval()
    return model

def recursive_on_window(x0, model, args, device):
    """
    x0: (T=12, N, C=9)
    returns next-step prediction (N,8)
    """
    # prepare input tensor
    x = torch.from_numpy(x0).permute(2,0,1).unsqueeze(0).to(device)  # (1,9,12,N)
    # use all past 12 steps for residual correction
    res0 = x[:, :8, :, :].permute(0,2,1,3).contiguous()  # (1,12,8,N)
    # one-step forecast using latest residual
    if model.__class__.__name__ == 'PooledResSTGCN':
        y_hat = model(x, res0).squeeze(2)  # (1,8,N)
    else:
        y_hat = model(x).squeeze(2)        # (1,8,N)
    # return numpy (N,8)
    return y_hat[0].detach().cpu().numpy().T

def compute_slot_errors(all_X, all_Y, all_DATE, model, args):
    INPUT_LEN = 12
    NUM_SLOTS = 288

    W,T,N,C = all_X.shape

    slot_errs = {slot: [] for slot in range(INPUT_LEN, NUM_SLOTS)}

    prev_date = None
    date_counter = 0

    for i in tqdm(range(W), desc="Processing windows", ncols=80):
        date0 = all_DATE[i]
        # reset counter when date changes
        if date0 != prev_date:
            prev_date = date0
            date_counter = 0
        # skip windows that would exceed this date's slots
        if date_counter + INPUT_LEN >= NUM_SLOTS:
            continue

        # predict one step ahead at this window
        y_pred = recursive_on_window(all_X[i], model, args, device=model.device if hasattr(model, 'device') else torch.device('cpu'))  # only 1-step ahead
        y_true = all_Y[i]  # (N,8)
        flat_t = y_true.ravel()
        flat_p = y_pred.ravel()
        if args.metric=='mae':
            err = mean_absolute_error(flat_t, flat_p)
        else:
            err = np.sqrt(mean_squared_error(flat_t, flat_p))
        slot = date_counter + INPUT_LEN
        slot_errs[slot].append(err)
        # advance counter
        date_counter += 1

    return slot_errs

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_X    = np.load(args.all_x)      # (W,T,N,C)
    all_Y    = np.load(args.all_y)      # (W,N,8)
    all_DATE = np.load(args.all_date)   # (W,)

    W,T,N,C = all_X.shape

    models = [
      ('LSTM', args.lstm_ckpt),
      ('ST-GCN', args.stgcn_ckpt),
      ('PooledRes', args.pooled_ckpt)
    ]
    results = {}
    for name, ckpt in models:
        model = load_model(name, ckpt, N, device)
        # attach device attribute for recursive_on_window
        setattr(model, 'device', device)
        slot_errs = compute_slot_errors(all_X, all_Y, all_DATE, model, args)
        slots = sorted(slot_errs.keys())
        vals = [ np.mean(slot_errs[s]) if slot_errs[s] else np.nan for s in slots ]
        results[name] = (slots, vals)

    plt.figure(figsize=(10,5))
    for name, (slots, vals) in results.items():
        plt.plot(slots, vals, marker='o', label=name)
    plt.xlabel('Time Slot (0=00:00)')
    plt.ylabel(args.metric.upper())
    plt.title(f'Average {args.metric.upper()} per Time Slot over All Dates')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=args.dpi)
    plt.show()

if __name__=='__main__':
    main()