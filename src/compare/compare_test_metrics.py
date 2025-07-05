#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_and_compare_test_metrics.py

1) Load best checkpoints for each model (LSTM, ST-GCN, PooledResSTGCN)
2) Run each on the test set, compute MAE, RMSE, MAPE, R²
3) Save results to CSV and plot a grouped bar chart

Usage:
    python compare_test_metrics.py \
        --lstm_ckpt ck_lstm/lstm_ep040.pt \
        --stgcn_ckpt ck_stgcn/stgcn_ep040.pt \
        --pooled_ckpt ck_pooled/pooled_ep040.pt \
        --batch 8 \
        --out_csv test_metrics.csv \
        --out_png performance_comparison.png \
        --dpi 150
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn

# import your models and data loader
from data_loader import get_dataloaders
from model.lstm_model import BasicLSTM
from model.stgcn_model import STGCN
from model.pooled_residual_stgcn import PooledResSTGCN

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lstm_ckpt',    type=str, required=True,  help='Path to LSTM checkpoint')
    p.add_argument('--stgcn_ckpt',   type=str, required=True,  help='Path to ST-GCN checkpoint')
    p.add_argument('--pooled_ckpt',  type=str, required=True,  help='Path to PooledResSTGCN checkpoint')
    p.add_argument('--batch',        type=int, default=8,      help='Test batch size')
    p.add_argument('--out_csv',      type=str, default='test_metrics.csv', help='Output CSV file')
    p.add_argument('--out_png',      type=str, default='performance_comparison.png', help='Output PNG file')
    p.add_argument('--dpi',          type=int, default=150,    help='Figure DPI')
    return p.parse_args()

def evaluate_lstm(ckpt_path, test_loader, device):
    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model = BasicLSTM(
        num_nodes=test_loader.dataset.X.shape[2],
        input_dim=8,
        hidden_dim=ckpt['model_state']['lstm.weight_ih_l0'].shape[0] // 4,
        num_layers=1,
        dropout=0.0
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for x, y_last, *_ in test_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()       # (B,8,N)
            true = y_last.numpy()               # (B,8,N)
            y_pred_all.append(pred.reshape(-1))
            y_true_all.append(true.reshape(-1))
    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    return y_true, y_pred

def evaluate_stgcn(ckpt_path, test_loader, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # load adjacency from data_loader or fixed path
    A = torch.from_numpy(np.load('3_tensor/adjacency/A_lane.npy')).float().to(device)
    num_nodes = A.size(0)
    model = STGCN(
        in_channels=9,
        hidden1=ckpt['model_state']['layer1.temp1.weight'].shape[0],
        out_channels=8,
        num_nodes=num_nodes,
        A=A,
        horizon=1
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for x, y_last, *_ in test_loader:
            x = x.to(device)
            pred = model(x).squeeze(2).cpu().numpy()  # (B,8,N)
            true = y_last.numpy()
            y_pred_all.append(pred.reshape(-1))
            y_true_all.append(true.reshape(-1))
    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    return y_true, y_pred

def evaluate_pooled(ckpt_path, test_loader, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # load adjacency and cluster map
    A = torch.from_numpy(np.load('3_tensor/adjacency/A_lane.npy')).float().to(device)
    cluster_id = torch.from_numpy(np.load('segment/lane_to_segment_id_by_edge.npy')).long().to(device)
    num_nodes = A.size(0)
    model = PooledResSTGCN(
        in_c=ckpt['model_state']['stgcn.layer1.temp1.weight'].shape[1],
        out_c=8,
        num_nodes=num_nodes,
        A=A,
        cluster_id=cluster_id,
        K=cluster_id.max().item()+1,
        hidden_lstm=ckpt['model_state']['clstm.pre_fc.weight'].shape[0],
        horizon=1
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for x, _, y_seq, *_ in test_loader:
            x = x.to(device)
            # residual_seq = first horizon steps of y_seq
            res = y_seq[:, :1].to(device)
            pred = model(x, residual_seq=res).squeeze(2).cpu().numpy()  # (B,8,N)
            true = y_seq[:, 0].numpy()                                 # (B,8,N)
            y_pred_all.append(pred.reshape(-1))
            y_true_all.append(true.reshape(-1))
    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    return y_true, y_pred

def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # avoid division by zero
    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test loader
    _, _, test_loader = get_dataloaders(batch_size=args.batch)

    results = []
    for name, fn, ckpt in [
        ('LSTM',      evaluate_lstm,   args.lstm_ckpt),
        ('ST-GCN',    evaluate_stgcn,  args.stgcn_ckpt),
        ('ST-GCN + LSTM', evaluate_pooled, args.pooled_ckpt)
    ]:
        print(f"Evaluating {name} …")
        y_true, y_pred = fn(ckpt, test_loader, device)
        mae, rmse, mape, r2 = compute_metrics(y_true, y_pred)
        results.append({'model': name, 'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2})

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved test metrics to {args.out_csv}")

    # Plot each metric separately
    metrics = ['mae', 'rmse', 'mape', 'r2']
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(df['model'], df[metric], width=0.6)
        ax.set_ylabel(metric.upper())
        ax.set_title(f'Test-Set {metric.upper()} Comparison')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        out_png_metric = args.out_png.replace('.png', f'_{metric}.png')
        plt.savefig(out_png_metric, dpi=args.dpi)
        print(f"Saved {metric.upper()} bar chart to: {out_png_metric}")
        plt.show()

if __name__ == '__main__':
    main()