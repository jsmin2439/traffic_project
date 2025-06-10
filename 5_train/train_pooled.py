#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pooled.py  (revised)
-------------------------
• **PooledResSTGCN** 단독 학습 (multi-step, default horizon=1)
• Epoch마다 **MSE + MAE** 기록 → `ck_pooled/metrics.csv`
• **5 epoch마다 또는 Val MSE 향상 시** 체크포인트를 `ck_pooled/`에 저장
• 학습曲선 PNG + 로그 NPZ 모두 같은 폴더에 저장

사용 예)
    python 5_train/train_pooled.py \
        --epochs 40 --batch 8 --lr 5e-4 \
        --hidden_lstm 256 --residual_steps 12 \
        --cluster_map lane_to_segment_id_by_edge.npy
"""

import os
import argparse
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────
# Path & imports
# ─────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from data_loader import get_dataloaders
from model.pooled_residual_stgcn import PooledResSTGCN

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Train PooledResSTGCN: residual history -> one-step prediction")
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch', type=int, default=8)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--clip', type=float, default=5.0)
parser.add_argument('--hidden_lstm', type=int, default=256)
parser.add_argument('--residual_steps', type=int, default=12)
parser.add_argument('--cluster_map', type=str, default='lane_to_segment_id_by_edge.npy')
parser.add_argument('--horizon', type=int, default=1)
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────
# Globals placeholders
# ─────────────────────────────────────────────────────────────
C_in = C_out = num_nodes = residual_steps = None
train_loader = val_loader = None
A = cluster_id = num_clusters = None

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def save_ckpt(path: str, epoch: int, model: nn.Module, val_mse: float, val_mae: float):
    torch.save({'epoch': epoch,
                'model_state': model.state_dict(),
                'val_mse': val_mse,
                'val_mae': val_mae}, path)

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    global train_loader, val_loader, C_in, C_out, num_nodes, residual_steps
    global A, cluster_id, num_clusters, args

    # Data ----------------------------------------------------------------
    train_loader, val_loader, _ = get_dataloaders(batch_size=args.batch)
    sample_x, sample_y, *_ = next(iter(train_loader))
    C_in = sample_x.shape[1]
    C_out = sample_y.shape[1]
    T = sample_x.shape[2]
    num_nodes = sample_x.shape[3]
    residual_steps = min(args.residual_steps, T)

    # Clusters & adjacency -------------------------------------------------
    cluster_np = np.load(args.cluster_map)
    cluster_id = torch.from_numpy(cluster_np).long().to(device)
    num_clusters = int(cluster_id.max()) + 1
    A_path = project_root / '3_tensor' / 'adjacency' / 'A_lane.npy'
    A = torch.from_numpy(np.load(A_path)).float().to(device)

    print(f"✔ Clusters {num_clusters} | Nodes {num_nodes} | Cin {C_in} | Cout {C_out} | Residual {residual_steps}")

    # Output dirs ---------------------------------------------------------
    out_dir = Path('ck_pooled')
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / 'metrics.csv'
    with open(csv_path, 'w') as f:
        f.write('epoch,train_mse,train_mae,val_mse,val_mae\n')

    # Train ---------------------------------------------------------------
    train_mse_hist, val_mse_hist = train_full(out_dir, csv_path)

    # Plot ----------------------------------------------------------------
    np.savez(out_dir / 'losses.npz', train=np.array(train_mse_hist), val=np.array(val_mse_hist))
    plt.figure(figsize=(6,4))
    epochs_arr = np.arange(1, len(train_mse_hist)+1)
    plt.plot(epochs_arr, train_mse_hist, label='train MSE')
    plt.plot(epochs_arr, val_mse_hist, label='val MSE')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / 'loss_curve.png', dpi=150)
    print("▶ Training complete. Outputs under ck_pooled/")

# ------------------------------------------------------------------------
# Training routine
# ------------------------------------------------------------------------

def train_full(out_dir: Path, csv_path: Path):
    global train_loader, val_loader, num_nodes, C_in, C_out, A, cluster_id, num_clusters, args

    model = PooledResSTGCN(in_c=C_in - 1,
                           out_c=C_out,
                           num_nodes=num_nodes,
                           A=A,
                           cluster_id=cluster_id,
                           horizon=args.horizon,
                           K=num_clusters,
                           hidden_lstm=args.hidden_lstm).to(device)

    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val = float('inf')
    train_mse_hist, val_mse_hist = [], []

    for ep in range(1, args.epochs + 1):
        # Train -----------------------------------------------------------
        model.train(); tr_mse_sum=tr_mae_sum=0.0
        pbar = tqdm(train_loader, desc=f"[Train] Ep{ep}", ncols=80)
        for x, _, y_seq, *_ in pbar:
            x = x.to(device)
            y_true = y_seq[:, :args.horizon].to(device)
            optimizer.zero_grad()
            y_hat = model(x, residual_seq=y_true).permute(0,2,1,3)
            mse = mse_criterion(y_hat, y_true)
            mae = mae_criterion(y_hat, y_true)
            mse.backward()
            if args.clip>0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            tr_mse_sum += mse.item(); tr_mae_sum += mae.item()
            pbar.set_postfix(mse=f"{mse.item():.4f}")
        pbar.close()
        train_mse = tr_mse_sum/len(train_loader); train_mae = tr_mae_sum/len(train_loader)

        # Val -------------------------------------------------------------
        model.eval(); val_mse_sum=val_mae_sum=0.0
        with torch.no_grad():
            for x, _, y_seq, *_ in tqdm(val_loader, desc=f"[Val]   Ep{ep}", ncols=80):
                x = x.to(device)
                y_true = y_seq[:, :args.horizon].to(device)
                y_hat = model(x, residual_seq=y_true).permute(0,2,1,3)
                val_mse_sum += mse_criterion(y_hat, y_true).item()
                val_mae_sum += mae_criterion(y_hat, y_true).item()
        val_mse = val_mse_sum/len(val_loader); val_mae = val_mae_sum/len(val_loader)

        scheduler.step(val_mse)

        # Log -------------------------------------------------------------
        print(f"[Ep{ep:02d}] Train MSE {train_mse:.4f} | Val MSE {val_mse:.4f} || Train MAE {train_mae:.4f} | Val MAE {val_mae:.4f}")
        with open(csv_path, 'a') as f:
            f.write(f"{ep},{train_mse:.6f},{train_mae:.6f},{val_mse:.6f},{val_mae:.6f}\n")

        train_mse_hist.append(train_mse); val_mse_hist.append(val_mse)

        # Checkpoint ------------------------------------------------------
        if ep % 5 == 0 or val_mse < best_val:
            ckpt_path = out_dir / f"pooled_ep{ep:03d}.pt"
            save_ckpt(ckpt_path, ep, model, val_mse, val_mae)
            print(f"▶ Saved checkpoint: {ckpt_path}")
        if val_mse < best_val:
            best_val = val_mse
            no_improve=0
        else:
            no_improve = getattr(locals(), 'no_improve', 0)+1
            if no_improve >= 10:
                print(f"▶ Early stop at epoch {ep}"); break

    return train_mse_hist, val_mse_hist

# ------------------------------------------------------------------------
if __name__ == '__main__':
    main()
