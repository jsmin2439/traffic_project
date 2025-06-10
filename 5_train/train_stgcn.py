#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_stgcn.py  (revised)
------------------------
• 전통적 ST-GCN 단독 학습 스크립트 (single-step, horizon=1)
• 매 epoch마다 **MSE + MAE** 두 지표를 CSV 기록
• 5 epoch마다 **또는** Val MSE 갱신 시 체크포인트를 **ck_stgcn/** 에 저장

사용 예)
    python 5_train/train_stgcn.py \
        --batch 8 --epochs 40 --lr 5e-4 \
        --hidden 64 --in_channels 9 --out_channels 8 \
        --save_freq 5
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

# ─────────────────────────────────────────────────────────────
# 프로젝트 루트 추가 및 모듈 로드
# ─────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from data_loader import get_dataloaders
from model.stgcn_model import STGCN


# ─────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Train single-step ST-GCN model")
    p.add_argument('--batch', type=int, default=8, help='Batch size')
    p.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    p.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    p.add_argument('--hidden', type=int, default=64, help='STGCN hidden channels')
    p.add_argument('--in_channels', type=int, default=9, help='Input feature channels')
    p.add_argument('--out_channels', type=int, default=8, help='Output feature channels')
    p.add_argument('--save_freq', type=int, default=5, help='Save checkpoint every N epochs')
    p.add_argument('--patience', type=int, default=10, help='EarlyStopping patience')
    p.add_argument('--clip', type=float, default=5.0, help='Gradient clipping norm')
    p.add_argument('--sched', type=str, default='plateau', choices=['plateau', 'cosine'], help='LR scheduler type')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Helper: checkpoint save
# ─────────────────────────────────────────────────────────────

def save_ckpt(path: str, epoch: int, model: nn.Module, val_mse: float, val_mae: float):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'val_mse': val_mse,
        'val_mae': val_mae,
    }, path)


# ─────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────

def main():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Using device: {device}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(batch_size=args.batch)

    # Adjacency
    A = torch.from_numpy(np.load(project_root / '3_tensor' / 'adjacency' / 'A_lane.npy')).float().to(device)
    num_nodes = A.size(0)

    # Model
    model = STGCN(in_channels=args.in_channels,
                  hidden1=args.hidden,
                  out_channels=args.out_channels,
                  num_nodes=num_nodes,
                  A=A,
                  horizon=1).to(device)
    print(f"▶ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Losses & optimiser
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = (optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
                 if args.sched == 'plateau'
                 else optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5))

    # Output dirs
    ckpt_dir = 'ck_stgcn'
    os.makedirs(ckpt_dir, exist_ok=True)
    csv_path = os.path.join(ckpt_dir, 'metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('epoch,train_mse,train_mae,val_mse,val_mae\n')

    best_val_mse = float('inf')
    patience_counter = 0

    # ──────────────── Training loop ────────────────
    for ep in range(1, args.epochs + 1):
        # -- Train ---------------------------------------------------------
        model.train()
        train_mse_sum, train_mae_sum = 0.0, 0.0
        for x, y_last, *_ in tqdm(train_loader, desc=f"[Train] Ep{ep}", ncols=80):
            x, y_last = x.to(device), y_last.to(device)
            pred = model(x).squeeze(2)
            loss_mse = mse_criterion(pred, y_last)
            loss_mae = mae_criterion(pred, y_last)
            loss = loss_mse  # optimise using MSE

            optimizer.zero_grad()
            loss.backward()
            if args.clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_mse_sum += loss_mse.item()
            train_mae_sum += loss_mae.item()
        train_mse = train_mse_sum / len(train_loader)
        train_mae = train_mae_sum / len(train_loader)

        # -- Validation ----------------------------------------------------
        model.eval()
        val_mse_sum, val_mae_sum = 0.0, 0.0
        with torch.no_grad():
            for x, y_last, *_ in tqdm(val_loader, desc=f"[Val]   Ep{ep}", ncols=80):
                x, y_last = x.to(device), y_last.to(device)
                pred = model(x).squeeze(2)
                val_mse_sum += mse_criterion(pred, y_last).item()
                val_mae_sum += mae_criterion(pred, y_last).item()
        val_mse = val_mse_sum / len(val_loader)
        val_mae = val_mae_sum / len(val_loader)

        # -- Scheduler -----------------------------------------------------
        if args.sched == 'plateau':
            scheduler.step(val_mse)
        else:
            scheduler.step()

        # -- Logging -------------------------------------------------------
        print(f"[Ep{ep:02d}] Train MSE {train_mse:.6f} | Val MSE {val_mse:.6f} || Train MAE {train_mae:.6f} | Val MAE {val_mae:.6f}")
        with open(csv_path, 'a') as f:
            f.write(f"{ep},{train_mse:.6f},{train_mae:.6f},{val_mse:.6f},{val_mae:.6f}\n")

        # -- Checkpoint ----------------------------------------------------
        if ep % args.save_freq == 0 or val_mse < best_val_mse:
            ckpt_path = os.path.join(ckpt_dir, f"stgcn_ep{ep:03d}.pt")
            save_ckpt(ckpt_path, ep, model, val_mse, val_mae)
            print(f"▶ Saved checkpoint: {ckpt_path}")

        # -- Early Stop ----------------------------------------------------
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"▶ Early stopping at epoch {ep}")
                break

    print("▶ Training complete.")


if __name__ == '__main__':
    main()