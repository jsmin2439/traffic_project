#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lstm.py  (revised)
-----------------------
• BasicLSTM 단독 학습 스크립트
• 9-채널 입력(Queue4+Speed4+holiday) → 다음 8-채널 한스텝 예측
• 성능 지표(MSE, MAE) 매 epoch CSV 기록
• 5 epoch마다 or best-val 향상 시 체크포인트를 ck_lstm/ 에 저장

사용 예)
    python 5_train/train_lstm.py \
        --batch 8 --epochs 40 --lr 5e-4 \
        --hidden_dim 64 --num_layers 1 --dropout 0.0 \
        --save_freq 5 --patience 7
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
# 프로젝트 루트 경로 추가
# ─────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from data_loader import get_dataloaders
from model.lstm_model import BasicLSTM


# ─────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Train BasicLSTM for traffic forecasting")
    p.add_argument('--batch', type=int, default=8, help='Batch size')
    p.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    p.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    p.add_argument('--hidden_dim', type=int, default=64, help='LSTM hidden dimension')
    p.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers')
    p.add_argument('--dropout', type=float, default=0.0, help='Dropout between LSTM layers')
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

    # Infer num_nodes from data
    sample_x, sample_y, *_ = next(iter(train_loader))
    num_nodes = sample_x.shape[-1]

    model = BasicLSTM(num_nodes=num_nodes,
                      input_dim=8,
                      hidden_dim=args.hidden_dim,
                      num_layers=args.num_layers,
                      dropout=args.dropout).to(device)
    print(f"▶ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Losses & optimiser
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = (optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
                 if args.sched == 'plateau'
                 else optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5))

    # Output dirs
    ckpt_dir = 'ck_lstm'
    os.makedirs(ckpt_dir, exist_ok=True)
    csv_path = os.path.join(ckpt_dir, 'metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('epoch,train_mse,train_mae,val_mse,val_mae\n')

    best_val = float('inf')
    patience_counter = 0

    # ──────────────── Training loop ────────────────
    for ep in range(1, args.epochs + 1):
        # -- Train ---------------------------------------------------------
        model.train()
        running_mse, running_mae = 0.0, 0.0
        for x, y_last, *_ in tqdm(train_loader, desc=f"[Train] Ep{ep}", ncols=80):
            x, y_last = x.to(device), y_last.to(device)
            pred = model(x)
            loss_mse = mse_criterion(pred, y_last)
            loss_mae = mae_criterion(pred, y_last)
            loss = loss_mse  # optimise with MSE only (common practice)

            optimizer.zero_grad()
            loss.backward()
            if args.clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            running_mse += loss_mse.item()
            running_mae += loss_mae.item()
        train_mse = running_mse / len(train_loader)
        train_mae = running_mae / len(train_loader)

        # -- Validation ----------------------------------------------------
        model.eval()
        val_mse_sum, val_mae_sum = 0.0, 0.0
        with torch.no_grad():
            for x, y_last, *_ in tqdm(val_loader, desc=f"[Val]   Ep{ep}", ncols=80):
                x, y_last = x.to(device), y_last.to(device)
                pred = model(x)
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
        if ep % args.save_freq == 0 or val_mse < best_val:
            ckpt_path = os.path.join(ckpt_dir, f"lstm_ep{ep:03d}.pt")
            save_ckpt(ckpt_path, ep, model, val_mse, val_mae)
            print(f"▶ Saved checkpoint: {ckpt_path}")

        # -- Early Stop ----------------------------------------------------
        if val_mse < best_val:
            best_val = val_mse
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"▶ Early stopping at epoch {ep}")
                break

    print("▶ Training complete.")


if __name__ == '__main__':
    main()
