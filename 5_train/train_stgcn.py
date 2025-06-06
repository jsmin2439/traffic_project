#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_stgcn.py

원본 ST‐GCN 모델 학습 스크립트

1) data_loader.py에서 train/val 데이터 로드
2) STGCN 모델 생성 (in_channels=9 → out_channels=8)
3) 학습/검증 루프
4) 매 epoch마다 loss 출력 + 체크포인트 저장
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders
from model.stgcn_model import STGCN
from train_utils import EarlyStopping, ensure_dir, print_memory_usage
from tqdm import tqdm

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 0. ArgumentParser 설정                                                    │
# └──────────────────────────────────────────────────────────────────────────┘
parser = argparse.ArgumentParser(description="Train Original ST-GCN on Traffic Data")
parser.add_argument('--batch', type=int, default=2, help='Batch size')
parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for Adam')
parser.add_argument('--patience', type=int, default=7, help='EarlyStopping patience')
parser.add_argument('--clip', type=float, default=5.0, help='Gradient clipping max norm (0=off)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_stgcn',
                    help='Directory to save model checkpoints')
parser.add_argument('--sched', type=str, default='plateau', choices=['plateau', 'cosine'],
                    help='Learning rate scheduler type')
args = parser.parse_args()

# 체크포인트 폴더 생성
ensure_dir(args.checkpoint_dir)

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 1. 장치(Device) 설정                                                       │
# └──────────────────────────────────────────────────────────────────────────┘
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"▶ Using device: {device}")

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 2. DataLoader 생성                                                         │
# └──────────────────────────────────────────────────────────────────────────┘
train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch)

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 3. 정규화된 인접행렬 A 로드                                                 │
# └──────────────────────────────────────────────────────────────────────────┘
import numpy as _np
from train_utils import ensure_dir

# 3. 정규화된 인접행렬 A 로드
adj_norm_path = os.path.join('3_tensor', 'adjacency', 'A_lane.npy')
if not os.path.exists(adj_norm_path):
    raise FileNotFoundError(f"정규화된 A_lane.npy 파일을 찾을 수 없습니다: {adj_norm_path}")
# 미리 정규화된 adjacency를 로드
A_norm = torch.from_numpy(np.load(adj_norm_path)).float().to(device)  # (1370,1370)

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 4. 모델 생성                                                               │
# └──────────────────────────────────────────────────────────────────────────┘
num_nodes = A_norm.shape[0]      # 1370
in_channels = 9   # Queue4 + Speed4 + Weekend Flag(1)
out_channels = 8  # Queue4 + Speed4

stgcn = STGCN(in_channels=in_channels, out_channels=out_channels,
              num_nodes=num_nodes, A=A_norm).to(device)
print(f"▶ ST-GCN Parameter count: {sum(p.numel() for p in stgcn.parameters() if p.requires_grad)}")

# Loss, Optimizer, Scheduler, EarlyStopping
criterion = nn.MSELoss()
optimizer = optim.Adam(stgcn.parameters(), lr=args.lr, weight_decay=1e-4)

if args.sched == 'plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=3)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs - args.patience,
                                                           eta_min=1e-5)

early_stopper = EarlyStopping(patience=args.patience, min_delta=1e-5)

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 5. 학습 루프                                                               │
# └──────────────────────────────────────────────────────────────────────────┘
for epoch in range(1, args.epochs + 1):
    stgcn.train()
    train_loss_meter = 0.0
    n_batches = 0

    # ─────────────────────────────────────────────────────────────────────────
    # A) Training
    # ─────────────────────────────────────────────────────────────────────────
    for x_batch, y_batch, _ in tqdm(train_loader, ncols=80, desc=f"[Epoch {epoch}/{args.epochs}] Train"):
        # x_batch: (B, 9,12,1370), y_batch: (B, 8,1370)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        # Forward
        y_pred = stgcn(x_batch)    # (B, 8, 1370)
        loss = criterion(y_pred, y_batch)
        loss.backward()

        # Gradient Clipping
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(stgcn.parameters(), args.clip)

        optimizer.step()
        train_loss_meter += loss.item()
        n_batches += 1

    train_loss = train_loss_meter / n_batches

    # ─────────────────────────────────────────────────────────────────────────
    # B) Validation
    # ─────────────────────────────────────────────────────────────────────────
    stgcn.eval()
    val_loss_meter = 0.0
    n_val_batches = 0
    with torch.no_grad():
        for x_batch, y_batch, _ in tqdm(val_loader, ncols=80, desc=f"[Epoch {epoch}/{args.epochs}] Val"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = stgcn(x_batch)
            loss_val = criterion(y_pred, y_batch)
            val_loss_meter += loss_val.item()
            n_val_batches += 1

    val_loss = val_loss_meter / n_val_batches

    # ─────────────────────────────────────────────────────────────────────────
    # C) 학습 로그 출력
    # ─────────────────────────────────────────────────────────────────────────
    print(f"[Epoch {epoch:02d}/{args.epochs:02d}] "
          f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    print_memory_usage(device)

    # ─────────────────────────────────────────────────────────────────────────
    # D) EarlyStopping 및 학습률 스케줄러
    # ─────────────────────────────────────────────────────────────────────────
    early_stopper.step(val_loss)
    if early_stopper.stop:
        print(f"▶ Early stopping triggered at epoch {epoch}")
        break

    if args.sched == 'plateau':
        scheduler.step(val_loss)  # type: ignore
    else:
        scheduler.step()    # type: ignore

    # ─────────────────────────────────────────────────────────────────────────
    # E) 체크포인트 저장
    # ─────────────────────────────────────────────────────────────────────────
    if epoch % 5 == 0:
        ckpt_path = os.path.join(args.checkpoint_dir, f'stgcn_epoch{epoch:03d}.pt')
        torch.save({'epoch': epoch,
                    'model_state_dict': stgcn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss}, ckpt_path)
        print(f"▶ Checkpoint saved: {ckpt_path}")

print("▶ ST-GCN Training finished.")