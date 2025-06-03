#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lstm.py

전통적 LSTM 모델 학습 스크립트

1) data_loader.py에서 제공하는 DataLoader로 train/val 데이터 로드
2) BasicLSTM 모델 생성
3) 학습 루프 및 Validation 루프
4) 매 epoch마다 train/val loss 출력 + 체크포인트 저장
5) TensorBoard 로그 작성 (선택)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders
from model.lstm_model import BasicLSTM
from train_utils import EarlyStopping, ensure_dir, print_memory_usage

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 0. ArgumentParser 설정                                                    │
# └──────────────────────────────────────────────────────────────────────────┘
parser = argparse.ArgumentParser(description="Train Basic LSTM on Traffic Data")
parser.add_argument('--batch', type=int, default=4, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for Adam')
parser.add_argument('--hidden', type=int, default=64, help='LSTM hidden dimension')
parser.add_argument('--patience', type=int, default=5, help='EarlyStopping patience')
parser.add_argument('--clip', type=float, default=5.0, help='Gradient clipping max norm (0=off)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_lstm',
                    help='Directory to save model checkpoints')
args = parser.parse_args()

# 학습 관련 디렉토리 생성
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
# get_dataloaders() → train_loader, val_loader, test_loader 반환
train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch)

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 3. 모델 생성                                                               │
# └──────────────────────────────────────────────────────────────────────────┘
# 각 배치에서 x_batch: (B, C, 12, 1370), y_batch: (B, 8, 1370)
# LSTM 입력 차원 = 8 (큐4 + 스피드4)
num_nodes = 1370
input_dim = 8  # LSTM 입력 feature = Queue4 + Speed4
model = BasicLSTM(num_nodes=num_nodes, input_dim=input_dim,
                  hidden_dim=args.hidden, num_layers=1, dropout=0.0).to(device)
print(f"▶ LSTM Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Loss 및 Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
early_stopper = EarlyStopping(patience=args.patience, min_delta=1e-5)

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 4. 학습 루프                                                               │
# └──────────────────────────────────────────────────────────────────────────┘
for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss_meter = 0.0
    n_batches = 0

    # ─────────────────────────────────────────────────────────────────────────
    # A) Training
    # ─────────────────────────────────────────────────────────────────────────
    for x_batch, y_batch, _ in train_loader:
        # x_batch: (B, C, 12, 1370), y_batch: (B, 8, 1370)
        x_batch = x_batch.to(device)    # (B, C_in, 12, 1370)
        y_batch = y_batch.to(device)    # (B,  8, 1370)

        # ─────────────────────────────────────────────────────────────────────
        # (1) 전통적 LSTM은 “채널=8” 만 필요 → x_batch[:, :8, :, :]
        # (2) LSTM이 기대하는 입력 형태: (B, 12, 1370, 8)
        #     → permute: (B, C, 12, N) → (B, 12, N, C)
        # ─────────────────────────────────────────────────────────────────────
        x_lstm = x_batch[:, :8, :, :].permute(0, 2, 3, 1).contiguous()
        # x_lstm.shape == (B, 12, 1370, 8)

        optimizer.zero_grad()
        y_pred = model(x_lstm)  # (B, 1370, 8)

        # Loss 계산: MSE between 예측값과 실제값 (차원 정렬)
        # y_batch: (B, 8, 1370) → permute → (B, 1370, 8)
        loss = criterion(y_pred, y_batch.permute(0, 2, 1))  
        loss.backward()

        # Gradient Clipping
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        train_loss_meter += loss.item()
        n_batches += 1

    train_loss = train_loss_meter / n_batches

    # ─────────────────────────────────────────────────────────────────────────
    # B) Validation
    # ─────────────────────────────────────────────────────────────────────────
    model.eval()
    val_loss_meter = 0.0
    n_val_batches = 0
    with torch.no_grad():
        for x_batch, y_batch, _ in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            x_lstm = x_batch[:, :8, :, :].permute(0, 2, 3, 1).contiguous()
            y_pred = model(x_lstm)
            loss_val = criterion(y_pred, y_batch.permute(0, 2, 1))
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
    # D) EarlyStopping 체크 + 체크포인트 저장
    # ─────────────────────────────────────────────────────────────────────────
    early_stopper.step(val_loss)
    if early_stopper.stop:
        print(f"▶ Early stopping triggered at epoch {epoch}")
        break

    # 매 5 epoch마다 체크포인트 저장
    if epoch % 5 == 0:
        ckpt_path = os.path.join(args.checkpoint_dir, f'lstm_epoch{epoch:03d}.pt')
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss}, ckpt_path)
        print(f"▶ Checkpoint saved: {ckpt_path}")

print("▶ Training finished.")