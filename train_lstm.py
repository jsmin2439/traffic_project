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
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────
# 0. 학습 파라미터 설정: 명령줄 인자를 파싱하여 hyperparameter 정의
# ─────────────────────────────────────────────────────────────────────────
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

# 체크포인트 저장 디렉토리 생성
ensure_dir(args.checkpoint_dir)

# ─────────────────────────────────────────────────────────────────────────
# 1. 학습에 사용할 장치(GPU/MPS/CPU) 설정
# ─────────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"\u25b6 Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────
# 2. 학습/검증/테스트 데이터 로드
# ─────────────────────────────────────────────────────────────────────────
train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch)

# ─────────────────────────────────────────────────────────────────────────
# 3. Basic LSTM 모델 인스턴스 생성
# ─────────────────────────────────────────────────────────────────────────
num_nodes = 1370            # 차로 수
input_dim = 8               # 입력 채널 수 (queue 4 + speed 4)
model = BasicLSTM(num_nodes=num_nodes, input_dim=input_dim,
                  hidden_dim=args.hidden, num_layers=1, dropout=0.0).to(device)
print(f"\u25b6 LSTM Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# 손실함수 및 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
early_stopper = EarlyStopping(patience=args.patience, min_delta=1e-5)

# ─────────────────────────────────────────────────────────────────────────
# 4. 학습 루프 시작
# ─────────────────────────────────────────────────────────────────────────
for epoch in range(1, args.epochs + 1):
    model.train()  # 학습 모드
    train_loss_meter = 0.0
    n_batches = 0

    # ───── A) Training 루프 ─────
    for x_batch, y_batch, _ in tqdm(train_loader, ncols=80, desc=f"[Epoch {epoch}/{args.epochs}] Train"):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # LSTM 입력 형태로 변환: (B, C, T, N) → (B, T, N, C) → (B, T, N, 8)
        x_lstm = x_batch[:, :8, :, :].permute(0, 2, 3, 1).contiguous()

        optimizer.zero_grad()
        y_pred = model(x_lstm)  # 예측 결과: (B, 1370, 8)

        # 실제 정답: (B, 8, 1370) → (B, 1370, 8)로 차원 맞춤
        loss = criterion(y_pred, y_batch.permute(0, 2, 1))
        loss.backward()

        # Gradient clipping (폭주 방지)
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        train_loss_meter += loss.item()
        n_batches += 1

    train_loss = train_loss_meter / n_batches

    # ───── B) Validation 루프 ─────
    model.eval()
    val_loss_meter = 0.0
    n_val_batches = 0
    with torch.no_grad():
        for x_batch, y_batch, _ in tqdm(val_loader, ncols=80, desc=f"[Epoch {epoch}/{args.epochs}] Val"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            x_lstm = x_batch[:, :8, :, :].permute(0, 2, 3, 1).contiguous()
            y_pred = model(x_lstm)
            loss_val = criterion(y_pred, y_batch.permute(0, 2, 1))
            val_loss_meter += loss_val.item()
            n_val_batches += 1

    val_loss = val_loss_meter / n_val_batches

    # ───── C) 로그 출력 ─────
    print(f"[Epoch {epoch:02d}/{args.epochs:02d}] "
          f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    print_memory_usage(device)

    # ───── D) EarlyStopping 체크 및 저장 ─────
    early_stopper.step(val_loss)
    if early_stopper.stop:
        print(f"\u25b6 Early stopping triggered at epoch {epoch}")
        break

    # 매 5 epoch마다 체크포인트 저장
    if epoch % 5 == 0:
        ckpt_path = os.path.join(args.checkpoint_dir, f'lstm_epoch{epoch:03d}.pt')
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss}, ckpt_path)
        print(f"\u25b6 Checkpoint saved: {ckpt_path}")

print("\u25b6 Training finished.")