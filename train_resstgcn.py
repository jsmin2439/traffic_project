#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_resstgcn.py

Residual + ST‐GCN 하이브리드 모델 학습 스크립트

1) data_loader.get_dataloaders()로 train/val 윈도우 로드
2) 정규화된 인접행렬 A 랜덤 로드 → 대칭 정규화
3) 모델 생성: ResSTGCN(in_channels=9, out_channels=8, num_nodes=1370, A=A_norm)
4) 학습 루프
   - A) ST‐GCN 단기 예측: 과거 12스텝 → 다음 스텝 y_pred_st
   - B) Residual 시퀀스 생성 (Rolling‐prediction 방식):
       1) 입력 윈도우 x_batch를 시간축 t=0..10 으로 “롤링”하며
       2) t별 ST‐GCN 예측 y_pred_st_t 얻기
       3) 실제 Y(t+1) 값 y_true_t1과의 차이를 res_seq[:, t, :, :]에 저장
       → 총 res_seq: (B, 12, 1370, 8)
   - C) Residual LSTM 보정: res_seq + weekend_flag → res_corr (B,8,1370)
   - D) 최종 예측: y_pred_st_last + res_corr
   - E) Loss 계산: MSE(y_final, y_true_next)
5) 매 epoch마다 train/val loss 출력 + 체크포인트 저장
"""

import os
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders

# ───────────────────────────────────────────────────────────────────────────
# 전체 정규화된 윈도우 배열을 메모리에 미리 로드 (Rolling‑prediction 용)
# ───────────────────────────────────────────────────────────────────────────
import numpy as _np
all_X_global = _np.load(Path('3_tensor') / 'windows' / 'all_X.npy')  # (num_windows, 12, 1370, C_in)
all_Y_global = _np.load(Path('3_tensor') / 'windows' / 'all_Y.npy')  # (num_windows, 1370, 8)
num_windows = all_X_global.shape[0]
# ───────────────────────────────────────────────────────────────────────────

from model.res_stgcn_model import ResSTGCN
from train_utils import EarlyStopping, ensure_dir, print_memory_usage
from tqdm import tqdm

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 0. ArgumentParser 설정                                                    │
# └──────────────────────────────────────────────────────────────────────────┘
parser = argparse.ArgumentParser(description="Train Residual + ST-GCN Hybrid Model")
parser.add_argument('--batch', type=int, default=2, help='Batch size')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--lr_stgcn', type=float, default=5e-4, help='STGCN 학습률')
parser.add_argument('--lr_reslstm', type=float, default=8e-4, help='Residual LSTM 학습률')
parser.add_argument('--hidden', type=int, default=256, help='Residual LSTM hidden dimension')
parser.add_argument('--patience', type=int, default=7, help='EarlyStopping patience')
parser.add_argument('--clip', type=float, default=5.0, help='Gradient clipping max norm (0=off)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_resstgcn',
                    help='Directory to save checkpoints')
parser.add_argument('--sched', type=str, default='plateau', choices=['plateau', 'cosine'],
                    help='LR scheduler type')
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
# │ 3. 정규화된 인접행렬 A 로드                                                │
# └──────────────────────────────────────────────────────────────────────────┘
import numpy as _np
adj_norm_path = Path('3_tensor') / 'adjacency' / 'A_lane.npy'
if not adj_norm_path.exists():
    raise FileNotFoundError(f"정규화된 A_lane.npy 파일을 찾을 수 없습니다: {adj_norm_path}")
# 미리 정규화된 adjacency를 로드
A_norm = torch.from_numpy(np.load(str(adj_norm_path))).float().to(device)  # (1370, 1370)

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 4. 모델 생성                                                               │
# └──────────────────────────────────────────────────────────────────────────┘
num_nodes = A_norm.shape[0]  # 1370
in_channels = 9    # Queue4 + Speed4 + WeekendFlag
out_channels = 8   # Queue4 + Speed4

model = ResSTGCN(in_channels=in_channels, out_channels=out_channels,
                 num_nodes=num_nodes, A=A_norm, hidden_dim=args.hidden).to(device)
print(f"▶ ResSTGCN Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 5. Loss, Optimizer, Scheduler, EarlyStopping 설정                          │
# └──────────────────────────────────────────────────────────────────────────┘
criterion = nn.MSELoss()
# ST-GCN 파라미터와 ResLSTM 파라미터를 각각 분리하여 다른 lr 사용
params_stgcn = list(model.stgcn.parameters())
params_reslstm = list(model.reslstm.parameters())

optimizer_stgcn = optim.Adam(params_stgcn, lr=args.lr_stgcn, weight_decay=1e-4)
optimizer_reslstm = optim.Adam(params_reslstm, lr=args.lr_reslstm, weight_decay=1e-4)

if args.sched == 'plateau':
    scheduler_stgcn = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_stgcn, mode='min', factor=0.5, patience=3)
    scheduler_reslstm = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_reslstm, mode='min', factor=0.5, patience=3)
else:
    scheduler_stgcn = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_stgcn, T_max=args.epochs - args.patience, eta_min=1e-5)
    scheduler_reslstm = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_reslstm, T_max=args.epochs - args.patience, eta_min=1e-5)

early_stopper = EarlyStopping(patience=args.patience, min_delta=1e-5)

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ 6. 학습 루프                                                               │
# └──────────────────────────────────────────────────────────────────────────┘
for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss_meter = 0.0
    n_batches = 0

    # ─────────────────────────────────────────────────────────────────────────
    # A) Training
    # ─────────────────────────────────────────────────────────────────────────
    for x_batch, y_batch, idx_batch in tqdm(train_loader, ncols=80, desc=f"[Epoch {epoch}/{args.epochs}] Train"):
        # x_batch: (B, 9, 12, 1370), y_batch: (B, 8, 1370)
        x_batch = x_batch.to(device)   # (B, 9, 12, 1370)
        y_batch = y_batch.to(device)   # (B, 8, 1370)
        idx_batch = idx_batch.cpu().numpy()  # (B,)
        B = x_batch.size(0)

        optimizer_stgcn.zero_grad()
        optimizer_reslstm.zero_grad()

        # ─────────────────────────────────────────────────────────────────────
        # 1) ST-GCN 단기 예측 (마지막 스텝만)
        # ─────────────────────────────────────────────────────────────────────
        # y_pred_st_last: (B, 8, 1370)
        y_pred_st_last = model.stgcn(x_batch)

        # ─────────────────────────────────────────────────────────────────────
        # 2) Residual 시퀀스 생성 (완전한 Rolling‑prediction, 벡터화)
        #    res_seq: (B, 12, 1370, 8)
        # ─────────────────────────────────────────────────────────────────────
        res_seq = torch.zeros((B, 12, num_nodes, 8), device=device, dtype=torch.float32)
        # (a) t = 0..10
        for t in range(0, 11):
            # all_X_global[idx_batch + t] → shape (B, 12, 1370, C_in)
            X_t_np = all_X_global[idx_batch + t]  # (B, 12, 1370, C_in)
            X_t = torch.from_numpy(X_t_np).permute(0, 3, 1, 2).float().to(device)  # (B, C_in, 12, 1370)
            y_pred_t = model.stgcn(X_t)  # (B, 8, 1370)
            # 실제 다음 스텝 Y: all_Y_global[idx_batch + t] → (B, 1370, 8)
            Y_true_np = all_Y_global[idx_batch + t]  # (B, 1370, 8)
            Y_true = torch.from_numpy(Y_true_np).to(device)  # (B, 1370, 8)
            y_pred_t_perm = y_pred_t.permute(0, 2, 1)  # (B, 1370, 8)
            res_seq[:, t, :, :] = Y_true - y_pred_t_perm

        # (b) t = 11: 현재 윈도우 마지막 스텝 사용 (벡터화)
        y_pred_last = y_pred_st_last.permute(0, 2, 1)  # (B, 1370, 8)
        y_true_last = y_batch.permute(0, 2, 1)         # (B, 1370, 8)
        res_seq[:, 11, :, :] = y_true_last - y_pred_last
        # ─────────────────────────────────────────────────────────────────────

        # ─────────────────────────────────────────────────────────────────────
        # 3) Residual LSTM 보정
        # ─────────────────────────────────────────────────────────────────────
        # weekend_flag: x_batch[:, 8, 0, 0] → (B,) 형태 (채널 8 = WeekendFlag)
        weekend_flag = x_batch[:, 8, 0, 0]  # (B,)
        res_corr = model.reslstm(res_seq, weekend_flag)  # (B, 8, 1370)

        # ─────────────────────────────────────────────────────────────────────
        # 4) 최종 예측: y_pred_st_last + res_corr
        # ─────────────────────────────────────────────────────────────────────
        y_final = y_pred_st_last + res_corr  # (B, 8, 1370)

        # ─────────────────────────────────────────────────────────────────────
        # 5) Loss 계산: MSE(y_final, y_batch)
        # ─────────────────────────────────────────────────────────────────────
        loss = criterion(y_final, y_batch)
        loss.backward()

        # Gradient clipping
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        # 두 옵티마이저 스텝
        optimizer_stgcn.step()
        optimizer_reslstm.step()

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
        for x_batch, y_batch, idx_batch in tqdm(val_loader, ncols=80, desc=f"[Epoch {epoch}/{args.epochs}] Val"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            B = x_batch.shape[0]

            # 1) ST-GCN 단기 예측
            y_pred_st_last = model.stgcn(x_batch)  # (B, 8, 1370)

            # 2) 잔차 생성 (간소화 버전: 마지막 스텝 잔차만)
            res_seq = torch.zeros((B, 12, num_nodes, 8), device=device)
            y_true_last = y_batch.permute(0, 2, 1)
            y_pred_last = y_pred_st_last.permute(0, 2, 1)
            res_seq[:, 11, :, :] = y_true_last - y_pred_last

            # 3) Residual LSTM 보정
            weekend_flag = x_batch[:, 8, 0, 0]  # (B,)
            res_corr = model.reslstm(res_seq, weekend_flag)

            # 4) 최종 예측
            y_final = y_pred_st_last + res_corr

            # 5) Loss 계산
            loss_val = criterion(y_final, y_batch)
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
    # D) EarlyStopping & Scheduler
    # ─────────────────────────────────────────────────────────────────────────
    early_stopper.step(val_loss)
    if early_stopper.stop:
        print(f"▶ Early stopping triggered at epoch {epoch}")
        break

    if args.sched == 'plateau':
        scheduler_stgcn.step(val_loss)  # type: ignore
        scheduler_reslstm.step(val_loss)  # type: ignore
    else:
        scheduler_stgcn.step(val_loss)       # type: ignore
       scheduler_reslstm.step(val_loss)      # type: ignore

    # ─────────────────────────────────────────────────────────────────────────
    # E) 체크포인트 저장
    # ─────────────────────────────────────────────────────────────────────────
    if epoch % 5 == 0:
        ckpt_path = os.path.join(args.checkpoint_dir, f'resstgcn_epoch{epoch:03d}.pt')
        torch.save({'epoch': epoch,
                    'stgcn_state_dict': model.stgcn.state_dict(),
                    'reslstm_state_dict': model.reslstm.state_dict(),
                    'optimizer_stgcn': optimizer_stgcn.state_dict(),
                    'optimizer_reslstm': optimizer_reslstm.state_dict(),
                    'val_loss': val_loss}, ckpt_path)
        print(f"▶ Checkpoint saved: {ckpt_path}")

print("▶ ResSTGCN Training finished.")