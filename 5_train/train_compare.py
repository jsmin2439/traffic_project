#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_compare.py

네 가지 모델을 동일한 조건에서 학습·평가하며,
각 모델별 서브폴더에 체크포인트를 저장하고
학습 진행 상황을 실시간 플롯으로 보여줍니다.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# 프로젝트 루트를 PYTHONPATH에 추가
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_loader import get_dataloaders
from model.stgcn_model import STGCN
from model.lstm_model import BasicLSTM
from model.gated_fusion_stgcn import GatedFusionSTGCN
from model.pooled_residual_stgcn import PooledResSTGCN

# 모델별 기본 하이퍼파라미터
MODEL_PARAMS = {
    'stgcn': { 'lr': 5e-4, 'hidden1': 64 },
    'lstm':  { 'lr': 1e-3, 'hidden2': 128 },
    'gated': { 'lr': 5e-4, 'hidden1': 64,  'hidden2': 128 },
    'pooled':{ 'lr': 5e-4, 'hidden1': 32,  'hidden2': 64 }
}

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH     = 8
EPOCHS    = 40
CLIP_NORM = 5.0
BASE_CKPT = 'checkpoints'


def ensure_subfolder(name):
    path = os.path.join(BASE_CKPT, name)
    os.makedirs(path, exist_ok=True)
    return path


def train_one_epoch(model, loader, criterion, optimizer, model_name, epoch):
    model.train()
    total_loss = 0.0
    for x, y, idx, date in tqdm(loader, desc=f"[{model_name}] Train Ep{epoch}", ncols=80):
        x, y = x.to(DEVICE), y.to(DEVICE)
        weekend = x[:, -1, 0, 0]
        optimizer.zero_grad()
        if model_name == 'lstm':
            y_hat = model(x)  # y_hat: (B, 8, N)
        elif model_name == 'stgcn':
            # ST-GCN outputs (B, 8, 1, N), squeeze time dimension for one-step prediction
            y_hat = model(x).squeeze(2)  # (B, 8, N)
        elif model_name == 'gated':
            # Positional arg for weekend_flag
            y_hat = model(x, weekend.float())
        else:
            # pooled: one-step forecast
            ema_r = y.unsqueeze(1)
            y_hat = model(x, ema_r, weekend.float()).squeeze(2)
        loss = criterion(y_hat, y)
        loss.backward()
        if CLIP_NORM > 0:
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(loader)
    # 체크포인트 저장
    sub = ensure_subfolder(model_name)
    if epoch % 5 == 0:
        ckpt = os.path.join(sub, f"epoch{epoch:03d}.pt")
        torch.save(model.state_dict(), ckpt)
    return avg

@torch.no_grad()
def evaluate(model, loader, criterion, model_name):
    model.eval()
    total_loss = 0.0
    for x, y, idx, date in tqdm(loader, desc=f"[{model_name}] Val", ncols=80):
        x, y = x.to(DEVICE), y.to(DEVICE)
        weekend = x[:, -1, 0, 0]
        if model_name == 'lstm':
            y_hat = model(x)
        elif model_name == 'stgcn':
            y_hat = model(x).squeeze(2)
        elif model_name == 'gated':
            y_hat = model(x, weekend.float())
        else:
            ema_r = y.unsqueeze(1)
            y_hat = model(x, y_true=ema_r, weekend_flag=weekend.float()).squeeze(2)
        total_loss += criterion(y_hat, y).item()
    return total_loss / len(loader)


def build_model(name, num_nodes, A):
    p = MODEL_PARAMS[name]
    if name == 'stgcn':
        return STGCN(in_channels=9, hidden1=p['hidden1'], out_channels=8,
                     num_nodes=num_nodes, A=A).to(DEVICE)
    if name == 'lstm':
        return BasicLSTM(num_nodes=num_nodes, input_dim=8,
                         hidden_dim=p['hidden2']).to(DEVICE)
    if name == 'gated':
        return GatedFusionSTGCN(in_channels=9,
                                hidden1=p['hidden1'],
                                hidden2=p['hidden2'],
                                out_channels=8,
                                num_nodes=num_nodes,
                                A=A).to(DEVICE)
    # pooled (one-step forecast)
    cluster_id = torch.zeros(num_nodes, dtype=torch.long)
    return PooledResSTGCN(
        in_c=9,
        out_c=8,
        num_nodes=num_nodes,
        A=A,
        cluster_id=cluster_id,
        K=p['hidden1'],
        hidden_lstm=p['hidden2'],
        horizon=1                # ← 여기만 1로 바꿔주세요
    ).to(DEVICE)


def plot_progress(train_losses, val_losses, model_name):
    plt.figure()
    epochs = np.arange(1, len(train_losses)+1)
    plt.plot(epochs, train_losses, label='train')
    plt.plot(epochs, val_losses,   label='val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'{model_name.upper()} Training Progress')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_progress.png')
    plt.close()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="Train specific model or all models.")
    parser.add_argument('--model', choices=['lstm','stgcn','gated','pooled','all'], default='all',
                        help="Which model to train (default: all)")
    args = parser.parse_args()
    model_list = ['lstm','stgcn','gated','pooled'] if args.model == 'all' else [args.model]

    # 데이터 로더 한번만 생성
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH)
    sample_x, _, *_ = next(iter(train_loader))
    num_nodes = sample_x.shape[-1]
    A = torch.from_numpy(np.load('3_tensor/adjacency/A_lane.npy')).float().to(DEVICE)

    os.makedirs('results', exist_ok=True)
    criterion = nn.MSELoss()

    for model_name in model_list:
        print(f"\n=== Training {model_name.upper()} ===")
        model = build_model(model_name, num_nodes, A)
        print(f"Parameter count: {count_params(model)}")
        lr = MODEL_PARAMS[model_name]['lr']
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

        train_losses, val_losses = [], []
        for epoch in range(1, EPOCHS+1):
            tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, model_name, epoch)
            va_loss = evaluate(model, val_loader, criterion, model_name)
            scheduler.step(va_loss)
            train_losses.append(tr_loss)
            val_losses.append(va_loss)
            print(f"[{model_name:6s}] Ep{epoch:02d} tr={tr_loss:.4f} va={va_loss:.4f}")

        plot_progress(train_losses, val_losses, model_name)

        test_loss = evaluate(model, test_loader, criterion, model_name)
        print(f"[{model_name:6s}] TEST MSE: {test_loss:.4f}")

        # 결과 저장
        np.savez(f"results/{model_name}_losses.npz",
                 train=np.array(train_losses),
                 val=np.array(val_losses),
                 test=test_loss)

if __name__ == '__main__':
    main()
