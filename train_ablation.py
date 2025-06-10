#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ablation.py

Ablation experiments for ResSTGCN:
1) baseline: ST-GCN only
2) residual-only: fix ST-GCN, train only residual LSTM
3) full: train both end-to-end

For each setting, plot train/validation loss curves and save to ablation_results/.
"""
import os, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from data_loader import get_dataloaders
from model.stgcn_model import STGCN
from model.pooled_residual_stgcn import PooledResSTGCN

# ─────────────────────────────────────────────────────────────
# 0) argparse 설정
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    '--mode',
    choices=['baseline','residual_only','full','all'],
    default='all',
    help='실행할 실험을 고릅니다.'
)

parser.add_argument(
    '--cluster_map',
    type=str,
    default='lane_to_segment_id_by_edge.npy',
    help='lane→segment ID 매핑(.npy)'
)

parser.add_argument(
    '--horizon',
    type=int,
    default=3,
    help='멀티스텝 예측 시 예측할 스텝 수 (H)'
)

args = parser.parse_args()
if args.mode == 'all':
    modes = ['baseline','residual_only','full']
else:
    modes = [args.mode]

# Hyperparameters
epochs = 30
batch_size = 8
lr_baseline = 5e-4
lr_residual = 1e-4
lr_full = 5e-4
clip_norm = 5.0

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare data
train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)
sample_x, *_ = next(iter(train_loader))
num_nodes = sample_x.shape[-1]

# -----------------------------
# cluster_map 로드
cluster_np   = np.load(args.cluster_map)
cluster_id   = torch.from_numpy(cluster_np).long().to(device)
num_clusters = int(cluster_id.max()) + 1
print(f"✔ Cluster mapping: nodes={num_nodes}, clusters={num_clusters}")
# -----------------------------

# adjacency matrix
dummy = np.load('3_tensor/adjacency/A_lane.npy')
A = torch.from_numpy(dummy).float().to(device)

# Make results dir
os.makedirs('ablation_results', exist_ok=True)
criterion = nn.MSELoss()

# 1) Baseline: ST-GCN only
def train_baseline():
    model = STGCN(
        in_channels=9,
        hidden1=64,
        out_channels=8,
        num_nodes=num_nodes,
        A=A,
        horizon=args.horizon
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_baseline, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    train_losses, val_losses = [], []
    for ep in range(1, epochs+1):
        model.train()
        t_loss = 0
        pbar = tqdm(train_loader, desc=f"[baseline] Train Ep{ep}", ncols=80)
        # DataLoader가 (x, y_last, y_seq, idx, date) 를 반환하므로 y_seq를 unpack
        for x, y_last, y_seq, idx, date in pbar:
            x = x.to(device)
            # y_seq: (B, T=12, C=8, N) → 처음 H 스텝만 정답으로 사용
            y_true = y_seq[:, :args.horizon, :, :].to(device)  # (B, H, C, N)
            optimizer.zero_grad()
            # 모델 출력: (B, C, H, N)
            y_hat = model(x)
            # (B, C, H, N) → (B, H, C, N) 로 맞춰서 loss 계산
            y_hat = y_hat.permute(0, 2, 1, 3)
            loss = criterion(y_hat, y_true)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            t_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train = t_loss / len(train_loader)
        train_losses.append(avg_train)
        pbar.close()
        # Validation
        model.eval()
        v_loss = 0
        pbar = tqdm(val_loader, desc=f"[baseline] Val   Ep{ep}", ncols=80)
        for x, y_last, y_seq, idx, date in pbar:
            x = x.to(device)
            y_true = y_seq[:, :args.horizon, :, :].to(device)
            y_hat = model(x)
            y_hat = y_hat.permute(0, 2, 1, 3)
            loss = criterion(y_hat, y_true)
            v_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_val = v_loss / len(val_loader)
        val_losses.append(avg_val)
        pbar.close()
        scheduler.step(avg_val)
        print(f"[baseline] Ep{ep:02d} Train_MSE={avg_train:.4f} Val_MSE={avg_val:.4f}")
    # Save pretrained ST-GCN backbone for residual-only
    torch.save(model.state_dict(), 'ablation_results/baseline_stgcn.pth')
    print("✔ Saved baseline ST-GCN weights to ablation_results/baseline_stgcn.pth")
    return train_losses, val_losses

# 2) Residual-only: fix backbone, train LSTM branch only
def train_residual_only():
    model = PooledResSTGCN(
        in_c=8, out_c=8, num_nodes=num_nodes,
        A=A, cluster_id=cluster_id, horizon=args.horizon,
        K=num_clusters,        # 매핑에서 알아낸 클러스터 수
        hidden_lstm=256
    ).to(device)
    # Load pretrained baseline ST-GCN weights, filtering out size mismatches
    raw_dict = torch.load('ablation_results/baseline_stgcn.pth', map_location=device)
    filtered_dict = {}
    for k, v in raw_dict.items():
        if k in model.stgcn.state_dict() and model.stgcn.state_dict()[k].shape == v.shape:
            filtered_dict[k] = v
    model.stgcn.load_state_dict(filtered_dict, strict=False)
    # Initial freeze of NetVLADPool parameters
    for p in model.pool.parameters():
        p.requires_grad = False
    # 1) backbone(ST-GCN)와 pooling, 보정기만 학습
    # model created with in_c=8 (8 traffic channels)
    for p in model.stgcn.parameters():
        p.requires_grad = False

    # Warm-up stages for residual-only training
    warmups = {
        'stage1': range(1, 4),
        'stage2': range(4, 7),
        'stage3': range(7, 10),
        'stage4': range(10, epochs+1)
    }

    # Initial optimizer: only post_fc.bias
    optimizer = optim.Adam(
        [{'params': model.clstm.post_fc.bias, 'lr': 1e-4}],
        weight_decay=1e-4
    )
    # Zero-initialize the post_fc bias of ClusterLSTM for stable start
    import torch.nn.init as init
    init.zeros_(model.clstm.post_fc.bias)
    # Warm-up configuration for δ learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    train_losses, val_losses = [], []
    for ep in range(1, epochs+1):
        # Layer-wise unfreeze and learning-rate groups
        if ep in warmups['stage2']:
            # Unfreeze pre_fc
            for p in model.clstm.pre_fc.parameters():
                p.requires_grad = True
            optimizer.add_param_group({'params': model.clstm.pre_fc.parameters(), 'lr': 5e-5})
        if ep in warmups['stage3']:
            # Unfreeze LSTM
            for p in model.clstm.lstm.parameters():
                p.requires_grad = True
            optimizer.add_param_group({'params': model.clstm.lstm.parameters(), 'lr': 1e-5})
        if ep in warmups['stage4']:
            # Unfreeze NetVLADPool
            for p in model.pool.parameters():
                p.requires_grad = True
            optimizer.add_param_group({'params': model.pool.parameters(), 'lr': 1e-6})

        model.train()
        # Update NetVLAD alpha schedule
        if hasattr(model.pool, 'update_alpha'):
            model.pool.update_alpha(ep, epochs)
        t_loss = 0
        pbar = tqdm(train_loader, desc=f"[residual_only] Train Ep{ep}", ncols=80)
        for x, y_true, idx, date in pbar:
            # x: (B, C_in, T, N), last channel is holiday_flag internally handled
            x = x.to(device)
            y_true = y_true.to(device)
            optimizer.zero_grad()
            # forward는 x와 residual_seq만 받습니다
            y_hat = model(x, residual_seq=y_true)
            y_hat = y_hat.permute(0, 2, 1, 3)
            loss  = criterion(y_hat, y_true)
            loss.backward()
            nn.utils.clip_grad_norm_(model.clstm.parameters(), 0.5)
            nn.utils.clip_grad_norm_(model.pool.parameters(), 0.5)
            optimizer.step()
            t_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train = t_loss / len(train_loader)
        train_losses.append(avg_train)
        pbar.close()
        # Validation
        model.eval()
        v_loss = 0
        pbar = tqdm(val_loader, desc=f"[residual_only] Val   Ep{ep}", ncols=80)
        for x, y_true, idx, date in pbar:
            x = x.to(device)
            y_true = y_true.to(device)
            y_hat = model(x, residual_seq=y_true)
            y_hat = y_hat.permute(0, 2, 1, 3)
            loss  = criterion(y_hat, y_true)
            v_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_val = v_loss / len(val_loader)
        val_losses.append(avg_val)
        pbar.close()
        scheduler.step(avg_val)
        print(f"[residual_only] Ep{ep:02d} Train_MSE={avg_train:.4f} Val_MSE={avg_val:.4f}")
    return train_losses, val_losses

# 3) Full training: both modules end-to-end
def train_full():
    model = PooledResSTGCN(
        in_c=8, out_c=8, num_nodes=num_nodes,
        A=A, cluster_id=cluster_id, horizon=args.horizon,
        K=num_clusters,
        hidden_lstm=256
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_full, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    train_losses, val_losses = [], []
    for ep in range(1, epochs+1):
        model.train()
        # Update NetVLAD alpha schedule
        if hasattr(model.pool, 'update_alpha'):
            model.pool.update_alpha(ep, epochs)
        t_loss = 0
        pbar = tqdm(train_loader, desc=f"[full] Train Ep{ep}", ncols=80)
        for x, y_last, y_seq, idx, date in pbar:
            x      = x.to(device)
            y_last = y_last.to(device)      # (B, out_c, N)
            y_seq  = y_seq.to(device)       # (B, T, out_c, N)
            # use the first H steps of the true sequence from the DataLoader
            y_true = y_seq[:, :args.horizon, :, :].contiguous()  # (B, H, C, N)
            optimizer.zero_grad()
            y_hat = model(x, residual_seq=y_true)
            y_hat = y_hat.permute(0, 2, 1, 3)
            loss = criterion(y_hat, y_true)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            t_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train = t_loss / len(train_loader)
        train_losses.append(avg_train)
        pbar.close()
        # Validation
        model.eval()
        v_loss = 0
        pbar = tqdm(val_loader, desc=f"[full] Val   Ep{ep}", ncols=80)
        for x, y_last, y_seq, idx, date in pbar:
            x = x.to(device)
            y_last = y_last.to(device)
            y_seq = y_seq.to(device)
            y_true = y_seq[:, :args.horizon, :, :].contiguous()
            y_hat = model(x, residual_seq=y_true)
            y_hat = y_hat.permute(0, 2, 1, 3)
            loss  = criterion(y_hat, y_true)
            v_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_val = v_loss / len(val_loader)
        val_losses.append(avg_val)
        pbar.close()
        scheduler.step(avg_val)
        print(f"[full] Ep{ep:02d} Train_MSE={avg_train:.4f} Val_MSE={avg_val:.4f}")
    return train_losses, val_losses

results = {}
for mode in modes:
    print(f"\n=== Running {mode} ===")
    if   mode == 'baseline':      t, v = train_baseline()
    elif mode == 'residual_only': t, v = train_residual_only()
    else:                         t, v = train_full()
    results[mode] = (t, v)
    np.savez(f"ablation_results/{mode}_losses.npz", train=np.array(t), val=np.array(v))
    print(f"[{mode}] 완료: train/val 곡선 저장 -> ablation_results/{mode}_losses.npz")


# Plot combined
plt.figure()
epochs_arr = np.arange(1, epochs+1)
for mode,(t,v) in results.items(): plt.plot(epochs_arr, v, label=f"{mode} val")
plt.xlabel('Epoch'); plt.ylabel('Val MSE'); plt.title('Ablation: Val Loss')
plt.legend(); plt.tight_layout()
plt.savefig('ablation_results/ablation_val_comparison.png')
print("Ablation experiments finished. Curves saved under ablation_results/")
    # model created with in_c=8 (8 traffic channels)