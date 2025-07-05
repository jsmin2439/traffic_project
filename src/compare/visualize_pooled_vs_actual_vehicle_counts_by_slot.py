#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_pooled_vs_actual_vehicle_counts_by_slot.py

PooledResSTGCN 모델이 예측한 채널 0~3 (차량 대기수)
과 실제 값을 5분 단위 슬롯별로 합산해 비교 시각화합니다.

Usage:
    python visualize_pooled_vs_actual_vehicle_counts_by_slot.py \
        --pooled_ckpt ck_pooled/pooled_ep040.pt \
        --cluster_map segment/lane_to_segment_id_by_edge.npy \
        --windows_dir 3_tensor/windows \
        --batch 64 \
        --out_png pooled_vs_actual_counts.png \
        --dpi 150
"""
import os
import argparse
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 자동 설정: 시스템에 설치된 폰트 중 우선순위로 선택
font_candidates = ['NanumGothic', 'AppleGothic', 'Malgun Gothic', 'Noto Sans CJK KR']
available = {f.name for f in fm.fontManager.ttflist}
for fname in font_candidates:
    if fname in available:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [fname]
        break
else:
    # 폰트 설치되지 않은 경우 경고 후 default
    import warnings
    warnings.warn(f"None of the Hangul fonts found: {font_candidates}")
# 음수 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

from model.pooled_residual_stgcn import PooledResSTGCN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pooled_ckpt', required=True)
    p.add_argument('--cluster_map', required=True)
    p.add_argument('--windows_dir', default='3_tensor/windows')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--out_png', default='pooled_vs_actual_counts.png')
    p.add_argument('--dpi', type=int, default=150)
    return p.parse_args()


def load_model(ckpt_path, cluster_map_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    A = torch.from_numpy(np.load('3_tensor/adjacency/A_lane.npy')).float().to(device)
    cluster_np = np.load(cluster_map_path)
    cluster_id = torch.from_numpy(cluster_np).long().to(device)
    in_c = ckpt['model_state']['stgcn.layer1.temp1.weight'].shape[1]
    hidden_lstm = ckpt['model_state']['clstm.pre_fc.weight'].shape[0]
    model = PooledResSTGCN(
        in_c=in_c,
        out_c=8,
        num_nodes=A.size(0),
        A=A,
        cluster_id=cluster_id,
        K=cluster_id.max().item()+1,
        hidden_lstm=hidden_lstm,
        horizon=1
    ).to(device)
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load windows
    all_X = np.load(os.path.join(args.windows_dir, 'all_X.npy'))   # (Nwin,12,1370,9)
    all_Y = np.load(os.path.join(args.windows_dir, 'all_Y.npy'))   # (Nwin,1370,8)
    all_DATE = np.load(os.path.join(args.windows_dir, 'all_DATE.npy'))  # (Nwin,)
    Nwin = all_X.shape[0]
    dates = np.unique(all_DATE)
    windows_per_day = Nwin // len(dates)

    # Compute per-date normalization stats from raw input tensors
    date_stats = {}
    for d in dates:
        # load raw input tensor with flag (channels 0-7 raw queue+spd, 8 flag)
        raw_path = os.path.join('3_tensor', str(int(d)), f'input_tensor_{int(d)}_with_flag.npy')
        raw = np.load(raw_path)  # shape=(288,1370,9)
        # channels 0-7 are raw values
        ch_data = raw[:, :, :8]
        means = ch_data.reshape(-1, 8).mean(axis=0)
        stds  = ch_data.reshape(-1, 8).std(axis=0)
        date_stats[int(d)] = (means, stds)

    # Prepare per-slot accumulators
    slot_true = {}   # slot -> list of shape-(4,) arrays
    slot_pred = {}
    for s in range(12, 288):
        slot_true[s] = []
        slot_pred[s] = []

    # Load model
    model = load_model(args.pooled_ckpt, args.cluster_map, device)

    # Inference by batch
    for start in range(0, Nwin, args.batch):
        end = min(Nwin, start + args.batch)
        x_batch = all_X[start:end]  # (B,12,1370,9)
        y_batch = all_Y[start:end]  # (B,1370,8)
        # model input
        x_t = torch.from_numpy(x_batch).permute(0,3,1,2).float().to(device)  # (B,9,12,1370)
        res = torch.from_numpy(y_batch).unsqueeze(1).permute(0,1,3,2).float().to(device)  # (B,1,8,1370)
        with torch.no_grad():
            p = model(x_t, residual_seq=res).squeeze(2).cpu().numpy()  # (B,8,1370)
        # transpose to (B,1370,8)
        p = p.transpose(0,2,1)
        for i, global_idx in enumerate(range(start, end)):
            slot = (global_idx % windows_per_day) + 12
            # get per-date normalization stats
            cur_date = int(all_DATE[global_idx])
            means, stds = date_stats[cur_date]  # each shape (8,)

            # denormalize true values (normalized Y)
            y_norm = all_Y[global_idx]             # (1370,8)
            y_raw = y_norm * stds[np.newaxis, :] + means[np.newaxis, :]  # (1370,8)

            # denormalize predicted values
            p_norm = p[i]                          # (1370,8)
            p_raw = p_norm * stds[np.newaxis, :] + means[np.newaxis, :]  # (1370,8)

            # sum over nodes for queue_count channels 0-3
            true_counts = y_raw[:, :4].sum(axis=0)   # (4,)
            pred_counts = p_raw[:, :4].sum(axis=0)

            slot_true[slot].append(true_counts)
            slot_pred[slot].append(pred_counts)

    # Aggregate mean per slot
    slots = sorted(slot_true.keys())
    true_mean = np.array([np.stack(slot_true[s],0).mean(axis=0) for s in slots])  # (slots,4)
    pred_mean = np.array([np.stack(slot_pred[s],0).mean(axis=0) for s in slots])

    # Time labels
    time_slots = np.array(slots)
    hours = (time_slots * 5) // 60
    minutes = (time_slots * 5) % 60
    labels = [f"{h:02d}:{m:02d}" for h,m in zip(hours, minutes)]

    # 각 차종별로 개별 플롯 생성
    categories = ['Car', 'Bus', 'Truck', 'Motorcycle']
    base_out = args.out_png.rstrip('.png')
    for i, name in enumerate(categories):
        fig, ax = plt.subplots(figsize=(12, 6))
        # 실제 vs 예측 곡선
        ax.plot(time_slots, true_mean[:, i], label=f"Actual {name}")
        ax.plot(time_slots, pred_mean[:, i], '--', label=f"Predicted {name}")

        ax.set_xticks(time_slots[::12])
        ax.set_xticklabels(labels[::12], rotation=45)
        ax.set_xlabel('Time (HH:MM)')
        ax.set_ylabel('Average Queue Count')
        ax.set_title(f'Time-of-Day: Actual vs Predicted Queue Count for {name}')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        out_file = f"{base_out}_{name.lower()}.png"
        fig.savefig(out_file, dpi=args.dpi)
        print(f"Saved {name} plot to {out_file}")
        plt.close(fig)

if __name__ == '__main__':
    main()
