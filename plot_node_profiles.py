#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_node_profiles.py

특정 날짜·노드에 대한 채널별(Queue/Speed) 일별 프로필(실제 vs 예측)을 그립니다.
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_node_daily_profiles(date: int, node_idx: int, results_dict: dict, out_dir: str):
    models = ['lstm', 'stgcn', 'resstgcn']
    # 1) 데이터 수집
    data = {m: {'slot': [], 'true_q': [], 'pred_q': [], 'true_s': [], 'pred_s': []} 
            for m in models}

    for m in models:
        preds = results_dict[m]['preds_orig']
        trues = results_dict[m]['trues_orig']
        dates = results_dict[m]['dates_sel']
        slots = results_dict[m]['slot_idx']

        mask = (dates == date)
        if not mask.any():
            raise ValueError(f"[{m}] 날짜 {date} 데이터 없음")
        idxs = np.where(mask)[0]

        for i in idxs:
            s = int(slots[i])
            tv = trues[i, node_idx]  # (8,)
            pv = preds[i, node_idx]  # (8,)
            data[m]['slot'].append(s)
            data[m]['true_q'].append(tv[:4])
            data[m]['pred_q'].append(pv[:4])
            data[m]['true_s'].append(tv[4:])
            data[m]['pred_s'].append(pv[4:])

        # NumPy 배열로 변환
        data[m]['slot']   = np.array(data[m]['slot'], dtype=int)
        data[m]['true_q'] = np.stack(data[m]['true_q'], axis=0)  # (n_samples,4)
        data[m]['pred_q'] = np.stack(data[m]['pred_q'], axis=0)
        data[m]['true_s'] = np.stack(data[m]['true_s'], axis=0)
        data[m]['pred_s'] = np.stack(data[m]['pred_s'], axis=0)

        # 슬롯 기준으로 정렬
        order = np.argsort(data[m]['slot'])
        data[m]['slot']   = data[m]['slot'][order]
        for key in ['true_q','pred_q','true_s','pred_s']:
            data[m][key] = data[m][key][order]

    # 2) 그리기 준비
    slots_all = np.arange(288)
    veh_labels = [f'ch{i}' for i in range(8)]

    # 3) Queue(채널 0~3)
    fig, axes = plt.subplots(4,1, figsize=(8,10), sharex=True)
    for ch in range(4):
        ax = axes[ch]
        for m in models:
            s = data[m]['slot']
            tq = data[m]['true_q'][:,ch]
            pq = data[m]['pred_q'][:,ch]
            arr_t = np.full(288, np.nan)
            arr_p = np.full(288, np.nan)
            arr_t[s] = tq
            arr_p[s] = pq
            ax.plot(slots_all, arr_t, label=f"{m.upper()} TRUE", linestyle='-', lw=1)
            ax.plot(slots_all, arr_p, label=f"{m.upper()} PRED", linestyle='--', lw=1)
        ax.set_ylabel(f"Channel {ch} Volume")
        ax.grid(alpha=0.3)
        if ch==0:
            ax.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("Slot (0~287)")
    fig.tight_layout()
    fig.savefig(Path(out_dir)/f"node_{node_idx}_{date}_volume.png", dpi=300)
    plt.close(fig)

    # 4) Speed(채널 4~7)
    fig, axes = plt.subplots(4,1, figsize=(8,10), sharex=True)
    for idx_ch, ch in enumerate(range(4)):
        ax = axes[idx_ch]
        for m in models:
            s = data[m]['slot']
            ts = data[m]['true_s'][:,ch]
            ps = data[m]['pred_s'][:,ch]
            arr_t = np.full(288, np.nan)
            arr_p = np.full(288, np.nan)
            arr_t[s] = ts
            arr_p[s] = ps
            ax.plot(slots_all, arr_t, label=f"{m.upper()} TRUE", linestyle='-', lw=1)
            ax.plot(slots_all, arr_p, label=f"{m.upper()} PRED", linestyle='--', lw=1)
        ax.set_ylabel(f"Channel {ch+4} Speed")
        ax.grid(alpha=0.3)
        if idx_ch==0:
            ax.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("Slot (0~287)")
    fig.tight_layout()
    fig.savefig(Path(out_dir)/f"node_{node_idx}_{date}_speed.png", dpi=300)
    plt.close(fig)

    print(f"✔ Charts saved to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--results', required=True,
                        help="results_dict.pkl 경로")
    parser.add_argument('-d','--date',   type=int, required=True,
                        help="YYYYMMDD")
    parser.add_argument('-n','--node',   type=int, required=True,
                        help="노드 인덱스")
    parser.add_argument('-o','--outdir', default=".",
                        help="출력 디렉터리")
    args = parser.parse_args()

    results = pickle.load(open(args.results,'rb'))
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    plot_node_daily_profiles(args.date, args.node, results, str(out))

if __name__ == "__main__":
    main()