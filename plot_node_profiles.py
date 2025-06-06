#!/usr/bin/env python3
# plot_node_profiles.py

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_node_daily_profiles(date: int, node_idx: int, results_dict: dict, out_dir: str):
    """
    특정 날짜(date: YYYYMMDD)와 특정 노드(node_idx: 0~1369)에 대하여,
    (1) 교통량(Queue 채널 0~3)
    (2) 속도(Speed 채널 4~7)
    를 차종별(각 채널별)로 실제값과 모델별 예측값을 시간대별(0~287 슬롯)로 그립니다.

    Args:
        date (int) : YYYYMMDD 형태의 날짜 (예: 20220810)
        node_idx (int) : 노드 인덱스 (0~1369)
        results_dict (dict) :
            'lstm': {
                'preds_orig': np.ndarray [M_sel, 1370, 8],
                'trues_orig': np.ndarray [M_sel, 1370, 8],
                'dates_sel':  np.ndarray [M_sel,] (int YYYYMMDD),
                'slot_idx':   np.ndarray [M_sel,] (0~287),
                'speeds_orig':np.ndarray [M_sel,1370,4]  # 속도 채널 (생략해도 무방)
            },
            'stgcn': { … 같은 구조 … },
            'resstgcn': { … 같은 구조 … }
        out_dir (str) : PNG 파일을 저장할 디렉터리 (예: "eval/results/230101_1230/")
    """
    models = ['lstm', 'stgcn', 'resstgcn']
    data_per_model = {
        m: {
            'slot':  [],
            'true_queue':  [],
            'true_speed':  [],
            'pred_queue':  [],
            'pred_speed':  []
        }
        for m in models
    }

    # 1) 날짜 필터링 및 데이터 수집
    for m in models:
        preds = results_dict[m]['preds_orig']   # (M_sel,1370,8)
        trues = results_dict[m]['trues_orig']   # (M_sel,1370,8)
        dates = results_dict[m]['dates_sel']    # (M_sel,)
        slots = results_dict[m]['slot_idx']     # (M_sel,)
        speeds = results_dict[m].get('speeds_orig', None)

        mask = (dates == date)
        if not np.any(mask):
            raise ValueError(f"[{m.upper()}] 날짜 {date} 데이터가 없습니다.")
        idxs = np.where(mask)[0]

        for i in idxs:
            s = int(slots[i])  # 슬롯 인덱스 (0~287)
            true_vals = trues[i, node_idx, :]   # (8,)
            pred_vals = preds[i, node_idx, :]   # (8,)

            # 채널 0~3: Queue
            tq = true_vals[0:4]
            pq = pred_vals[0:4]
            # 채널 4~7: Speed
            ts = true_vals[4:8]
            ps = pred_vals[4:8]

            data_per_model[m]['slot'].append(s)
            data_per_model[m]['true_queue'].append(tq)
            data_per_model[m]['pred_queue'].append(pq)
            data_per_model[m]['true_speed'].append(ts)
            data_per_model[m]['pred_speed'].append(ps)

        # NumPy 배열화 후 슬롯 오름차순 정렬
        for key in ['slot','true_queue','pred_queue','true_speed','pred_speed']:
            arr = np.stack(data_per_model[m][key], axis=0)
            order = np.argsort(arr if key=='slot' else data_per_model[m]['slot'])
            data_per_model[m][key] = arr[order] if key!='slot' else arr[order]

    # 2) 전체 0~287 슬롯 길이 배열 준비
    slots_all = np.arange(288)
    veh_labels = [f'ch{i}' for i in range(8)]

    # 3) 교통량(0~3) 그리기
    fig_q, axes_q = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for ch in range(4):
        ax = axes_q[ch]
        for m in models:
            slot_m = data_per_model[m]['slot']             # (N_m,)
            true_q_m = data_per_model[m]['true_queue'][:, ch]
            pred_q_m = data_per_model[m]['pred_queue'][:, ch]

            arr_true = np.full((288,), np.nan, dtype=float)
            arr_pred = np.full((288,), np.nan, dtype=float)
            arr_true[slot_m] = true_q_m
            arr_pred[slot_m] = pred_q_m

            ax.plot(slots_all, arr_true,  label=f"{m.upper()} TRUE",   linestyle='-',  lw=1)
            ax.plot(slots_all, arr_pred,  label=f"{m.upper()} PRED",   linestyle='--', lw=1)

        ax.set_ylabel(f"Channel {ch} Volume")
        ax.set_title(f"Node {node_idx} - {veh_labels[ch]} (Volume) on {date}")
        ax.grid(alpha=0.3)
        if ch == 0:
            ax.legend(fontsize=8, ncol=2, loc='upper right')

    axes_q[-1].set_xlabel("Slot Index (0~287)")
    fig_q.tight_layout()
    fig_q.savefig(f"{out_dir}/node_{node_idx}_{date}_volume.png", dpi=300)
    plt.close(fig_q)

    # 4) 속도(4~7) 그리기
    fig_s, axes_s = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for idx_ch, ch in enumerate(range(4, 8)):
        ax = axes_s[idx_ch]
        for m in models:
            slot_m = data_per_model[m]['slot']
            true_s_m = data_per_model[m]['true_speed'][:, idx_ch]
            pred_s_m = data_per_model[m]['pred_speed'][:, idx_ch]

            arr_true = np.full((288,), np.nan, dtype=float)
            arr_pred = np.full((288,), np.nan, dtype=float)
            arr_true[slot_m] = true_s_m
            arr_pred[slot_m] = pred_s_m

            ax.plot(slots_all, arr_true,  label=f"{m.upper()} TRUE",   linestyle='-',  lw=1)
            ax.plot(slots_all, arr_pred,  label=f"{m.upper()} PRED",   linestyle='--', lw=1)

        ax.set_ylabel(f"Channel {ch} Speed")
        ax.set_title(f"Node {node_idx} - {veh_labels[ch]} (Speed) on {date}")
        ax.grid(alpha=0.3)
        if idx_ch == 0:
            ax.legend(fontsize=8, ncol=2, loc='upper right')

    axes_s[-1].set_xlabel("Slot Index (0~287)")
    fig_s.tight_layout()
    fig_s.savefig(f"{out_dir}/node_{node_idx}_{date}_speed.png", dpi=300)
    plt.close(fig_s)

    print(f"✔ Node {node_idx} @ {date}: Volume & Speed charts saved under {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="특정 날짜+노드에 대한 교통량/속도 (실제 vs 예측) 프로필을 그립니다."
    )
    parser.add_argument(
        "--results", "-r",
        required=True,
        help="results_dict.pkl 파일 경로 (예: eval/results/230101_1230/results_dict.pkl)"
    )
    parser.add_argument(
        "--date", "-d",
        type=int,
        required=True,
        help="YYYYMMDD 형식의 날짜 (예: 20220810)"
    )
    parser.add_argument(
        "--node", "-n",
        type=int,
        required=True,
        help="노드 인덱스 (0~1369)"
    )
    parser.add_argument(
        "--outdir", "-o",
        default=".",
        help="출력 PNG를 저장할 디렉터리 (기본값: 현재 디렉터리)"
    )
    args = parser.parse_args()

    # 1) 피클 결과 로드
    pkl_path = Path(args.results)
    if not pkl_path.exists():
        raise FileNotFoundError(f"결과 피클 파일을 찾을 수 없습니다: {pkl_path}")
    with open(pkl_path, "rb") as f:
        results_dict = pickle.load(f)

    # 2) 폴더가 없으면 생성
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3) 플롯 생성
    plot_node_daily_profiles(
        date=args.date,
        node_idx=args.node,
        results_dict=results_dict,
        out_dir=str(out_dir)
    )


if __name__ == "__main__":
    main()