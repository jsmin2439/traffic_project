# run_eval.py

import os
import yaml
import datetime
import pytz
import argparse
import numpy as np
import pickle
from pathlib import Path

from eval.evaluator import evaluate
import warnings
from eval import plots

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True, help='Path to YAML configuration file')
args = parser.parse_args()

# ----------------------------
# 1) Load configuration
# ----------------------------
cfg = yaml.safe_load(open(args.cfg))
seoul_tz = pytz.timezone('Asia/Seoul')
now_kst = datetime.datetime.now(seoul_tz)
stamp = now_kst.strftime('%y%m%d_%H%M')
save_root = Path("eval/results") / stamp
save_root.mkdir(parents=True, exist_ok=True)
cfg['save_dir'] = str(save_root)

# ───────────────────────────────────────────────────────────────────
# 불필요한 seaborn FutureWarning 무시 (원한다면 주석 처리 가능)
warnings.simplefilter("ignore", category=FutureWarning)
# ───────────────────────────────────────────────────────────────────


# ----------------------------
# 2) Run evaluation → CSV + pickled result dict 생성
# ----------------------------
# evaluator.evaluate() 함수는 res_dir(Path)를 반환합니다.
# 우리는 추가 plotting을 위해, evaluator 내부에서 predictions/truths 등을
# results_dict 형태로 pickle로 저장했다고 가정합니다.
#
# 예: evaluator.py 끝 부분에
#     with open(os.path.join(res_dir, 'results_dict.pkl'), 'wb') as f:
#         pickle.dump(results_dict, f)
#
res_dir = evaluate(cfg)  # → eval/results/YYMMDD_HHMM/

# ----------------------------
# 3) Load pickled results_dict
# ----------------------------
# results_dict 구조 예시:
# {
#   'lstm': {
#       'preds_orig': np.ndarray(shape=(M_sel,1370,8)),
#       'trues_orig': np.ndarray(shape=(M_sel,1370,8)),
#       'dates_sel':  np.ndarray(shape=(M_sel,)),
#       'slot_idx':   np.ndarray(shape=(M_sel,)),       # 각 윈도우의 하루 슬롯 인덱스(0~287)
#       'is_weekend': np.ndarray(shape=(M_sel,)),       # 0 or 1
#       'speeds_orig':np.ndarray(shape=(M_sel,1370,4))  # 각 윈도우의 raw 속도 채널 (denorm)
#   },
#   'stgcn': { ... },
#   'resstgcn': { ... },
#   # 그 외, epoch별 글로벌 지표 및 노드별 RMSE 정보 (Epoch 분석용)
#   'epoch_list': [5,10,15,20,25,30,35,40],
#   'metrics_epoch': {
#       'lstm': {'RMSE': [...], 'MAE':[...], 'MAPE':[...], 'R2':[...], },
#       'stgcn': {...}, 'resstgcn': {...}
#   },
#   'node_epoch_rmse': {
#       'lstm': { node_idx: [rmse_ep5, rmse_ep10, ...], ... },
#       'stgcn': {...}, 'resstgcn': {...}
#   }
# }
#
results_dict_path = res_dir / "results_dict.pkl"
if not results_dict_path.exists():
    raise FileNotFoundError(f"Expected pickled results_dict at {results_dict_path}")
with open(results_dict_path, 'rb') as f:
    results_dict = pickle.load(f)

# ----------------------------
# 4) Generate all plots
# ----------------------------

print("\n▶ Generating all plots...\n")

# 4.1 Global metrics bar chart
print("▶ Generating: Global metrics bar chart...")
plots.plot_global_bar(
    csv_path=str(res_dir / 'metrics_global.csv'),
    out_png=str(res_dir / 'global_bar.png')
)

print("→ Done: global_bar.png\n")

# 4.2 Channel-wise radar chart
print("▶ Generating: Channel-wise radar chart...")
plots.plot_channel_radar(
    csv_dir=str(res_dir),
    out_png=str(res_dir / 'radar_channel.png')
)
print("→ Done: radar_channel.png\n")

# 4.3 Node×Channel heatmap (denormalized RMSE)
print("▶ Generating: Node×Channel heatmap...")
plots.plot_node_channel_heatmap(
    results_dict={
        'lstm':    {'preds_orig': results_dict['lstm']['preds_orig'],
                    'trues_orig': results_dict['lstm']['trues_orig']},
        'stgcn':   {'preds_orig': results_dict['stgcn']['preds_orig'],
                    'trues_orig': results_dict['stgcn']['trues_orig']},
        'resstgcn':{'preds_orig': results_dict['resstgcn']['preds_orig'],
                    'trues_orig': results_dict['resstgcn']['trues_orig']}
    },
    out_png=str(res_dir / 'heatmap_node_channel.png'),
    normalize=False
)
print("→ Done: heatmap_node_channel.png\n")

# 4.4 Window-wise RMSE trend (전체 윈도우)
print("▶ Generating: Window-wise RMSE trend...")
plots.plot_window_rmse_trend(
     results_dict={
        'lstm':    {'preds_orig': results_dict['lstm']['preds_orig'],
                    'trues_orig': results_dict['lstm']['trues_orig']},
        'stgcn':   {'preds_orig': results_dict['stgcn']['preds_orig'],
                    'trues_orig': results_dict['stgcn']['trues_orig']},
        'resstgcn':{'preds_orig': results_dict['resstgcn']['preds_orig'],
                    'trues_orig': results_dict['resstgcn']['trues_orig']}
    },
    out_png=str(res_dir / 'window_rmse_trend_all.png'),
    window_indices=None
)
print("→ Done: window_rmse_trend_all.png\n")

# 4.5 Diurnal ribbon plot (일교차)
print("▶ Generating: Diurnal ribbon plot...")
plots.plot_diurnal_ribbon(
    results_dict={
        'lstm':   {'preds_orig': results_dict['lstm']['preds_orig'],
                   'trues_orig': results_dict['lstm']['trues_orig'],
                   'slot_idx':   results_dict['lstm']['slot_idx']},
        'stgcn':  {'preds_orig': results_dict['stgcn']['preds_orig'],
                   'trues_orig': results_dict['stgcn']['trues_orig'],
                   'slot_idx':   results_dict['stgcn']['slot_idx']},
        'resstgcn': {'preds_orig': results_dict['resstgcn']['preds_orig'],
                     'trues_orig': results_dict['resstgcn']['trues_orig'],
                     'slot_idx':   results_dict['resstgcn']['slot_idx']}
    },
    out_png=str(res_dir / 'diurnal_ribbon.png')
)
print("→ Done: diurnal_ribbon.png\n")

# 4.6 Weekday vs Weekend boxplot + p-value
print("▶ Generating: Weekday vs Weekend boxplot...")
plots.plot_weekday_vs_weekend_box(
    results_dict={
        'lstm':   {'preds_orig': results_dict['lstm']['preds_orig'],
                   'trues_orig': results_dict['lstm']['trues_orig'],
                   'dates_sel':  results_dict['lstm']['dates_sel'],
                   'is_weekend': results_dict['lstm']['is_weekend']},
        'stgcn':  {'preds_orig': results_dict['stgcn']['preds_orig'],
                   'trues_orig': results_dict['stgcn']['trues_orig'],
                   'dates_sel':  results_dict['stgcn']['dates_sel'],
                   'is_weekend': results_dict['stgcn']['is_weekend']},
        'resstgcn': {'preds_orig': results_dict['resstgcn']['preds_orig'],
                    'trues_orig': results_dict['resstgcn']['trues_orig'],
                    'dates_sel':  results_dict['resstgcn']['dates_sel'],
                    'is_weekend': results_dict['resstgcn']['is_weekend']}
    },
    out_png=str(res_dir / 'weekday_vs_weekend_box.png'),
    pvalue_annot=True
)
print("→ Done: weekday_vs_weekend_box.png\n")

# 4.7 Speed-level RMSE bar chart (저속/중/고속)
print("▶ Generating: Speed-level RMSE bar chart...")
plots.plot_speed_level_bar(
    results_dict={
        'lstm':   {'preds_orig': results_dict['lstm']['preds_orig'],
                   'trues_orig': results_dict['lstm']['trues_orig'],
                   'speeds_orig':results_dict['lstm']['speeds_orig']},
        'stgcn':  {'preds_orig': results_dict['stgcn']['preds_orig'],
                   'trues_orig': results_dict['stgcn']['trues_orig'],
                   'speeds_orig':results_dict['stgcn']['speeds_orig']},
        'resstgcn': {'preds_orig': results_dict['resstgcn']['preds_orig'],
                    'trues_orig': results_dict['resstgcn']['trues_orig'],
                    'speeds_orig':results_dict['resstgcn']['speeds_orig']}
    },
    out_png=str(res_dir / 'speed_level_rmse_bar.png')
)
print("→ Done: speed_level_rmse_bar.png\n")

# 4.8 Error histogram + KDE
print("▶ Generating: Error histogram + KDE...")
plots.plot_error_histogram_kde(
    results_dict={
        'lstm':   {'preds_orig': results_dict['lstm']['preds_orig'],
                   'trues_orig': results_dict['lstm']['trues_orig']},
        'stgcn':  {'preds_orig': results_dict['stgcn']['preds_orig'],
                   'trues_orig': results_dict['stgcn']['trues_orig']},
        'resstgcn': {'preds_orig': results_dict['resstgcn']['preds_orig'],
                    'trues_orig': results_dict['resstgcn']['trues_orig']}
    },
    out_png=str(res_dir / 'error_histogram_kde.png'),
    bins=100
)
print("→ Done: error_histogram_kde.png\n")

# 4.9 Error ECDF
print("▶ Generating: Error ECDF...")
plots.plot_error_ecdf(
    results_dict={
        'lstm':   {'preds_orig': results_dict['lstm']['preds_orig'],
                   'trues_orig': results_dict['lstm']['trues_orig']},
        'stgcn':  {'preds_orig': results_dict['stgcn']['preds_orig'],
                   'trues_orig': results_dict['stgcn']['trues_orig']},
        'resstgcn': {'preds_orig': results_dict['resstgcn']['preds_orig'],
                    'trues_orig': results_dict['resstgcn']['trues_orig']}
    },
    out_png=str(res_dir / 'error_ecdf.png')
)
print("→ Done: error_ecdf.png\n")

# 4.10 True vs Pred scatter
print("▶ Generating: True vs Pred scatter...")
plots.plot_true_vs_pred_scatter(
    results_dict={
        'lstm':   {'preds_orig': results_dict['lstm']['preds_orig'],
                   'trues_orig': results_dict['lstm']['trues_orig']},
        'stgcn':  {'preds_orig': results_dict['stgcn']['preds_orig'],
                   'trues_orig': results_dict['stgcn']['trues_orig']},
        'resstgcn': {'preds_orig': results_dict['resstgcn']['preds_orig'],
                    'trues_orig': results_dict['resstgcn']['trues_orig']}
    },
    out_png=str(res_dir / 'true_vs_pred_scatter.png'),
    sample_fraction=0.01
)
print("→ Done: true_vs_pred_scatter.png\n")

# 4.11 Epoch vs Global metrics curves
print("▶ Generating: Epoch vs Global metrics curves...")
plots.plot_epoch_global_curve(
    epoch_list=results_dict['epoch_list'],
    metrics_dict=results_dict['metrics_epoch'],
    out_png=str(res_dir / 'epoch_vs_global_metrics.png')
)
print("→ Done: epoch_vs_global_metrics.png\n")

# 4.12 Epoch vs Node RMSE curves (예: 노드 42와 100번 노드)
node_list = cfg.get('node_list', [cfg.get('compare_node', 42)])
print("▶ Generating: Epoch vs Node RMSE curves (Nodes: 42, 100)...")
plots.plot_epoch_node_curve(
    epoch_list=results_dict['epoch_list'],
    node_rmse_dict=results_dict['node_epoch_rmse'],
    node_list=node_list,
    out_png=str(res_dir / 'epoch_vs_node_rmse.png')
)
print("→ Done: epoch_vs_node_rmse.png\n")

# 4.13 Diurnal ribbon plot (기존 함수)
print("▶ Generating: Diurnal ribbon plot...")
plots.plot_diurnal_ribbon(
    results_dict={
        'lstm':   {'preds_orig': results_dict['lstm']['preds_orig'],
                   'trues_orig': results_dict['lstm']['trues_orig'],
                   'slot_idx':   results_dict['lstm']['slot_idx']},
        'stgcn':  {'preds_orig': results_dict['stgcn']['preds_orig'],
                   'trues_orig': results_dict['stgcn']['trues_orig'],
                   'slot_idx':   results_dict['stgcn']['slot_idx']},
        'resstgcn': {'preds_orig': results_dict['resstgcn']['preds_orig'],
                     'trues_orig': results_dict['resstgcn']['trues_orig'],
                     'slot_idx':   results_dict['resstgcn']['slot_idx']}
    },
    out_png=str(res_dir / 'diurnal_ribbon.png')
)
print("→ Done: Diurnal ribbon plot\n")

# ─── 여기에 추가 ───────────────────────────────────────────────────────────
# 4.14 Diurnal (Day vs Weekend) 비교
print("▶ Generating: Diurnal (Day vs Weekend)...")
plots.plot_diurnal_weekday_vs_weekend(
    results_dict={
        'lstm':   {
            'preds_orig': results_dict['lstm']['preds_orig'],
            'trues_orig': results_dict['lstm']['trues_orig'],
            'slot_idx':   results_dict['lstm']['slot_idx'],
            'is_weekend': results_dict['lstm']['is_weekend']
        },
        'stgcn':  {
            'preds_orig': results_dict['stgcn']['preds_orig'],
            'trues_orig': results_dict['stgcn']['trues_orig'],
            'slot_idx':   results_dict['stgcn']['slot_idx'],
            'is_weekend': results_dict['stgcn']['is_weekend']
        },
        'resstgcn': {
            'preds_orig': results_dict['resstgcn']['preds_orig'],
            'trues_orig': results_dict['resstgcn']['trues_orig'],
            'slot_idx':   results_dict['resstgcn']['slot_idx'],
            'is_weekend': results_dict['resstgcn']['is_weekend']
        }
    },
    out_png=str(res_dir / 'diurnal_wd_vs_we.png')
)
print("→ Done: Diurnal (Day vs Weekend)\n")

# 4.15 날짜별 RMSE 추이 (Weekday vs Weekend)
print("▶ Generating: 날짜별 RMSE 추이 (Weekday vs Weekend)...")
plots.plot_daily_rmse_trend(
    results_dict={
        'lstm':   {
            'preds_orig': results_dict['lstm']['preds_orig'],
            'trues_orig': results_dict['lstm']['trues_orig'],
            'dates_sel':  results_dict['lstm']['dates_sel'],
            'is_weekend': results_dict['lstm']['is_weekend']
        },
        'stgcn':  {
            'preds_orig': results_dict['stgcn']['preds_orig'],
            'trues_orig': results_dict['stgcn']['trues_orig'],
            'dates_sel':  results_dict['stgcn']['dates_sel'],
            'is_weekend': results_dict['stgcn']['is_weekend']
        },
        'resstgcn': {
            'preds_orig': results_dict['resstgcn']['preds_orig'],
            'trues_orig': results_dict['resstgcn']['trues_orig'],
            'dates_sel':  results_dict['resstgcn']['dates_sel'],
            'is_weekend': results_dict['resstgcn']['is_weekend']
        }
    },
    out_png=str(res_dir / 'daily_rmse_trend_wd_vs_we.png')
)
print("→ Done: 날짜별 RMSE 추이 (Weekday vs Weekend)\n")
# ────────────────────────────────────────────────────────────────────────────────
print("✔️ All plots generated →", res_dir)