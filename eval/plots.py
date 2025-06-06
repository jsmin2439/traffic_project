# eval/plots.py

"""
plots.py

이 파일은 세 가지 모델(LSTM, ST-GCN, ResSTGCN)에 대한 다양한 비교 지표와 시각화를 생성하는 함수들을 모아둔 모듈입니다.
각 함수는 CSV 파일 혹은 DataFrame을 입력받아 논문에 포함할 만한 시각화를 그리고, PNG 파일로 저장합니다.

각 함수의 역할:
    - plot_global_bar: 모델별 Global 지표(예: RMSE, MAE, MAPE, R2)를 막대그래프로 시각화
    - plot_channel_radar: 채널(8개 차종)별 RMSE를 Radar Chart로 시각화
    - plot_node_channel_heatmap: 노드×채널 평균 RMSE를 heatmap으로 시각화
    - plot_window_rmse_trend: 전체 윈도우 인덱스에 따른 RMSE 변화 추이를 라인 플롯으로 시각화
    - plot_diurnal_ribbon: 하루(288슬롯) 기준 평균 오차 ± 표준편차 Ribbon Plot
    - plot_weekday_vs_weekend_box: 주중 vs 주말 RMSE 분포를 Boxplot으로 시각화 (p-value Annot 추가 가능)
    - plot_speed_level_bar: 저속/중간/고속 구간별 RMSE를 Grouped Bar Chart로 시각화
    - plot_error_histogram_kde: 모델별 오차 히스토그램 및 KDE를 한 그래프에 중첩
    - plot_error_ecdf: 모델별 Error ECDF(누적분포함수) 플롯
    - plot_true_vs_pred_scatter: 실제값 vs 예측값 산점도로 시각화 (1:1 reference 라인 포함)
    - plot_epoch_global_curve: Epoch vs Global RMSE (및 기타 지표) 변화 라인 플롯
    - plot_epoch_node_curve: 특정 노드(또는 노드 집합)에 대한 Epoch별 RMSE 변화 라인 플롯

사용 예:
    import plots
    plots.plot_global_bar("results/metrics_global.csv", "results/global_bar.png")
    plots.plot_channel_radar("results/", "results/radar_channel.png")
    ...
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 한글 폰트 설정 (필요시)
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] = False

def plot_global_bar(csv_path, out_png):
    """
    모델별 Global 지표(RMSE, MAE, MAPE, R2)를 막대그래프로 시각화합니다.
    
    Args:
        csv_path (str or Path): 'metrics_global.csv' 파일 경로. 열: ['model','RMSE','MAE','MAPE','R2']
        out_png (str or Path): 저장할 PNG 파일 경로
    """
    # CSV 로드
    df = pd.read_csv(csv_path)
    
    # 시각화를 위한 설정
    metrics = ['RMSE', 'MAE', 'MAPE', 'R2']   # 시각화할 지표 목록
    n_metrics = len(metrics)
    
    # Figure, Axes 생성: 하나의 row에 n_metrics개의 subplot
    fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 4))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        # seaborn barplot 사용: x축=model, y축=metric
        sns.barplot(data=df, x='model', y=metric, ax=ax, palette='Set2')
        ax.set_title(metric)
        ax.set_xlabel('')
        ax.set_ylabel(metric)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_channel_radar(csv_dir, out_png):
    """
    채널(8개 차종)별 RMSE를 Radar Chart(Spider Chart)로 시각화합니다.
    각 모델별로 하나의 폴리곤을 그려서 비교합니다.
    
    Args:
        csv_dir (str or Path): 'metrics_channel_<model>.csv' 파일들이 위치한 디렉터리
                              파일 패턴: metrics_channel_lstm.csv, metrics_channel_stgcn.csv, metrics_channel_resstgcn.csv
        out_png (str or Path): 저장할 PNG 파일 경로
    """
    # 모델명과 채널 레이블 정의
    models = ['lstm', 'stgcn', 'resstgcn']
    channels = [f'ch{i}' for i in range(8)]   # ch0 ~ ch7
    
    # 각 모델별 RMSE 값을 리스트에 수집
    data = {}
    for m in models:
        # 각 모델별 metrics_channel CSV 로드
        path = os.path.join(csv_dir, f'metrics_channel_{m}.csv')
        df = pd.read_csv(path)
        # 각 채널별 RMSE 추출
        rmse_vals = df['RMSE'].values.tolist()  
        data[m] = rmse_vals
    
    # 각 채널 레이블에 첫 값을 다시 추가하여 radar 차트 폐쇄(완전 순환)    
    angles = np.linspace(0, 2 * np.pi, len(channels), endpoint=False).tolist()
    angles += angles[:1]  # 마지막 값을 첫 값으로 반복
    
    # Figure 생성: polar 축
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    
    for m in models:
        vals = data[m]
        vals_cycle = vals + vals[:1]  # 닫힌 형태로 한 칸 추가
        ax.plot(angles, vals_cycle, label=m.upper(), linewidth=2)
        ax.fill(angles, vals_cycle, alpha=0.1)
    
    # X축 각도에 채널 레이블 매핑
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(channels)
    
    # Y축 레이블 설정 (원 거리)
    ax.set_rlabel_position(0)
    ax.set_title("Channel-wise RMSE (Radar Chart)", y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_node_channel_heatmap(results_dict, out_png, normalize: bool = False):
    """
    노드×채널 평균 RMSE를 heatmap으로 시각화합니다.
    - results_dict: {'lstm': np.ndarray(M,1370,8), 'stgcn': ..., 'resstgcn': ...}
      각 값은 denormalized 예측값과 실제값의 차이를 통해 RMSE를 계산해야 함.
    - normalize: True인 경우 각 RMSE 값을 해당 채널의 최대 RMSE로 나누어 정규화
    
    Args:
        results_dict (dict): 
            각 모델의 예측(preds_orig)과 실제(trues_orig)가 포함된 딕셔너리. 
            예) {'lstm': {'preds': (M,1370,8), 'trues': (M,1370,8)}, ...}
        out_png (str or Path): 저장할 PNG 파일 경로
        normalize (bool): 채널별 RMSE 정규화 여부
    """
    models = ['lstm', 'stgcn', 'resstgcn']
    num_nodes = 1370
    num_channels = 8
    
    # 노드×채널 RMSE 매트릭스 딕셔너리 저장
    rmse_matrices = {}
    
    for m in models:
        preds = results_dict[m]['preds']  # shape: (M_sel, 1370, 8)
        trues = results_dict[m]['trues']
        
        # M_sel 윈도우와 채널별 T 차원 평균 오차 계산: (1370,8)
        # step1: 차이 계산
        diff = preds - trues                          # (M_sel,1370,8)
        mse_per_node_channel = np.mean(diff ** 2, axis=0)  # (1370,8)
        rmse_per_node_channel = np.sqrt(mse_per_node_channel)  # (1370,8)
        
        if normalize:
            # 채널별 최대 RMSE로 나누기
            max_per_channel = rmse_per_node_channel.max(axis=0)  # (8,)
            rmse_per_node_channel = rmse_per_node_channel / (max_per_channel + 1e-6)
        
        rmse_matrices[m] = rmse_per_node_channel  # (1370,8)
    
    # Figure: 모델별 subplot (세로 3행)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True)
    for idx, m in enumerate(models):
        ax = axes[idx]
        data = rmse_matrices[m].T  # (8,1370) -> 채널이 y축, 노드가 x축
        im = ax.imshow(data, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_ylabel(m.upper(), rotation=0, labelpad=40, va='center')
        if idx == 2:
            ax.set_xlabel('Node Index')
        ax.set_yticks(np.arange(num_channels))
        ax.set_yticklabels([f'ch{i}' for i in range(num_channels)])
    # 컬러바 추가
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, orientation='vertical')
    cbar.set_label('RMSE' + (' (Normalized)' if normalize else ''))
    fig.suptitle('Node × Channel RMSE Heatmap', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_window_rmse_trend(results_dict, out_png, window_indices=None):
    """
    전체 윈도우 인덱스별 RMSE 변화 추이를 라인 플롯으로 시각화합니다.
    
    Args:
        results_dict (dict): {'lstm': {'preds':..., 'trues':...}, ...}
        out_png (str or Path): 저장할 PNG 파일 경로
        window_indices (list of tuples): 시각화할 윈도우 구간 리스트. 
                                         예: [(0,100), (101,200), (201,300)]. 
                                         None인 경우 전체 윈도우를 한번에 그립니다.
    """
    models = ['lstm', 'stgcn', 'resstgcn']
    # 각 모델별 윈도우 RMSE 계산: (M_sel,)
    window_rmse = {}
    for m in models:
        preds = results_dict[m]['preds']  # (M_sel,1370,8)
        trues = results_dict[m]['trues']
        diff = preds - trues
        # 윈도우별 RMSE: 각 윈도우의 (1370×8) 평균 제곱근
        rmse_per_window = np.sqrt(np.mean(diff**2, axis=(1,2)))  # (M_sel,)
        window_rmse[m] = rmse_per_window
    
    # 전체 윈도우 인덱스
    M_sel = window_rmse['lstm'].shape[0]
    full_indices = np.arange(M_sel)
    
    # 시각화: 하나의 figure에 subplot 또는 단일 plot
    if window_indices is None:
        # 전체를 한 그래프에
        plt.figure(figsize=(10, 4))
        for m in models:
            plt.plot(full_indices, window_rmse[m], label=m.upper(), linewidth=1.5)
        plt.xlabel('Window Index')
        plt.ylabel('RMSE per Window')
        plt.title('Window-wise RMSE Trend (All Windows)')
        plt.legend()
        plt.grid(linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()
    else:
        # 각 구간별 subplot 생성
        n_ranges = len(window_indices)
        fig, axes = plt.subplots(n_ranges, 1, figsize=(10, 4 * n_ranges))
        if n_ranges == 1:
            axes = [axes]
        for idx, (start, end) in enumerate(window_indices):
            ax = axes[idx]
            end = min(end, M_sel - 1)
            x = np.arange(start, end + 1)
            for m in models:
                y = window_rmse[m][start:end + 1]
                ax.plot(x, y, label=m.upper(), linewidth=1.5)
            ax.set_xlabel('Window Index')
            ax.set_ylabel('RMSE')
            ax.set_title(f'Window-wise RMSE ({start} to {end})')
            ax.legend(fontsize=8)
            ax.grid(linestyle='--', alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_png, dpi=300)
        plt.close(fig)


def plot_diurnal_ribbon(results_dict, out_png):
    """
    하루(288 슬롯) 기준 ‘평균 오차 ± 표준편차’ Ribbon Plot을 그려서 
    모델별로 Rush-hour 대 예측 성능 변화를 시각화합니다.
    
    Args:
        results_dict (dict): {'lstm': {'preds':..., 'trues':...}, ...}
        out_png (str or Path): 저장할 PNG 파일 경로
    """
    models = ['lstm', 'stgcn', 'resstgcn']
    num_slots = 288  # 하루 5분 단위
    
    # 슬롯별 평균 오차와 표준편차 계산: 
    # X: 슬롯 인덱스 0~287, Y: 모든 날짜·모든 노드·모든 채널 집합
    diurnal_stats = {}
    for m in models:
        preds = results_dict[m]['preds_orig']  # (M_sel,1370,8), denorm 상태
        trues = results_dict[m]['trues_orig']
        # 모든 윈도우별 slot index를 저장하기 위해 all_DATE, all_IDX 등을 사용해야 합니다.
        # 여기서는 결과 배열이 “윈도우 단위”이므로, 각 윈도우가 하루 중 어느 슬롯인지 알 수 있어야 합니다.
        # 일반적으로 data_loader에서 all_DATE.npy, all_IDX(윈도우 시작 슬롯)를 활용합니다.
        # 가정: results_dict[m]['slot_idx'] = array of shape (M_sel,) each ∈ [0,287]
        slot_idx = results_dict[m]['slot_idx']  # (M_sel,)
        
        # 슬롯별 diff 축적: 리스트 형태로
        slot_diffs = {i: [] for i in range(num_slots)}
        M_sel = preds.shape[0]
        for i in range(M_sel):
            s = slot_idx[i]
            # i번째 윈도우 예측과 실제의 차이를 1370×8 flatten 후 추가
            diff_i = (preds[i] - trues[i]).flatten()  # (1370*8,)
            slot_diffs[s].extend(diff_i.tolist())
        
        # 슬롯별 평균 및 표준편차 계산
        means = np.zeros(num_slots)
        stds  = np.zeros(num_slots)
        for i in range(num_slots):
            arr = np.array(slot_diffs[i])
            means[i] = np.mean(np.abs(arr))       # 절대 오차 기반 평균
            stds[i]  = np.std(np.abs(arr))
        diurnal_stats[m] = (means, stds)
    
    # Ribbon Plot
    plt.figure(figsize=(12, 4))
    for m in models:
        means, stds = diurnal_stats[m]
        x = np.arange(num_slots)
        plt.plot(x, means, label=m.upper(), linewidth=1.5)
        plt.fill_between(x, means - stds, means + stds, alpha=0.2)
    
    plt.xlabel('Slot Index (0~287)')
    plt.ylabel('Mean Absolute Error (± Std)')
    plt.title('Diurnal (Daily) Error Profile: Mean ± Std by Slot')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_weekday_vs_weekend_box(results_dict, out_png, pvalue_annot=True):
    """
    주중 vs 주말 RMSE 분포를 Boxplot으로 시각화하고, Mann-Whitney U Test p-value를 표기합니다.
    
    Args:
        results_dict (dict): {
            'lstm': {'preds':(M,1370,8),'trues':(M,1370,8),'dates':(M,), 'is_weekend':(M,)},
            'stgcn': {...}, 
            'resstgcn': {...}
        }
        out_png (str or Path): 저장할 PNG 파일 경로
        pvalue_annot (bool): Boxplot 위에 p-value 어노테이션을 추가할지 여부
    """
    models = ['lstm', 'stgcn', 'resstgcn']
    
    # Boxplot용 DataFrame 생성: columns = ['model', 'period', 'rmse']
    rows = []
    for m in models:
        preds = results_dict[m]['preds_orig']  # (M_sel,1370,8)
        trues = results_dict[m]['trues_orig']
        dates = results_dict[m]['dates_sel']   # (M_sel,)
        is_weekend = results_dict[m]['is_weekend']  # (M_sel,)
        
        # 윈도우별 RMSE 계산 (모든 채널·노드 평균)
        diff = preds - trues
        rmse_per_window = np.sqrt(np.mean(diff ** 2, axis=(1,2)))  # (M_sel,)
        
        # 주중/주말 구분하여 rows 추가
        for i in range(len(rmse_per_window)):
            period = 'Weekend' if is_weekend[i] else 'Weekday'
            rows.append({'model': m.upper(), 'period': period, 'rmse': rmse_per_window[i]})
    
    df_box = pd.DataFrame(rows)
    
    # Boxplot 그리기
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_box, x='model', y='rmse', hue='period', palette='Set2')
    plt.title('Weekday vs Weekend RMSE Distribution by Model')
    plt.xlabel('Model')
    plt.ylabel('RMSE per Window')
    plt.legend(title='Period')
    
    if pvalue_annot:
        # 모델별 주중 vs 주말 p-value 계산 및 어노테이션
        xpos = np.arange(len(models))
        for idx, m in enumerate(models):
            # 모델 대문자
            mod = m.upper()
            rmse_wd = df_box[(df_box['model'] == mod) & (df_box['period'] == 'Weekday')]['rmse']
            rmse_we = df_box[(df_box['model'] == mod) & (df_box['period'] == 'Weekend')]['rmse']
            # Mann-Whitney U Test (비정규 분포 가정)
            stat, p = stats.mannwhitneyu(rmse_wd, rmse_we, alternative='two-sided')
            # y축 최대값에 살짝 위에 어노테이션
            y_max = max(rmse_wd.max(), rmse_we.max())
            plt.text(xpos[idx], y_max * 1.05, f"p={p:.3e}", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_speed_level_bar(results_dict, out_png):
    """
    저속(<20), 중간(20~40), 고속(>=40) 구간별 RMSE를 Grouped Bar Chart로 시각화합니다.
    실제 속도값을 기준으로 윈도우별 평균 속도를 계산해야 합니다.
    
    Args:
        results_dict (dict): {
            'lstm': {'preds':..., 'trues':..., 'speeds':(M_sel,1370,4), 'dates':(M_sel,)},
            'stgcn': {...}, 'resstgcn': {...}
        }
        - 'speeds': 각 윈도우별 1370개 노드의 4개 speed 채널(정규화 해제 전 혹은 후) 중 평균 속도를 
                    하나의 스칼라로 요약해야 함. 
                    예시: speeds[i].mean()을 통해 윈도우 i의 전체 평균 속도를 구함.
        out_png (str or Path): 저장할 PNG 파일 경로
    """
    models = ['lstm', 'stgcn', 'resstgcn']
    speed_bins = {'Low (<20)': (None, 20), 'Mid (20~40)': (20, 40), 'High (>=40)': (40, None)}
    
    # DataFrame 생성: ['model','speed_level','rmse']
    rows = []
    for m in models:
        preds = results_dict[m]['preds_orig']  # (M_sel,1370,8)
        trues = results_dict[m]['trues_orig']
        speeds = results_dict[m]['speeds_orig']  # (M_sel,1370,4) denorm이거나 raw 속도
        # 윈도우별 전체 평균 속도 계산 (1370개 노드, 4채널 모두 평균)
        avg_speeds = np.mean(speeds, axis=(1,2))  # (M_sel,)
        
        # 윈도우별 RMSE 계산
        diff = preds - trues
        rmse_per_window = np.sqrt(np.mean(diff**2, axis=(1,2)))  # (M_sel,)
        
        for i in range(len(rmse_per_window)):
            sp = avg_speeds[i]
            lvl = None
            # 속도 Bin 구분
            if sp < 20:
                lvl = 'Low (<20)'
            elif sp < 40:
                lvl = 'Mid (20~40)'
            else:
                lvl = 'High (>=40)'
            rows.append({'model': m.upper(), 'speed_level': lvl, 'rmse': rmse_per_window[i]})
    
    df_sp = pd.DataFrame(rows)
    
    # Grouped Bar Chart: x축=speed_level, hue=model, y축=평균 RMSE
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_sp, x='speed_level', y='rmse', hue='model', palette='Set2', ci='sd')
    plt.title('Speed Level-wise RMSE by Model')
    plt.xlabel('Speed Level')
    plt.ylabel('RMSE per Window')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_error_histogram_kde(results_dict, out_png, bins=100):
    """
    모델별 에러(절대오차 또는 원시 오차) 히스토그램과 KDE를 한 그래프에 중첩하여 시각화합니다.
    
    Args:
        results_dict (dict): {'lstm': {'preds_orig':..., 'trues_orig':...}, ...}
        out_png (str or Path): 저장할 PNG 파일 경로
        bins (int): 히스토그램 bin 개수
    """
    models = ['lstm', 'stgcn', 'resstgcn']
    
    plt.figure(figsize=(8, 6))
    for m in models:
        preds = results_dict[m]['preds_orig']  # (M_sel,1370,8)
        trues = results_dict[m]['trues_orig']
        # 에러(절대값) flatten
        errors = (preds - trues).flatten()  # (M_sel*1370*8,)
        abs_errors = np.abs(errors)
        # 히스토그램
        sns.histplot(abs_errors, bins=bins, stat='density', alpha=0.3, label=m.upper())
        # KDE 곡선
        sns.kdeplot(abs_errors, bw_adjust=1, label=f"{m.upper()} KDE")
    
    plt.xlabel('Absolute Error')
    plt.ylabel('Density')
    plt.title('Error Histogram & KDE Comparison')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_error_ecdf(results_dict, out_png):
    """
    Error ECDF(누적분포함수)를 그려 모델별로 “절대 오차 ≤ x” 비율을 비교합니다.
    
    Args:
        results_dict (dict): {'lstm':{'preds_orig':...,'trues_orig':...}, ...}
        out_png (str or Path): 저장할 PNG 파일 경로
    """
    models = ['lstm', 'stgcn', 'resstgcn']
    
    plt.figure(figsize=(8, 6))
    for m in models:
        preds = results_dict[m]['preds_orig']
        trues = results_dict[m]['trues_orig']
        errors = np.abs((preds - trues).flatten())
        # ECDF 계산
        sorted_err = np.sort(errors)
        y = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        plt.step(sorted_err, y, where='post', label=m.upper())
    
    plt.xlabel('Absolute Error')
    plt.ylabel('ECDF')
    plt.title('Error ECDF Comparison')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_true_vs_pred_scatter(results_dict, out_png, sample_fraction=0.01):
    """
    실제값(True) vs 예측값(Pred) 산점도를 그려 모델별 편향 및 분산을 시각적으로 비교합니다.
    전체 점을 그리면 너무 많으므로, 일부 표본 추출하여 scatter plot으로 표현합니다.
    
    Args:
        results_dict (dict): {'lstm': {'preds_orig':(M,1370,8), 'trues_orig':(M,1370,8)}, ...}
        out_png (str or Path): 저장할 PNG 파일 경로
        sample_fraction (float): 전체 데이터 중에서 scatter plot용 샘플 비율 (예: 0.01 = 1%)
    """
    models = ['lstm', 'stgcn', 'resstgcn']
    plt.figure(figsize=(6, 6))
    
    for m in models:
        preds = results_dict[m]['preds_orig'].reshape(-1)  # 벡터화
        trues = results_dict[m]['trues_orig'].reshape(-1)
        # 샘플링: 전체 중 sample_fraction만 무작위 선택
        n_total = len(preds)
        n_sample = max(1000, int(n_total * sample_fraction))
        idxs = np.random.choice(n_total, size=n_sample, replace=False)
        plt.scatter(trues[idxs], preds[idxs], s=5, alpha=0.3, label=m.upper())
    
    # 1:1 reference 선
    min_val = min(plt.xlim()[0], plt.ylim()[0])
    max_val = max(plt.xlim()[1], plt.ylim()[1])
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
    
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title('True vs Pred Scatter (Sampled)')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_epoch_global_curve(epoch_list, metrics_dict, out_png):
    """
    Epoch vs Global RMSE (및 기타 지표) 변화 곡선을 그립니다.
    
    Args:
        epoch_list (list of int): [5, 10, 15, ..., 40]
        metrics_dict (dict): {
            'lstm': {'RMSE': [...], 'MAE': [...], 'MAPE': [...], 'R2': [...]},
            'stgcn': {...}, 'resstgcn': {...}
        }
        out_png (str or Path): 저장할 PNG 파일 경로
    """
    models = ['lstm', 'stgcn', 'resstgcn']
    metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
    
    # Figure와 Axes 생성: subplot 개수 = 메트릭 개수
    fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 4))
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for m in models:
            y = metrics_dict[m][metric]  # 리스트 길이 = len(epoch_list)
            ax.plot(epoch_list, y, marker='o', label=m.upper())
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(f'Epoch vs {metric}')
        ax.grid(linestyle='--', alpha=0.4)
        if idx == 0:
            ax.legend(loc='best', fontsize=8)
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_epoch_node_curve(epoch_list, node_rmse_dict, out_png, node_list):
    """
    특정 노드 혹은 노드 그룹에 대해 Epoch별 RMSE 변화를 비교하는 라인 플롯을 그립니다.
    
    Args:
        epoch_list (list of int): [5, 10, 15, ..., 40]
        node_rmse_dict (dict): {
            'lstm': {node_idx: [rmse_ep5, rmse_ep10, ...], ...}, 
            'stgcn': {...}, 'resstgcn': {...}
        }
        out_png (str or Path): 저장할 PNG 파일 경로
        node_list (list of int): 시각화할 노드 인덱스 목록 (예: [42, 100, 500])
    """
    models = ['lstm', 'stgcn', 'resstgcn']
    n_nodes = len(node_list)
    
    # Figure: subplot 개수 = 노드 개수
    fig, axes = plt.subplots(n_nodes, 1, figsize=(6, 3*n_nodes), sharex=True)
    if n_nodes == 1:
        axes = [axes]
    
    for idx, node in enumerate(node_list):
        ax = axes[idx]
        for m in models:
            rmse_vals = node_rmse_dict[m][node]  # [rmse_ep5, rmse_ep10, ...]
            ax.plot(epoch_list, rmse_vals, marker='o', label=m.upper())
        ax.set_ylabel(f'Node {node} RMSE')
        ax.set_title(f'Node {node}: Epoch-wise RMSE')
        ax.grid(linestyle='--', alpha=0.4)
        if idx == 0:
            ax.legend(loc='best', fontsize=8)
    
    axes[-1].set_xlabel('Epoch')
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)