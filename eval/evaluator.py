# eval/evaluator.py

import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
import sys

from eval.loader import WindowLoader
from eval.metrics import calc_basic
from eval.infer import predict

def evaluate(cfg: dict):
    """
    cfg 예시 (eval_config.yaml):
      batch: 64
      tensor_dir: 3_tensor
      ckpt:
        lstm:      checkpoints_lstm/lstm_epoch040.pt
        stgcn:     checkpoints_stgcn/stgcn_epoch040.pt
        resstgcn:  checkpoints_resstgcn/resstgcn_epoch040.pt
      compare_node: 42
      # 추가로 epoch_list, ckpt_tpl_* 등을 포함할 수 있음
    """

    # ─── 0) Device 설정 ──────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Using device: {device}\n")

    # ─── 1) Data Loader 준비 ──────────────────────────────────────────────────────
    wloader = WindowLoader(cfg['tensor_dir'], cfg['batch'])
    total_windows = wloader.M
    batch_size = cfg['batch']
    total_batches = (total_windows + batch_size - 1) // batch_size
    print(f"▶ Loaded WindowLoader: total_windows={total_windows}, batch_size={batch_size}, total_batches={total_batches}\n")

    # ─── 2) 모델 로드 함수 정의 ────────────────────────────────────────────────────
    from model.lstm_model import BasicLSTM
    from model.stgcn_model import STGCN
    from model.res_stgcn_model import ResSTGCN

    def load_single(mtype: str):
        """
        mtype: 'lstm' | 'stgcn' | 'resstgcn'
        - 체크포인트를 읽어 모델을 초기화·로드·eval로 반환합니다.
        - STGCN / ResSTGCN의 경우, adjacency A를 GPU로 올린 후
          모델 내부 모든 서브모듈(module)에 동일한 cuda상의 A를 덮어씁니다.
        """
        ckpt_path = cfg['ckpt'][mtype]
        ckpt = torch.load(ckpt_path, map_location=device)

        if mtype == 'lstm':
            # BasicLSTM 모델 초기화 및 체크포인트 로드
            m = BasicLSTM(num_nodes=1370, input_dim=8, hidden_dim=64)
            m.load_state_dict(ckpt['model_state_dict'])
            m = m.to(device).eval()
            return m

        elif mtype == 'stgcn':
            # A 행렬 로드 (CPU)
            A_cpu = np.load(os.path.join(cfg['tensor_dir'], 'adjacency', 'A_lane.npy'))
            # GPU로 이동
            A_cuda = torch.from_numpy(A_cpu).float().to(device)

            # STGCN 초기화 (A는 CPU로 넣어두고, 이후 덮어쓰기)
            m = STGCN(in_channels=9, out_channels=8, num_nodes=1370, A=torch.from_numpy(A_cpu).float())
            m.load_state_dict(ckpt['model_state_dict'])
            m = m.to(device).eval()

            # 모델 내 모든 서브모듈에 A_cuda 덮어쓰기
            for submodule in m.modules():
                if hasattr(submodule, 'A'):
                    submodule.A = A_cuda

            return m

        else:  # 'resstgcn'
            # A 행렬 로드 (CPU)
            A_cpu = np.load(os.path.join(cfg['tensor_dir'], 'adjacency', 'A_lane.npy'))
            # GPU로 이동
            A_cuda = torch.from_numpy(A_cpu).float().to(device)

            # ResSTGCN 초기화 (hidden_dim=256로 학습 시와 일치)
            m = ResSTGCN(in_channels=9, out_channels=8, num_nodes=1370,
                         A=torch.from_numpy(A_cpu).float(), hidden_dim=256)
            m.stgcn.load_state_dict(ckpt['stgcn_state_dict'])
            m.reslstm.load_state_dict(ckpt['reslstm_state_dict'])
            m = m.to(device).eval()

            # 모델 내 모든 서브모듈에 A_cuda 덮어쓰기
            for submodule in m.modules():
                if hasattr(submodule, 'A'):
                    submodule.A = A_cuda

            return m

    # ─── 3) 모든 모델 로드 + A-device 확인 ─────────────────────────────────────────
    models = {}
    for key in ['lstm', 'stgcn', 'resstgcn']:
        print(f"▶ Loading checkpoint for {key.upper()}...")
        models[key] = load_single(key)
        found_A = False
        for subm in models[key].modules():
            if hasattr(subm, 'A'):
                # subm.A는 GPU로 올라간 torch.Tensor여야 함
                print(f"   → {key.upper()} A-device (submodule): {subm.A.device}\n")
                found_A = True
                break
        if not found_A:
        # LSTM의 경우에는 애초에 A 속성을 사용하지 않으므로 정상
            print(f"   → {key.upper()} has no attribute A (LSTM 모델)\n")

    # ─── 4) 추론(inference) 루프: 모든 윈도우에 대해 예측 수행 + 결과 수집 ──────────
    agg = {k: [] for k in models.keys()}
    t0 = time.time()
    print(f"▶ Starting inference on {total_windows} windows ({total_batches} batches)...\n")

    for Xb, Yb, idx_batch, date_batch in tqdm(
        wloader.batches(),
        desc="Inference batches",
        total=total_batches,
        file=sys.stdout
    ):
        # Xb: (B,12,1370,9), Yb: (B,1370,8)
        x_t = torch.from_numpy(np.transpose(Xb, (0, 3, 1, 2))).float()  # (B,9,12,1370)

        for mtype, m in models.items():
            weekend_flag = x_t[:, 8, 0, 0]  # (B,) 형태

            preds_norm = predict(m, x_t, mtype, device, weekend_flag=weekend_flag)  # (B,1370,8)

            # slot_idx 계산
            input_window_size = wloader.X.shape[1]  # 예: 12
            slot_indices = ((idx_batch % 288) + (input_window_size - 1)) % 288  # (B,)

            # is_weekend: YYYYMMDD → 0/1
            is_wd_list = []
            for d in date_batch:
                import datetime as _dt
                dt_obj = _dt.datetime.strptime(str(int(d)), "%Y%m%d")
                is_wd_list.append(1 if dt_obj.weekday() >= 5 else 0)
            is_wd = np.array(is_wd_list, dtype=np.int64)

            # speeds_norm: Yb[..., 4:8] (B,1370,4)
            speeds_norm = Yb[..., 4:8].copy()

            # agg에 저장
            agg[mtype].append({
                'preds_norm': preds_norm,       # (B,1370,8)
                'trues_norm': Yb.copy(),        # (B,1370,8)
                'date': date_batch.copy(),      # (B,)
                'slot_idx': slot_indices.copy(),# (B,)
                'is_weekend': is_wd.copy(),     # (B,)
                'speeds_norm': speeds_norm      # (B,1370,4)
            })

    elapsed = time.time() - t0
    print(f"▶ Inference done in {elapsed:.1f}s\n")

    # ─── 5) agg 결과를 모두 합치고, denormalize 수행 ─────────────────────────────
    results_dict = {'lstm': {}, 'stgcn': {}, 'resstgcn': {}}

    for mtype in ['lstm', 'stgcn', 'resstgcn']:
        print(f"▶ Concatenating and denormalizing for {mtype.upper()}...")
        preds_list   = [entry['preds_norm']   for entry in agg[mtype]]
        trues_list   = [entry['trues_norm']   for entry in agg[mtype]]
        dates_list   = [entry['date']         for entry in agg[mtype]]
        slots_list   = [entry['slot_idx']     for entry in agg[mtype]]
        weekend_list = [entry['is_weekend']   for entry in agg[mtype]]
        speeds_list  = [entry['speeds_norm']  for entry in agg[mtype]]

        preds_norm_all  = np.concatenate(preds_list,   axis=0)  # (M_sel,1370,8)
        trues_norm_all  = np.concatenate(trues_list,   axis=0)  # (M_sel,1370,8)
        dates_all       = np.concatenate(dates_list,   axis=0)  # (M_sel,)
        slots_all       = np.concatenate(slots_list,   axis=0)  # (M_sel,)
        weekend_all     = np.concatenate(weekend_list, axis=0)  # (M_sel,)
        speeds_norm_all = np.concatenate(speeds_list,  axis=0)  # (M_sel,1370,4)

        # denormalize preds, trues
        preds_orig = wloader.denorm(preds_norm_all, dates_all)  # (M_sel,1370,8)
        trues_orig = wloader.denorm(trues_norm_all, dates_all)  # (M_sel,1370,8)

        # denormalize speeds_norm_all → 속도 채널 4개 (M_sel,1370,4)
        speeds_orig = np.zeros_like(speeds_norm_all, dtype=np.float32)
        for unique_date in np.unique(dates_all):
            mask = (dates_all == unique_date)
            mu, std = wloader.stats[int(unique_date)]
            speeds_orig[mask] = speeds_norm_all[mask] * std[4:8].reshape(1,1,4) + mu[4:8].reshape(1,1,4)

        print(f"   → {mtype.upper()} concatenation & denorm complete (shape={preds_orig.shape})\n")

        results_dict[mtype]['preds_orig']  = preds_orig
        results_dict[mtype]['trues_orig']  = trues_orig
        results_dict[mtype]['dates_sel']   = dates_all.astype(int)
        results_dict[mtype]['slot_idx']    = slots_all.astype(int)
        results_dict[mtype]['is_weekend']  = weekend_all.astype(int)
        results_dict[mtype]['speeds_orig'] = speeds_orig

    # ─── 6) CSV 저장 (Global + Channel-wise Metrics) ──────────────────────────
    save_dir = cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    print(f"▶ Saving global metrics and channel metrics to \"{save_dir}\"...\n")
    global_rows = []
    for mtype in ['lstm', 'stgcn', 'resstgcn']:
        pred_o = results_dict[mtype]['preds_orig']
        true_o = results_dict[mtype]['trues_orig']
        met = calc_basic(pred_o, true_o)
        met['model'] = mtype
        global_rows.append(met)
    pd.DataFrame(global_rows).to_csv(os.path.join(save_dir, 'metrics_global.csv'), index=False)

    for mtype in ['lstm', 'stgcn', 'resstgcn']:
        ch_rows = []
        pred_o = results_dict[mtype]['preds_orig']
        true_o = results_dict[mtype]['trues_orig']
        for ch in range(8):
            m_ch = calc_basic(pred_o[:, :, ch], true_o[:, :, ch])
            m_ch.update(model=mtype, channel=f'ch{ch}')
            ch_rows.append(m_ch)
        pd.DataFrame(ch_rows).to_csv(os.path.join(save_dir, f'metrics_channel_{mtype}.csv'), index=False)

    # ─── 7) Epoch 지표 수집 (옵션) ───────────────────────────────────────────────
    results_dict['epoch_list']      = []
    results_dict['metrics_epoch']   = {'lstm': [], 'stgcn': [], 'resstgcn': []}
    results_dict['node_epoch_rmse'] = {'lstm': {}, 'stgcn': {}, 'resstgcn': {}}

    epoch_list = cfg.get('epoch_list', [])
    if epoch_list and cfg.get('ckpt_tpl_lstm') and cfg.get('ckpt_tpl_stgcn') and cfg.get('ckpt_tpl_resstgcn'):
        print(f"▶ Starting epoch-wise evaluation for epochs: {epoch_list}\n")

        def load_ckpt(path_tmpl, epoch):
            p = path_tmpl.format(epoch=epoch)
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            return torch.load(p, map_location=device)

        cn = int(cfg.get('compare_node', 42))

        for ep in epoch_list:
            print(f"   ▶ Epoch {ep}: evaluating each model...")
            # (1) LSTM
            ckpt_l = load_ckpt(cfg['ckpt_tpl_lstm'], ep)
            lstm = BasicLSTM(num_nodes=1370, input_dim=8, hidden_dim=64).to(device)
            lstm.load_state_dict(ckpt_l['model_state_dict'])
            lstm.eval()

            preds_norm_ep = []
            trues_norm_ep = []
            with torch.no_grad():
                for Xb, Yb, idx_b, date_b in wloader.batches():
                    x_t = torch.from_numpy(np.transpose(Xb, (0,3,1,2))).float()
                    p_ep = predict(lstm, x_t, 'lstm', device, weekend_flag=x_t[:,8,0,0])
                    preds_norm_ep.append(p_ep)
                    trues_norm_ep.append(Yb)
            preds_norm_ep = np.concatenate(preds_norm_ep, axis=0)
            trues_norm_ep = np.concatenate(trues_norm_ep, axis=0)

            preds_o_ep = wloader.denorm(preds_norm_ep, wloader.D)
            trues_o_ep = wloader.denorm(trues_norm_ep, wloader.D)
            met_ep = calc_basic(preds_o_ep, trues_o_ep)
            results_dict['metrics_epoch']['lstm'].append(met_ep['RMSE'])
            node_rmse = np.sqrt(np.mean((preds_o_ep[:, cn, :] - trues_o_ep[:, cn, :])**2))
            results_dict['node_epoch_rmse']['lstm'].setdefault(cn, []).append(node_rmse)
            print(f"      → LSTM Epoch {ep} done. RMSE={met_ep['RMSE']:.3f}, Node{cn} RMSE={node_rmse:.3f}")
            torch.cuda.empty_cache()

            # (2) ST-GCN
            ckpt_s = load_ckpt(cfg['ckpt_tpl_stgcn'], ep)
            A_cpu = np.load(os.path.join(cfg['tensor_dir'], 'adjacency', 'A_lane.npy'))
            A_cuda = torch.from_numpy(A_cpu).float().to(device)
            stgcn = STGCN(in_channels=9, out_channels=8, num_nodes=1370, A=torch.from_numpy(A_cpu).float()).to(device)
            stgcn.load_state_dict(ckpt_s['model_state_dict'])
            stgcn.eval()
            for subm in stgcn.modules():
                if hasattr(subm, 'A'):
                    subm.A = A_cuda

            preds_norm_ep = []
            trues_norm_ep = []
            with torch.no_grad():
                for Xb, Yb, idx_b, date_b in wloader.batches():
                    x_t = torch.from_numpy(np.transpose(Xb, (0,3,1,2))).float()
                    p_ep = predict(stgcn, x_t, 'stgcn', device, weekend_flag=x_t[:,8,0,0])
                    preds_norm_ep.append(p_ep)
                    trues_norm_ep.append(Yb)
            preds_norm_ep = np.concatenate(preds_norm_ep, axis=0)
            trues_norm_ep = np.concatenate(trues_norm_ep, axis=0)

            preds_o_ep = wloader.denorm(preds_norm_ep, wloader.D)
            trues_o_ep = wloader.denorm(trues_norm_ep, wloader.D)
            met_ep = calc_basic(preds_o_ep, trues_o_ep)
            results_dict['metrics_epoch']['stgcn'].append(met_ep['RMSE'])
            node_rmse = np.sqrt(np.mean((preds_o_ep[:, cn, :] - trues_o_ep[:, cn, :])**2))
            results_dict['node_epoch_rmse']['stgcn'].setdefault(cn, []).append(node_rmse)
            print(f"      → STGCN Epoch {ep} done. RMSE={met_ep['RMSE']:.3f}, Node{cn} RMSE={node_rmse:.3f}")
            torch.cuda.empty_cache()

            # (3) ResSTGCN
            ckpt_r = load_ckpt(cfg['ckpt_tpl_resstgcn'], ep)
            resm = ResSTGCN(in_channels=9, out_channels=8, num_nodes=1370,
                            A=torch.from_numpy(A_cpu).float(), hidden_dim=256).to(device)
            resm.stgcn.load_state_dict(ckpt_r['stgcn_state_dict'])
            resm.reslstm.load_state_dict(ckpt_r['reslstm_state_dict'])
            resm.eval()
            for subm in resm.modules():
                if hasattr(subm, 'A'):
                    subm.A = A_cuda

            preds_norm_ep = []
            trues_norm_ep = []
            with torch.no_grad():
                for Xb, Yb, idx_b, date_b in wloader.batches():
                    x_t = torch.from_numpy(np.transpose(Xb, (0,3,1,2))).float()
                    p_ep = predict(resm, x_t, 'resstgcn', device, weekend_flag=x_t[:,8,0,0])
                    preds_norm_ep.append(p_ep)
                    trues_norm_ep.append(Yb)
            preds_norm_ep = np.concatenate(preds_norm_ep, axis=0)
            trues_norm_ep = np.concatenate(trues_norm_ep, axis=0)

            preds_o_ep = wloader.denorm(preds_norm_ep, wloader.D)
            trues_o_ep = wloader.denorm(trues_norm_ep, wloader.D)
            met_ep = calc_basic(preds_o_ep, trues_o_ep)
            results_dict['metrics_epoch']['resstgcn'].append(met_ep['RMSE'])
            node_rmse = np.sqrt(np.mean((preds_o_ep[:, cn, :] - trues_o_ep[:, cn, :])**2))
            results_dict['node_epoch_rmse']['resstgcn'].setdefault(cn, []).append(node_rmse)
            print(f"      → ResSTGCN Epoch {ep} done. RMSE={met_ep['RMSE']:.3f}, Node{cn} RMSE={node_rmse:.3f}\n")
            torch.cuda.empty_cache()

        results_dict['epoch_list'] = epoch_list
        print(f"▶ Epoch-wise evaluation complete.\n")

    # ─── 8) results_dict.pkl 저장 및 반환 ───────────────────────────────────────
    print(f"▶ Saving results_dict.pkl to {save_dir}...")
    with open(os.path.join(save_dir, 'results_dict.pkl'), 'wb') as f:
        pickle.dump(results_dict, f)

    print(f"▶ CSVs & results_dict.pkl saved to {save_dir}")
    return Path(save_dir)