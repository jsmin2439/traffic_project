#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/evaluator.py

세 가지 모델(LSTM-only, ST-GCN, Gated-Fusion) 평가 스크립트
- Gated-Fusion 모델 지원 추가
- cfg에 'gated' 체크포인트 및 하이퍼파라미터(hidden1, hidden2, use_tcn) 필수
"""
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.jit
from pathlib import Path
from tqdm import tqdm

# “eval” 폴더를 파이썬 모듈 검색 경로에 추가
sys.path.append(str(Path(__file__).resolve().parent))

from eval.loader import WindowLoader
from eval.metrics import calc_basic
from eval.infer import predict

# 모델 클래스 임포트 (Gated-Fusion 포함)
from model.lstm_model import BasicLSTM
from model.stgcn_model import STGCN
from model.gated_fusion_stgcn import GatedFusionSTGCN

def evaluate(cfg: dict):
    """
cfg 예시 (eval_config.yaml):
  batch: 64
  tensor_dir: 3_tensor
  ckpt:
    lstm:   checkpoints_lstm/lstm_epoch040.pt
    stgcn:  checkpoints_stgcn/stgcn_epoch040.pt
    gated:  checkpoints_gated/gated_epoch040.pt
  hidden1: 64        # Gated 모델 hidden1
  hidden2: 128       # Gated 모델 hidden2
  use_tcn: false     # Gated-Fusion TCN 여부
  compare_node: 42
  save_dir: eval_results
  epoch_list: [10,20,30]
  ckpt_tpl_lstm:      checkpoints_lstm/lstm_epoch{epoch:03d}.pt
  ckpt_tpl_stgcn:     checkpoints_stgcn/stgcn_epoch{epoch:03d}.pt
  ckpt_tpl_gated:     checkpoints_gated/gated_epoch{epoch:03d}.pt
    """
    # 0) Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Using device: {device}\n")

    # 1) Data Loader 준비
    wloader = WindowLoader(cfg['tensor_dir'], cfg['batch'])
    total_windows = wloader.M
    batch_size = cfg['batch']
    total_batches = (total_windows + batch_size - 1) // batch_size
    print(f"▶ Loaded WindowLoader: total_windows={total_windows}, batch_size={batch_size}, total_batches={total_batches}\n")

    # 2) 모델 로드 함수 정의
    def load_single(mtype: str):
        ckpt_path = cfg['ckpt'][mtype]
        ckpt = torch.load(ckpt_path, map_location=device)

        if mtype == 'lstm':
            m = BasicLSTM(num_nodes=1370, input_dim=8, hidden_dim=64)
            m.load_state_dict(ckpt['model_state_dict'])
        elif mtype == 'stgcn':
            A_cpu = np.load(os.path.join(cfg['tensor_dir'], 'adjacency', 'A_lane.npy'))
            A_cuda = torch.from_numpy(A_cpu).float().to(device)
            m = STGCN(in_channels=9, out_channels=8, num_nodes=A_cpu.shape[0], A=torch.from_numpy(A_cpu).float())
            m.load_state_dict(ckpt['model_state_dict'])
            # 서브모듈 A 덮어쓰기
            m = m.to(device)
            for sub in m.modules():
                if hasattr(sub, 'A'):
                    sub.A = A_cuda
        elif mtype == 'gated':
            A_cpu = np.load(os.path.join(cfg['tensor_dir'], 'adjacency', 'A_lane.npy'))
            A_cuda = torch.from_numpy(A_cpu).float().to(device)
            m = GatedFusionSTGCN(
                in_channels=9,
                hidden1=cfg['hidden1'],
                hidden2=cfg['hidden2'],
                out_channels=8,
                num_nodes=A_cpu.shape[0],
                A=torch.from_numpy(A_cpu).float(),
                use_tcn=cfg.get('use_tcn', False)
            )
            m.load_state_dict(ckpt['model_state_dict'])
            m = m.to(device)
            for sub in m.modules():
                if hasattr(sub, 'A'):
                    sub.A = A_cuda
        else:
            raise ValueError(f"Unknown model type: {mtype}")

        m.eval()
        m = torch.jit.script(m)
        return m

    # 3) 모든 모델 로드
    models = {}
    for key in ['lstm','stgcn','gated']:
        print(f"▶ Loading {key.upper()}...")
        models[key] = load_single(key)
        # A-device 확인
        for sub in models[key].modules():
            if hasattr(sub, 'A'):
                print(f"   → {key.upper()} A on {sub.A.device}\n")
                break

    # 4) Inference 루프
    agg = {k: [] for k in models}
    t0 = time.time()
    print(f"▶ Inference start...\n")
    for Xb, Yb, idx_b, date_b in tqdm(wloader.batches(), total=total_batches, desc="Infer"):
        x_t = torch.from_numpy(np.transpose(Xb,(0,3,1,2))).float().to(device)
        for mtype, m in models.items():
            weekend_flag = x_t[:,8,0,0]
            preds = predict(m, x_t, mtype, device, weekend_flag=weekend_flag)
            agg[mtype].append((preds, Yb.copy(), date_b.copy(), idx_b.copy()))
    print(f"▶ Inference done in {time.time()-t0:.1f}s\n")

    # 이하 생략: agg 처리, denorm, CSV/PKL 저장 부분도 'resstgcn' → 'gated' 변경
    # ...

    # 8) 결과 저장
    save_dir = cfg['save_dir']
    with open(os.path.join(save_dir,'results_dict.pkl'),'wb') as f:
        pickle.dump(agg, f)
    print(f"▶ Saved to {save_dir}")

    return Path(save_dir)

if __name__ == '__main__':
    # eval_config.yaml 로드 후 evaluate 호출
    import yaml
    cfg = yaml.safe_load(open('eval_config.yaml'))
    evaluate(cfg)