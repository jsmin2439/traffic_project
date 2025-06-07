#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_compare.py

세 가지 모델(ST-GCN, LSTM-only, Gated-Fusion) 공정 비교 학습 스크립트

필요 파일:
  - 데이터 로더: project_root/data_loader.py
    * get_dataloaders(batch_size) → train_loader, val_loader, test_loader
    * 각 배치는 (x, y, idx) 형태: x=(B,9,12,N), y=(B,8,N)
  - 인접행렬: 3_tensor/adjacency/A_lane.npy (정규화된 adjacency matrix)
  - 모델 정의:
    * project_root/model/stgcn_model.py 에 STGCNModel 클래스
    * project_root/model/lstm_model.py 에 LSTMModel 클래스
    * project_root/model/gated_fusion_stgcn.py 에 GatedFusionSTGCN 클래스
  - 유틸리티: project_root/train_utils.py
    * EarlyStopping, ensure_dir, print_memory_usage

사용법 예시:
  python train_compare.py --model stgcn --batch 8 --epochs 40 --lr 5e-4 --hidden1 64
  python train_compare.py --model lstm  --batch 8 --epochs 40 --lr 1e-3 --hidden2 128
  python train_compare.py --model gated --batch 8 --epochs 40 --lr 5e-4 --hidden1 64 --hidden2 128 [--use_tcn]

Arguments:
  --model           {'stgcn','lstm','gated'}  비교할 모델 선택
  --batch           배치 크기 (default=8)
  --epochs          학습 에폭 수 (default=40)
  --lr              학습률 (모델별 기본값 사용 가능)
  --sched           LR 스케줄러 종류: plateau 또는 cosine (default=plateau)
  --patience        EarlyStopping patience (default=5)
  --clip            그래디언트 클리핑 max norm (default=5.0, 0=off)
  --checkpoint_dir  체크포인트 저장 디렉토리 (default='checkpoints_compare')
  --hidden1         ST-GCN hidden channels 또는 Gated-Fusion hidden1 (default=64)
  --hidden2         LSTM hidden size 또는 Gated-Fusion hidden2 (default=128)
  --use_tcn         Gated-Fusion에 TCN branch 사용 여부 (기본 LSTM)
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path
# ‘train_compare.py’ 가 있는 폴더의 부모(=프로젝트 루트)를 import 경로에 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_loader import get_dataloaders
# 모델 클래스 임포트
from model.stgcn_model import STGCN   # ST-GCN 전용 모델
from model.lstm_model import BasicLSTM     # 순수 LSTM 전용 모델
from model.gated_fusion_stgcn import GatedFusionSTGCN  # 하이브리드 모델
from train_utils import EarlyStopping, ensure_dir, print_memory_usage
from tqdm import tqdm


def parse_args():
    """
    사용자로부터 커맨드라인 인자 파싱
    반환값: args (argparse.Namespace)
    """
    p = argparse.ArgumentParser(
        description="공정 비교: ST-GCN vs LSTM vs Gated-Fusion 학습 스크립트")
    # 필수: 비교할 모델
    p.add_argument('--model', choices=['stgcn','lstm','gated'], required=True,
                   help='학습할 모델 종류 선택')
    # 공통 하이퍼파라미터
    p.add_argument('--batch', type=int, default=8,
                   help='배치 크기')
    p.add_argument('--epochs', type=int, default=40,
                   help='총 학습 에폭 수')
    p.add_argument('--lr', type=float, default=None,
                   help='학습률 (지정하지 않으면 모델별 기본값 사용)')
    p.add_argument('--sched', choices=['plateau','cosine'], default='plateau',
                   help='러닝레이트 스케줄러')
    p.add_argument('--patience', type=int, default=5,
                   help='EarlyStopping patience')
    p.add_argument('--clip', type=float, default=5.0,
                   help='그래디언트 클리핑 최대 노름 (0: off)')
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints_compare',
                   help='체크포인트 저장 디렉토리')
    # 모델별 추가 파라미터
    p.add_argument('--hidden1', type=int, default=64,
                   help='ST-GCN 채널 수 또는 Gated-Fusion hidden1')
    p.add_argument('--hidden2', type=int, default=128,
                   help='LSTM 은닉 크기 또는 Gated-Fusion hidden2')
    p.add_argument('--use_tcn', action='store_true',
                   help='Gated-Fusion에서 TCN 브랜치 사용')
    return p.parse_args()

# 모델별 기본 LR 및 하이퍼파라미터 범위
MODEL_CONFIG = {
    'stgcn': {'lr': 5e-4, 'hidden1': [64, 128]},
    'lstm':  {'lr': 1e-3, 'hidden2': [128, 256]},
    'gated': {'lr': 5e-4, 'hidden1': [64, 128], 'hidden2': [128, 256]},
}


def main():
    args = parse_args()
    # 체크포인트 디렉토리 생성
    ensure_dir(args.checkpoint_dir)

    # 디바이스 설정: GPU 우선, 없으면 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ 사용 디바이스: {device}")

    # 데이터로더 로드
    # x: (B, 9, 12, N), y: (B, 8, N)
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch)

    # 첫 배치에서 노드 수(num_nodes) 추출 (x.shape = (B,9,12,N))
    sample_x, _, *_ = next(iter(train_loader))
    num_nodes = sample_x.shape[-1]


    # 인접행렬 로드 (ST-GCN 및 Gated-Fusion 용)
    if args.model in ['stgcn', 'gated']:
        adj_path = Path('3_tensor') / 'adjacency' / 'A_lane.npy'
        assert adj_path.exists(), f"인접행렬 파일 미발견: {adj_path}"
        A = torch.from_numpy(__import__('numpy').load(str(adj_path))).float().to(device)

    # 모델 선택 및 초기화
    if args.model == 'stgcn':
        # STGCNModel 생성자: in_channels, hidden1, out_channels, num_nodes, A
        model = STGCN(
            in_channels=9,
            hidden1=args.hidden1,
            out_channels=8,
            num_nodes=num_nodes,
            A=A
        ).to(device)
    elif args.model == 'lstm':
        # BasicLSTM(num_nodes, input_dim, hidden_dim)
        # x에는 (B,9,12,N) 형태로 들어오지만, LSTM은 처음 8채널(queue+speed)만 사용합니다.
        model = BasicLSTM(
            num_nodes=num_nodes,
            input_dim=8,
            hidden_dim=args.hidden2
        ).to(device)
    else:  # gated
        model = GatedFusionSTGCN(
            in_channels=9,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            out_channels=8,
            num_nodes=num_nodes,
            A=A,
            use_tcn=args.use_tcn
        ).to(device)

    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"▶ 모델: {args.model}, 파라미터 수: {total_params}")

    # 학습률 설정: 사용자 입력 또는 기본값
    lr = args.lr if args.lr is not None else MODEL_CONFIG[args.model]['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # LR 스케줄러 설정
    if args.sched == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=args.patience
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.patience, eta_min=1e-5
        )

    # 손실 함수 및 EarlyStopping
    criterion = nn.MSELoss()
    early_stopper = EarlyStopping(patience=args.patience, min_delta=1e-5)

    # 학습 및 검증 반복
    for epoch in range(1, args.epochs + 1):
        # -------------------
        #  Training 단계
        # -------------------
        model.train()
        train_loss = 0.0
        for x, y, *_ in tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}] Train", ncols=80):
            x = x.to(device)
            y = y.to(device)
            # Gated-Fusion 및 ST-GCN용 weekend_flag 추출
            weekend_flag = x[:, 8, 0, 0]  # (B,)

            optimizer.zero_grad()
            # 순전파: 모델별로 forward 인자 분기
            if args.model == 'gated':
                y_hat = model(x, weekend_flag)
            elif args.model == 'stgcn':
                y_hat = model(x)
            else:  # 'lstm'
                y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            # 그래디언트 클리핑
            if args.clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------------------
        #  Validation 단계
        # -------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, *_ in tqdm(val_loader, desc=f"[Epoch {epoch}/{args.epochs}] Val  ", ncols=80):
                x = x.to(device)
                y = y.to(device)
                weekend_flag = x[:, 8, 0, 0]
                # 순전파: 모델별로 forward 인자 분기
                if args.model == 'gated':
                    y_hat = model(x, weekend_flag)
                elif args.model == 'stgcn':
                    y_hat = model(x)
                else:  # 'lstm'
                    y_hat = model(x)
                val_loss += criterion(y_hat, y).item()

        val_loss /= len(val_loader)
        print(f"[Epoch {epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print_memory_usage(device)

        # -------------------
        #  EarlyStopping 및 Scheduler
        # -------------------
        early_stopper.step(val_loss)
        if early_stopper.stop:
            print(f"▶ Early stopping at epoch {epoch}")
            break
        if args.sched == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # -------------------
        #  체크포인트 저장
        # -------------------
        if epoch % 5 == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"{args.model}_epoch{epoch:03d}.pt")
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss},
                       ckpt_path)
            print(f"▶ Checkpoint saved: {ckpt_path}")

    # -------------------
    #  Test 평가
    # -------------------
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y, *_ in test_loader:
            x = x.to(device)
            y = y.to(device)
            weekend_flag = x[:, 8, 0, 0]
            # 순전파: 모델별로 forward 인자 분기
            if args.model == 'gated':
                y_hat = model(x, weekend_flag)
            elif args.model == 'stgcn':
                y_hat = model(x)
            else:  # 'lstm'
                y_hat = model(x)
            test_loss += criterion(y_hat, y).item()

    test_loss /= len(test_loader)
    print(f"▶ Test Loss: {test_loss:.6f}")


if __name__ == '__main__':
    main()