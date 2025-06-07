#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gated_fusion.py

Gated Fusion ST-GCN + Temporal Branch 모델 학습 스크립트
  - ST-GCN: 과거 교통 데이터(큐량+속도+주말 플래그)로 단기 시공간 패턴 추출
  - Temporal Branch: LSTM 또는 TCN으로 주중/주말 등 반복 주기 패턴 보정
  - Gated Fusion: 두 분기의 출력을 적응적으로 융합하여 최종 예측
  - 단일 스텝(5분 후) 트래픽 상태 예측

1) get_dataloaders()로 학습/검증/테스트 데이터 로드
2) 정규화된 인접행렬 A 로드
3) GatedFusionSTGCN 모델 인스턴스화
4) 학습 루프:
   - 모델 예측 y_hat = model(x, weekend_flag)
   - 손실 MSE(y_hat, y_true) 계산 및 역전파
5) 검증, 로그 출력, EarlyStopping, LR 스케줄러, 체크포인트 저장
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders
from model.gated_fusion_stgcn import GatedFusionSTGCN
from train_utils import EarlyStopping, ensure_dir, print_memory_usage
from tqdm import tqdm


def parse_args():
    """
    명령행 인자 파싱 함수
    --batch: 배치 크기
    --epochs: 학습 에폭 수
    --lr: 전체 파라미터 학습률
    --hidden1: ST-GCN 히든 채널 수
    --hidden2: Temporal Branch 히든 크기
    --sched: LR 스케줄러 종류(plateau 또는 cosine)
    --patience: EarlyStopping patience
    --clip: 그래디언트 클리핑 최대 노름
    --checkpoint_dir: 체크포인트 저장 디렉토리
    --use_tcn: Temporal Branch로 TCN 대신 LSTM 사용 여부
    """
    p = argparse.ArgumentParser(description="Train Gated Fusion ST-GCN + Temporal Branch Model")
    p.add_argument('--batch', type=int, default=4, help='Batch size')
    p.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate for all params')
    p.add_argument('--hidden1', type=int, default=64, help='ST-GCN hidden channels')
    p.add_argument('--hidden2', type=int, default=128, help='Temporal branch hidden size')
    p.add_argument('--sched', choices=['plateau','cosine'], default='plateau', help='LR scheduler')
    p.add_argument('--patience', type=int, default=5, help='EarlyStopping patience')
    p.add_argument('--clip', type=float, default=5.0, help='Gradient clipping max norm (0=off)')
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints_gated', help='Checkpoint directory')
    p.add_argument('--use_tcn', action='store_true', help='Use TCN branch instead of LSTM')
    return p.parse_args()


def main():
    # 1) 인자 파싱 및 체크포인트 디렉토리 준비
    args = parse_args()
    ensure_dir(args.checkpoint_dir)

    # 2) 디바이스 설정 (CUDA > CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True  # cuDNN 최적화 활성화
    else:
        device = torch.device('cpu')
    print(f"▶ Using device: {device}")

    # 3) 데이터로더 생성
    #    train_loader, val_loader, test_loader 반환
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch)

    # 4) 인접행렬 A 로드
    adj_path = Path('3_tensor') / 'adjacency' / 'A_lane.npy'
    assert adj_path.exists(), f"Adjacency file not found: {adj_path}"
    # NumPy 로드 후 Tensor로 변환
    A = torch.from_numpy(__import__('numpy').load(str(adj_path))).float().to(device)

    # 5) 모델 생성
    num_nodes = A.shape[0]
    model = GatedFusionSTGCN(
        in_channels=9,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        out_channels=8,
        num_nodes=num_nodes,
        A=A,
        use_tcn=args.use_tcn
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"▶ Parameter count: {total_params}")

    # 6) 손실 함수 및 최적화 도구 설정
    criterion = nn.MSELoss()  # 회귀 문제에 MSE 사용
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # LR 스케줄러 설정
    if args.sched == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.patience, eta_min=1e-5)

    # EarlyStopping 초기화
    early_stopper = EarlyStopping(patience=args.patience, min_delta=1e-5)

    # 7) 학습 루프
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        # A) Training 단계
        for x, y, *_ in tqdm(train_loader,
                             desc=f"[Epoch {epoch}/{args.epochs}] Train",
                             ncols=80):
            # x: (B,9,12,N), y: (B,8,N)
            x = x.to(device)
            y = y.to(device)
            # weekend_flag: 주말 여부 채널 추출 (B,)
            weekend_flag = x[:, 8, 0, 0]

            optimizer.zero_grad()
            # 모델 순전파
            y_hat = model(x, weekend_flag)  # (B,8,N)
            # 손실 계산 및 역전파
            loss = criterion(y_hat, y)
            loss.backward()
            # 그래디언트 클리핑
            if args.clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        # B) Validation 단계
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, *_ in tqdm(val_loader,
                                 desc=f"[Epoch {epoch}/{args.epochs}] Val",
                                 ncols=80):
                x = x.to(device)
                y = y.to(device)
                weekend_flag = x[:, 8, 0, 0]
                y_hat = model(x, weekend_flag)
                val_loss += criterion(y_hat, y).item()
            val_loss /= len(val_loader)

        # C) 로그 출력 및 메모리 사용량 모니터링
        print(f"[Epoch {epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print_memory_usage(device)

        # D) EarlyStopping 및 LR 스케줄러 업데이트
        early_stopper.step(val_loss)
        if early_stopper.stop:
            print(f"▶ Early stopping at epoch {epoch}")
            break
        if args.sched == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # E) 체크포인트 저장 (5 에폭마다)
        if epoch % 5 == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f'gated_epoch{epoch:03d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, ckpt_path)
            print(f"▶ Checkpoint saved: {ckpt_path}")

    # 8) 테스트 평가
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y, *_ in test_loader:
            x = x.to(device)
            y = y.to(device)
            weekend_flag = x[:, 8, 0, 0]
            y_hat = model(x, weekend_flag)
            test_loss += criterion(y_hat, y).item()
        test_loss /= len(test_loader)
    print(f"▶ Test Loss: {test_loss:.6f}")


if __name__ == '__main__':
    main()