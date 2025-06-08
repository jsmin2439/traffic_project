#!/usr/bin/env python3
# extract_all_predictions_v3.py
#───────────────────────────────────────────────────────────────────────────
# 1) ResSTGCN 체크포인트 로드
# 2) all_X.npy 순회하며 예측
# 3) --inverse 옵션 시, 날짜별 input_tensor_{date}.pkl 로부터
#    채널 0‒7 의 μ·σ를 계산하여 역정규화 (+클램핑)
# 4) CSV 저장
#───────────────────────────────────────────────────────────────────────────
import argparse, csv, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from tqdm import tqdm

#───────────────────────── CLI ─────────────────────────
def parse_args():
    p = argparse.ArgumentParser("Dump ResSTGCN predictions → CSV")
    p.add_argument("--ckpt", required=True,            help="checkpoint .pt")
    p.add_argument("--A",    required=True,            help="adjacency A.npy")
    p.add_argument("--X",    required=True,            help="all_X.npy (B,12,N,9)")
    p.add_argument("--tensor_root", required=True,     help="root dir with per-date tensors")
    p.add_argument("--lane_txt",     help="lanes_list.txt (optional)")
    p.add_argument("--map_csv",      help="window_idx↔date CSV (col: window_idx,date)")
    p.add_argument("--out_csv",      default="predictions.csv")
    p.add_argument("--inverse",      action="store_true",
                   help="apply per-date denormalization & clamp to ≥0")
    p.add_argument("--hidden_dim",   type=int, default=256)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

args = parse_args()
DEVICE = torch.device(args.device)

#──────────────────────── lane 목록 ──────────────────────
lane_ids = (Path(args.lane_txt).read_text().splitlines()
            if args.lane_txt and Path(args.lane_txt).exists() else None)

#──────────────────── window→date 매핑 ───────────────────
if args.map_csv and Path(args.map_csv).exists():
    df_map  = pd.read_csv(args.map_csv)
    win2date = dict(zip(df_map.window_idx, df_map.date.astype(str)))
else:
    win2date = {}          # 매핑이 없으면 '00000000' 가상 날짜 사용

#───────────────── μ·σ 로더 (채널 0–7) ───────────────────
_mu_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

def load_mu_sigma(date: str) -> tuple[np.ndarray, np.ndarray]:
    """
    주어진 날짜 폴더에서 input_tensor_{date}.pkl 을 열어
    (288, N, 8/9) 텐서를 평평하게 펼친 뒤 채널 0–7 의 μ·σ 계산.
    캐시 사용으로 중복 계산 최소화.
    """
    if date in _mu_cache:
        return _mu_cache[date]

    pkl_path = Path(args.tensor_root) / date / f"input_tensor_{date}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"[μ·σ] {pkl_path} 가 존재하지 않습니다.")

    with open(pkl_path, "rb") as f:
        tensor = pickle.load(f)              # (288, N, 8) or (288, N, 9)
    tensor = np.asarray(tensor, dtype=np.float32)
    if tensor.shape[-1] < 8:
        sys.exit(f"[μ·σ] 잘못된 채널 수: {tensor.shape}")

    flat = tensor[..., :8].reshape(-1, 8)     # queue+speed 8 채널만
    mu  = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    _mu_cache[date] = (mu, std)
    return mu, std

#────────────────── ResSTGCN 모델 로드 ───────────────────
from model.res_stgcn_model import ResSTGCN
A = torch.from_numpy(np.load(args.A)).float().to(DEVICE)
model = ResSTGCN(9, 8, A.shape[0], A, args.hidden_dim).to(DEVICE)

ckpt = torch.load(args.ckpt, map_location=DEVICE)
model.stgcn.load_state_dict(ckpt["stgcn_state_dict"])
model.reslstm.load_state_dict(ckpt["reslstm_state_dict"])
model.eval()

#─────────────────── 입력 윈도우 로드 ────────────────────
X = np.load(args.X)                     # (B,12,N,9)
B, _, N, _ = X.shape

#────────────────────── CSV 덤프 ─────────────────────────
header = ["window_idx", "channel", "lane_idx", "lane_id", "pred_norm"]
if args.inverse:
    header.append("pred_phys")

with open(args.out_csv, "w", newline="", encoding="utf-8") as fp:
    wr = csv.writer(fp);  wr.writerow(header)

    for w in tqdm(range(B), ncols=80, desc="Predict"):
        x = torch.from_numpy(X[w]).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            y = model(x).cpu().squeeze(0).numpy()        # (8, N)

        if args.inverse:
            date = win2date.get(w, "00000000")
            mu, std = load_mu_sigma(date)

        for ch in range(8):
            for ln in range(N):
                lane_id = lane_ids[ln] if lane_ids else ""
                row = [w, ch, ln, lane_id, float(y[ch, ln])]
                if args.inverse:
                    phys = max(0.0, y[ch, ln] * std[ch] + mu[ch]) if std[ch] > 0 else mu[ch]
                    row.append(float(phys))
                wr.writerow(row)

        if (w + 1) % 50 == 0:
            fp.flush()

print(f"\n✅  예측 CSV 저장 완료 → {args.out_csv}")