import os, pickle, numpy as np
from pathlib import Path

class WindowLoader:
    """
    NumPy 윈도우·정규화통계 로더
    """
    def __init__(self, tensor_dir: str, batch: int = 64):
        self.tensor_dir = Path(tensor_dir)
        self.batch      = batch

        # 윈도우 · 날짜
        win_dir = self.tensor_dir / "windows"
        # → 디렉터리나 파일이 없을 경우 명시적 에러를 띄워주도록 변경
        if not (win_dir / "all_X.npy").exists():
            raise FileNotFoundError(f"{win_dir}/all_X.npy 파일을 찾을 수 없습니다.")
        if not (win_dir / "all_Y.npy").exists():
            raise FileNotFoundError(f"{win_dir}/all_Y.npy 파일을 찾을 수 없습니다.")
        if not (win_dir / "all_DATE.npy").exists():
            raise FileNotFoundError(f"{win_dir}/all_DATE.npy 파일을 찾을 수 없습니다.")

        self.X  = np.load(win_dir / "all_X.npy")   # (M,12,1370,9)
        self.Y  = np.load(win_dir / "all_Y.npy")   # (M,1370,8)
        self.D  = np.load(win_dir / "all_DATE.npy")# (M,)
        self.M  = self.X.shape[0]

        # 날짜별 mean/std 캐시
        self.stats = {}
        self._prepare_stats()

    # ---------- public ----------------------------------------------------
    def batches(self):
        """yield batched (X,Y,idx,DATE)"""
        for i in range(0, self.M, self.batch):
            sl = slice(i, i+self.batch)
            yield (self.X[sl], self.Y[sl],
                   np.arange(*sl.indices(self.M)),
                   self.D[sl])

    def denorm(self, arr_norm, date_vec):
        """배열(arr_norm[*,1370,8])을 날짜별 mean/std로 역정규화"""
        arr_o = np.empty_like(arr_norm, dtype=np.float32)
        for d in np.unique(date_vec):
            mask = (date_vec == d)
            if int(d) not in self.stats:
                raise KeyError(f"stats에 날짜 {d}에 대한 mean/std 정보가 없습니다.")
            mu, std = self.stats[int(d)]   # mu, std 모두 shape=(8,)
            # arr_norm[mask]의 shape=(N_date,1370,8)이므로, std+mu 브로드캐스트가 정상 동작
            arr_o[mask] = arr_norm[mask] * std + mu
        return arr_o
    # ----------------------------------------------------------------------

    def _prepare_stats(self):
        dates = np.unique(self.D)
        for d in dates:
            day_dir = self.tensor_dir / str(int(d))
            pkl = day_dir / f"normalized_tensor_{d}_with_weekend.pkl"
            mu = std = None
            if pkl.exists():
                obj = pickle.load(open(pkl, 'rb'))
                if isinstance(obj, dict) and 'mean' in obj and 'std' in obj:
                    mu  = np.asarray(obj['mean'][:8], dtype=np.float32)
                    std = np.asarray(obj['std'] [:8], dtype=np.float32)
            if mu is None:
                ipkl = day_dir / f"input_tensor_{d}.pkl"
                if not ipkl.exists():
                    # normalized_tensor에도 통계 정보가 없고, input_tensor 파일도 없을 때 명확한 예외를 던집니다.
                    raise FileNotFoundError(f"{day_dir}/normalized_tensor_{d}_with_weekend.pkl 에 통계 정보가 없고, {ipkl}도 존재하지 않습니다.")
                raw  = pickle.load(open(ipkl,'rb'))  # (288,1370,>=8)
                # raw[:,:,0:8]은 queue4+speed4 채널을 의미하므로 axis=(0,1)으로 평균/표준편차를 구합니다.
                mu  = raw[:, :, :8].mean((0, 1)).astype(np.float32)
                std = raw[:, :, :8].std ((0, 1)).astype(np.float32)
            self.stats[int(d)] = (mu, std)