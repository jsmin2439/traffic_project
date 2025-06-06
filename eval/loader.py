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
            mask = date_vec == d
            mu, std = self.stats[int(d)]
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
                raw  = pickle.load(open(ipkl,'rb'))  # (288,1370,>=8)
                mu  = raw[:,:,:8].mean((0,1)).astype(np.float32)
                std = raw[:,:,:8].std ((0,1)).astype(np.float32)
            self.stats[int(d)] = (mu, std)