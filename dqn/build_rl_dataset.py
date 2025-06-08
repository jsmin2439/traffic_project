# ===============================
# build_rl_dataset.py
# ===============================
"""Re‑generate replay buffers (states.npy, rewards.npy)
with the new weighted‑reward formulation:
    R_t = - Σ_cat w_cat ( α * Q_norm + β * speed_pen )

* weights  = [1, 1.5, 2, 0.5]            # PV, BUS, TRK, MC
* α, β     = 0.7, 0.3
* Q_max    : empirical single‑lane capacity (veh) – CLI arg, default 25
* v_ref    : desired free‑flow speed (m/s) – CLI arg, default 8.3 (≃30 km/h)

Usage
-----
$ python build_rl_dataset.py \
        --state-npy  raw_states.npy     # shape (T, 8, n_lanes)
        --out-prefix bucheon            # => bucheon_states.npy / bucheon_rewards.npy
        --q-max 23 --v-ref 8.9
"""
import argparse, pathlib, numpy as np

W_DEFAULT = np.array([1.0, 1.5, 2.0, 0.5], dtype=np.float32)  # PV / BUS / TRK / MC


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--state-npy", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--q-max", type=float, default=25.0)
    p.add_argument("--v-ref", type=float, default=8.3)
    p.add_argument("--weights", type=float, nargs=4, default=W_DEFAULT,
                   help="Four weights for PV, BUS, TRK, MC – in that order")
    return p.parse_args()


def main():
    args = parse_args()
    state = np.load(args.state_npy)  # (T, 8, n_lanes)
    if state.ndim != 3 or state.shape[1] != 8:
        raise ValueError("state array must have shape (T,8,N_lanes)")

    # queues : 1st 4 features (per lane) – average over lanes
    queues = state[:, 0:4, :].mean(axis=2)   # (T,4)
    # speeds : next 4 features – average over lanes
    speeds = state[:, 4:8, :].mean(axis=2)   # (T,4)  – in m/s

    q_norm  = queues / args.q_max            # 0‑1 clip handled later
    speed_pen = np.clip(1 - speeds / args.v_ref, 0.0, 1.0)

    rewards = - (0.7 * q_norm + 0.3 * speed_pen) @ np.array(args.weights)

    # save
    prefix = pathlib.Path(args.out_prefix)
    np.save(prefix.with_suffix("_states.npy"), state)
    np.save(prefix.with_suffix("_rewards.npy"), rewards.astype(np.float32))
    print(f"saved {prefix}_states.npy and {prefix}_rewards.npy – shape", state.shape, rewards.shape)


if __name__ == "__main__":
    main()
