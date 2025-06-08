# ===============================
# train_dqn.py
# ===============================
"""Unified launcher for offline pre‑training + online fine‑tuning.

Offline mode  (TrafficReplayEnv) →  produces dqn_pretrain.zip
Online  mode  (TrafficSimEnv)   →  loads the above and fine‑tunes via TraCI.

Rewards/actions evolve each epoch because the environment always recomputes
   * reward   = α/β/weights ‑ weighted aggregate (see above)
   * action   = ε‑greedy choice from the current DQN – policy keeps updating
"""
import argparse, pathlib, json
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# local imports
from traffic_sumo_env import TrafficSimEnv
from traffic_replay_env import TrafficReplayEnv

DEVICE = "cuda"  # change to "cpu" if no GPU


def make_replay_env(state_path, reward_path, ep_len):
    states  = np.load(state_path)
    rewards = np.load(reward_path)
    return lambda: TrafficReplayEnv(states, rewards, ep_len)


def make_sumo_env(net, rou, tl_id, phase_ids, lane_map_json):
    lane_cat_map = json.load(open(lane_map_json))  # {laneID: cat_idx}
    return lambda: TrafficSimEnv(net, rou, tl_id, phase_ids, lane_cat_map)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "sim"], required=True)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--state", help="states.npy (offline mode)")
    parser.add_argument("--reward", help="rewards.npy (offline mode)")
    parser.add_argument("--net", help=".net.xml (sim mode)")
    parser.add_argument("--rou", help=".rou.xml (sim mode)")
    parser.add_argument("--tl-id", default="cluster_1")
    parser.add_argument("--phase-ids", nargs="*", type=int, default=[0, 2, 4, 6])
    parser.add_argument("--lane-map", help="lane_cat_map.json (sim mode)")
    parser.add_argument("--load", default=None, help="path to pre‑trained .zip")
    parser.add_argument("--save", default="model_out.zip")
    args = parser.parse_args()

    if args.mode == "offline":
        if not (args.state and args.reward):
            parser.error("--state and --reward required for offline mode")
        env = DummyVecEnv([make_replay_env(args.state, args.reward, 288)])
        model = DQN("MlpPolicy", env, verbose=1, device=DEVICE, buffer_size=100_000)

    else:  # sim
        if not (args.net and args.rou and args.lane_map):
            parser.error("--net, --rou, --lane-map required for sim mode")
        env = DummyVecEnv([make_sumo_env(args.net, args.rou, args.tl_id, args.phase_ids, args.lane_map)])
        if args.load:
            model = DQN.load(args.load, env=env, device=DEVICE)
        else:
            model = DQN("MlpPolicy", env, verbose=1, device=DEVICE, buffer_size=50_000)

    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)
    print("model saved to", args.save)


if __name__ == "__main__":
    main()
