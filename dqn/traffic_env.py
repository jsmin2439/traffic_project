# ===============================
# traffic_sumo_env.py
# ===============================
"""Gym‑style SUMO environment for online fine‑tuning.
Implements weighted reward & continuous control intervals.
Follows the same α/β/weights convention as build_rl_dataset.py.
"""
import os, subprocess, contextlib, itertools
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import traci
except ImportError:
    raise RuntimeError("TraCI (sumo‑tools) not found; make sure SUMO_HOME is set and tools/ is in PYTHONPATH")


class TrafficSimEnv(gym.Env):
    """SUMO online RL environment.

    Observation (shape = (8,)) ::
        [ q_norm_cat0..3 , speed_pen_cat0..3 ]  – float32, 0‑1

    Action :: Discrete( n_phases ) – index into `phase_ids` list
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 net_file: str,
                 route_file: str,
                 tl_id: str,
                 phase_ids: list[int],
                 lane_cat_map: dict[str, int],
                 step_length: float = 5.0,
                 sim_duration: int = 86400,
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 weights: tuple[float, float, float, float] = (1.0, 1.5, 2.0, 0.5),
                 sumo_binary: str = "sumo",
                 gui: bool = False):
        super().__init__()
        self.net_file = net_file
        self.route_file = route_file
        self.tl_id = tl_id
        self.phase_ids = phase_ids
        self.lane_cat_map = lane_cat_map  # laneID -> cat_idx (0‑3)
        self.step_len = step_length
        self.sim_dur = sim_duration
        self.alpha, self.beta = alpha, beta
        self.w = np.asarray(weights, dtype=np.float32)
        self.sumo_binary = "sumo‑gui" if gui else sumo_binary

        self.action_space = spaces.Discrete(len(self.phase_ids))
        self.observation_space = spaces.Box(0.0, 1.0, (8,), dtype=np.float32)

        self._conn = None
        self._launch_sumo()

    # ----------------------------- SUMO helpers ----------------------------- #
    def _launch_sumo(self):
        if self._conn:
            with contextlib.suppress(Exception):
                self._conn.close()
        sumo_cmd = [self.sumo_binary,
                    "-n", self.net_file,
                    "-r", self.route_file,
                    "--step-length", str(self.step_len),
                    "--no-warnings", "true"]
        self._conn = traci.start(sumo_cmd)[0]
        self._time = 0

    # ----------------------------- Gym API ---------------------------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._launch_sumo()
        obs = self._get_state()
        return obs, {}

    def step(self, action: int):
        # 1) set traffic‑light phase
        traci.trafficlight.setPhase(self.tl_id, self.phase_ids[action])

        # 2) advance for `step_len` seconds (SUMO internal smaller step)
        ticks = int(np.ceil(self.step_len / traci.simulation.getDeltaT()))
        for _ in range(ticks):
            traci.simulationStep()
            self._time += traci.simulation.getDeltaT()

        obs = self._get_state()
        reward = -np.dot(self.w, self.alpha * self._q_norm + self.beta * self._spd_pen)
        terminated = self._time >= self.sim_dur
        return obs, float(reward), terminated, False, {}

    # ------------------------------------------------------------------------ #
    def _lane_stats(self, cat_idx: int):
        """Return (queue_len, speed_avg) for a vehicle category index."""
        queues, speeds = [], []
        for lane_id, cat in self.lane_cat_map.items():
            if cat != cat_idx:
                continue
            veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            q_len = sum(traci.vehicle.getSpeed(v) < 0.1 for v in veh_ids)  # stopped vehicles ≈ queue
            if veh_ids:
                spd = np.mean([traci.vehicle.getSpeed(v) for v in veh_ids])
            else:
                spd = self._lane_vmax[lane_id]
            queues.append(q_len)
            speeds.append(spd)
        # handle empty lists gracefully
        q = np.mean(queues) if queues else 0.0
        v = np.mean(speeds) if speeds else self._v_ref
        return q, v

    @property
    def _lane_vmax(self):
        if not hasattr(self, "__lane_vmax"):
            self.__lane_vmax = {l: traci.lane.getMaxSpeed(l) for l in traci.lane.getIDList()}
        return self.__lane_vmax

    @property
    def _v_ref(self):
        if not hasattr(self, "__v_ref"):
            self.__v_ref = np.mean(list(self._lane_vmax.values()))
        return self.__v_ref

    def _get_state(self):
        queues, speeds = zip(*[self._lane_stats(c) for c in range(4)])
        queues = np.array(queues, dtype=np.float32)
        speeds = np.array(speeds, dtype=np.float32)
        self._q_norm = np.clip(queues / 25.0, 0.0, 1.0)
        self._spd_pen = np.clip(1.0 - speeds / self._v_ref, 0.0, 1.0)
        return np.concatenate([self._q_norm, self._spd_pen], dtype=np.float32)

    # ----------------------------- utils ------------------------------------ #
    def close(self):
        if self._conn:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None
