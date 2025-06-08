#!/usr/bin/env python3
# sumo_utils.py
"""
SUMO 실시간 통계 래퍼:
  - step 단위로 lane 단위 queue / speed를 수집하여 dict 반환
  - 필요하면 pandas DataFrame 으로 누적 저장
"""

from typing import Dict, Tuple
import traci
import time

def read_lane_stats(lanes: list[str]) -> Dict[str, Tuple[int, float]]:
    """
    Returns
    -------
    stats : { lane_id: (queue_cnt, mean_speed_kmh) }
    """
    stats = {}
    for lid in lanes:
        q = traci.lane.getLastStepHaltingNumber(lid)
        v = traci.lane.getLastStepMeanSpeed(lid) * 3.6  # m/s → km/h
        stats[lid] = (q, v)
    return stats


class LaneStatsLogger:
    """누적 버퍼 + 파일 저장용 간단 래퍼"""
    def __init__(self):
        self.buffer = []   # (sim_time, lane_id, queue, speed)

    def log_step(self, sim_time: float, lanes: list[str]):
        stats = read_lane_stats(lanes)
        for lid, (q, v) in stats.items():
            self.buffer.append((sim_time, lid, q, v))

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.buffer,
                            columns=['t', 'lane', 'queue', 'speed'])

    def dump_csv(self, path: str):
        df = self.to_dataframe()
        df.to_csv(path, index=False)