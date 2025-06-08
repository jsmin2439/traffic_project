# run_sumo_collect.py
import traci
import sys
from sumo_utils import LaneStatsLogger
import os

# 실제 .sumocfg 파일 경로
SUMOCFG_PATH = "2_sumoprep/20220810/bucheon.sumocfg"
OUT_CSV_PATH = "lane_stats.csv"

STEP_SECONDS = 3600       # 1시간 = 3600초
STEP_LENGTH = 1.0         # SUMO 시간 간격 (초)

def main():
    print("▶ SUMO 시뮬 시작")
    traci.start(["sumo", "-c", SUMOCFG_PATH, "--step-length", str(STEP_LENGTH)])

    lanes = traci.lane.getIDList()
    logger = LaneStatsLogger()
    
    print(f"▶ 시뮬레이션 대상 레인 수: {len(lanes)}")

    for step in range(STEP_SECONDS):
        traci.simulationStep()
        sim_time = traci.simulation.getTime()
        logger.log_step(sim_time, lanes)

        # 디버깅용 출력 (100 step마다)
        if step % 100 == 0:
            print(f"⏱️ Step {step}/{STEP_SECONDS} | Simulation Time: {sim_time:.1f} sec")

    traci.close()
    print("▶ 시뮬 종료. CSV 저장 중...")
    logger.dump_csv(OUT_CSV_PATH)
    print(f"✅ 저장 완료: {OUT_CSV_PATH}")

if __name__ == "__main__":
    main()