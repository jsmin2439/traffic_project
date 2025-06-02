import os
import shutil
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from string import Template
from tqdm import tqdm
from datetime import datetime, timedelta
import sumolib

def build_sumoprep(
    lake_dir: str = "1_lake",
    prep_dir: str = "2_sumoprep",
    turn_parquet: str = "turn.parquet",
    signal_parquet: str = "signal_xml.parquet",
    net_dir: str = "1_lake/net",
    net_file: str = "bucheon.net.xml",
    exp_route: str = "exp.rou.xml"
):
    """
    Generate SUMO prep files: demand.rou.xml, detectors.add.xml,
    edgeGroups.add.xml, tls_plan.tll.xml, sim_conf.sumocfg
    for each date folder under 2_sumoprep.
    """
    # Ensure directories are relative to project root, not script location
    base_dir = Path(__file__).resolve().parents[2]
    lake = base_dir / lake_dir
    prep = base_dir / prep_dir
    # lake = Path(lake_dir)
    # prep = Path(prep_dir)
    prep.mkdir(parents=True, exist_ok=True)
    # ensure source exist
    turn_pq = lake / turn_parquet
    sig_pq = lake / signal_parquet
    net_src = lake / "net" / net_file
    # 실제 exp.rou.xml은 1_lake/net 디렉토리에 있으므로 lake 디렉토리를 기준으로 복사 경로 지정
    exp_src = lake / "net" / exp_route

    # generate all dates from 20220810 to 20221031
    start = datetime.strptime("20220810", "%Y%m%d")
    end = datetime.strptime("20221031", "%Y%m%d")
    dates = []
    curr = start
    while curr <= end:
        dates.append(curr.strftime("%Y%m%d"))
        curr += timedelta(days=1)

    # load net edges and lanes once for detectors
    net_obj = sumolib.net.readNet(str(net_src))
    node_list = [edge.getID() for edge in net_obj.getEdges()]
    # prepare lane list and lengths for detectors
    lane_list = []
    lane_lengths = {}
    for edge in net_obj.getEdges():
        for lane in edge.getLanes():
            lid = lane.getID()
            lane_list.append(lid)
            lane_lengths[lid] = lane.getLength()
    # Build set of valid edge_from → edge_to pairs (direct connections)
    valid_pairs = set()
    for edge in net_obj.getEdges():
        for succ in edge.getOutgoing():
            valid_pairs.add((edge.getID(), succ.getID()))

    # template for sim_conf.sumocfg
    CFG_TMPL = Template(
        '<configuration>\n'
        '  <input>\n'
        '    <net-file value="$net"/>\n'
        '    <route-files value="$exp,demand.rou.xml"/>\n'
        '    <additional-files value="detectors.add.xml,edgeGroups.add.xml,tls_plan.tll.xml"/>\n'
        '  </input>\n'
        '  <time begin="0" end="86400"/>\n'
        '</configuration>\n'
    )

    for date in tqdm(dates, desc="날짜별 SUMO 준비", unit="일"):
        date_dir = prep / date
        date_dir.mkdir(parents=True, exist_ok=True)

        # compute epoch for simulation start of this date
        date_midnight = datetime.strptime(date, "%Y%m%d")
        date_epoch = int(date_midnight.timestamp())

        # 1) demand.rou.xml
        df_turn = (
            pq.ParquetDataset(turn_pq, filters=[('date', '=', int(date))])
            .read(columns=['dt', 'edge_from', 'edge_to', 'veh_type', 'count', 'begin', 'end'])
            .to_pandas()
        )
        # ensure flows are sorted by departure time
        df_turn['begin'] = df_turn['begin'].astype(float)
        df_turn = df_turn.sort_values(by='begin')
        # map raw veh_type codes to defined vType ids in exp.rou.xml
        vtype_map = {
            'p': '1',       # passenger
            'b': '20',      # bus
            't': '10',      # truck
            'st': '10',     # small truck
            'lt': '10',     # large truck
            's': 'emergency',  # special/emergency vehicles
            'm': '40'       # motorcycle
        }
        if df_turn.empty:
            print(f'[SUMOPREP] No turn data for date {date}, skipping.')
            continue
        with open(date_dir / 'demand.rou.xml', 'w', encoding='utf-8') as f:
            f.write('<routes>\n')
            for i, (_, r) in enumerate(df_turn.iterrows()):
                vt = vtype_map.get(r.veh_type, r.veh_type)
                # convert absolute UNIX time to seconds since simulation start
                beg = int(float(r.begin)) - date_epoch
                end = int(float(r.end)) - date_epoch
                count = int(r['count'])
                if count <= 0:
                    continue
                if beg < 0 or beg > 86400:
                    continue
                # Only write flow if edge_from and edge_to are directly connected
                if (r.edge_from, r.edge_to) in valid_pairs:
                    f.write(
                        f'  <flow id="f_{r.edge_from}_{r.edge_to}_{beg}_{i}" '
                        f'type="{vt}" begin="{beg}" end="{end}" '
                        f'from="{r.edge_from}" to="{r.edge_to}" number="{count}"/>\n'
                    )
                else:
                    continue
            f.write('</routes>\n')

        # 2) detectors.add.xml (E2 detectors on every lane)
        with open(date_dir / 'detectors.add.xml', 'w', encoding='utf-8') as f:
            f.write('<additional>\n')
            for lid in lane_list:
                length = lane_lengths.get(lid, 0)
                f.write(
                    f'  <e2Detector id="det_{lid}" lane="{lid}" pos="0" endPos="{length}" freq="60" '
                    f'file="detectors_output.xml"/>\n'
                )
            f.write('</additional>\n')

        # 3) edgeGroups.add.xml (관심 구간 KPI 그룹)
        # Define one or more edge groups for KPI measurement
        # Example: a single group including all edges in this route
        with open(date_dir / 'edgeGroups.add.xml', 'w', encoding='utf-8') as f:
            f.write('<additional>\n')
            # Here, lane_list contains all lane IDs; convert to edge IDs by splitting at '_'
            # or use net_obj.getEdges() to collect edge IDs
            edge_ids = [edge.getID() for edge in net_obj.getEdges()]
            f.write(f'  <edgeGroup id="all_edges" edges="{" ".join(edge_ids)}"/>\n')
            f.write('</additional>\n')

        # 4) tls_plan.tll.xml
        df_sig = (
            pq.ParquetDataset(sig_pq, filters=[('date', '=', int(date))])
            .read(columns=['id', 'phase_state', 'phase_duration'])
            .to_pandas()
        )
        avg_dur = df_sig.groupby(['id', 'phase_state'])['phase_duration'].mean()
        with open(date_dir / 'tls_plan.tll.xml', 'w', encoding='utf-8') as f:
            f.write('<additional>\n')
            for (iid, state), dur in avg_dur.items():
                f.write(f'  <phase duration="{int(dur)}" state="{state}"/>\n')
            f.write('</additional>\n')

        # 5) copy net and exp.rou.xml
        shutil.copy(net_src, date_dir / net_file)
        shutil.copy(exp_src, date_dir / exp_route)

        # 6) sim_conf.sumocfg
        cfg = CFG_TMPL.substitute(net=net_file, exp=exp_route)
        with open(date_dir / 'bucheon.sumocfg', 'w', encoding='utf-8') as f:
            f.write(cfg)

        print(f'[SUMOPREP] Generated for date {date}')


if __name__ == '__main__':
    build_sumoprep()