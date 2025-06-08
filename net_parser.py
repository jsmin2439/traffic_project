#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
net_parser.py

SUMO 네트워크 파일(*.net.xml)에서
‘phase 제어 대상 connection (linkIndex)’ 정보를 추출하여 CSV로 저장합니다.

출력 CSV 형식:
  tl_id, linkIndex, fromLane, toLane

사용법:
  python net_parser.py --net path/to/1200012n.net.xml --out path/to/linkIndex_map.csv
"""

import xml.etree.ElementTree as ET
import argparse
import csv
import sys

def build_link_map(net_xml_path: str, out_csv_path: str) -> None:
    """
    1) SUMO net XML을 파싱
    2) <connection> 태그 중 'tl' 속성이 있는 것만 필터
    3) (tl_id, linkIndex, fromLane, toLane) 정보를 CSV로 저장
    """
    try:
        tree = ET.parse(net_xml_path)
    except Exception as e:
        print(f"▶ Error: 네트워크 XML 파싱 실패: {e}")
        sys.exit(1)

    root = tree.getroot()

    records = []
    # <connection> 태그 탐색
    for conn in root.iter('connection'):
        tl   = conn.get('tl')  # traffic light ID
        if tl is None:
            # tl 속성이 없는 연결은 신호제어 대상이 아님
            continue

        try:
            idx  = int(conn.get('linkIndex'))
        except (TypeError, ValueError):
            # linkIndex가 없거나 정수 변환이 불가능하면 건너뜀
            continue

        frm  = conn.get('from')
        to   = conn.get('to')
        if frm is None or to is None:
            continue

        records.append((tl, idx, frm, to))

    if not records:
        print("▶ Warning: tl 속성(신호제어 대상)이 있는 <connection>이 한 건도 없습니다.")
    else:
        print(f"▶ 추출된 신호제어 connection 개수: {len(records)}")

    # CSV로 저장
    try:
        with open(out_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['tl_id', 'linkIndex', 'fromLane', 'toLane'])
            writer.writerows(records)
        print(f"▶ linkIndex 매핑 정보가 '{out_csv_path}'에 저장되었습니다.")
    except Exception as e:
        print(f"▶ Error: CSV 저장 실패: {e}")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(
        description="SUMO net XML에서 traffic-light 제어 대상 connection(linkIndex) 정보를 CSV로 추출"
    )
    parser.add_argument(
        '--net',
        type=str,
        required=True,
        help="파싱할 SUMO 네트워크 파일 경로 (*.net.xml)"
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help="출력할 CSV 파일 경로 (예: linkIndex_map.csv)"
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    build_link_map(args.net, args.out)