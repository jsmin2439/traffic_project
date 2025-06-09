#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_by_edge.py

SUMO .net.xml 파일에서 edge→lane 매핑을 추출하고,
동일 도로, 동일 방향, 인접 edge 단위로 최대 N개씩 묶어
lanes.txt 순서대로 lane→segment ID 배열을 생성합니다.

Usage:
  python cluster_by_edge.py \
    --net bucheon.net.xml \
    --lanes lanes.txt \
    --output lane_to_segment_id_by_edge.npy \
    [--max_group_size N]

옵션:
  --max_group_size N : 하나의 segment에 묶을 최대 edge 개수
                       (예: N=3이면, 그룹당 최대 3개 edge → 병합)
                       기본값: 1 (병합 없음)
"""
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import sys
from collections import defaultdict, deque

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate lane→segment mapping with dynamic merging"
    )
    p.add_argument('--net',            required=True, help='SUMO .net.xml')
    p.add_argument('--lanes',          required=True, help='lanes.txt (one lane ID per line)')
    p.add_argument('--output',         required=True, help='out .npy file')
    p.add_argument('--max_group_size', type=int, default=1,
                   help='그룹당 최대 edge 개수 (default:1)')
    return p.parse_args()

def main():
    args = parse_args()
    M = args.max_group_size

    # 1) .net.xml 파싱
    try:
        tree = ET.parse(args.net)
    except Exception as e:
        print(f"네트워크 파일 파싱 실패: {e}", file=sys.stderr)
        sys.exit(1)
    root = tree.getroot()

    # 2) edge 정보 수집: id, from-node, to-node
    edge_info = {}
    for edge in root.findall('.//edge'):
        eid  = edge.get('id')
        src  = edge.get('from')
        dst  = edge.get('to')
        edge_info[eid] = (src, dst)

    # 3) 인접성 그래프 생성 (방향 고려)
    neighbors = defaultdict(set)
    for e1, (s1,d1) in edge_info.items():
        for e2, (s2,d2) in edge_info.items():
            if d1 == s2:
                neighbors[e1].add(e2)
                neighbors[e2].add(e1)

    # 4) 연결 컴포넌트별로 최대 M개씩 sub-group으로 나눔
    visited = set()
    segment_id = {}
    seg_counter = 0
    for eid in edge_info:
        if eid in visited:
            continue
        comp = []
        dq = deque([eid])
        visited.add(eid)
        while dq:
            cur = dq.popleft()
            comp.append(cur)
            for nb in neighbors[cur]:
                if nb not in visited:
                    visited.add(nb)
                    dq.append(nb)
        # comp 내에서 M개씩 끊어서 ID 할당
        for i in range(0, len(comp), M):
            for e in comp[i:i+M]:
                segment_id[e] = seg_counter
            seg_counter += 1

    # 5) lanes.txt 읽고 lane→edge→segment 매핑
    with open(args.lanes) as f:
        lanes = [ln.strip() for ln in f if ln.strip()]

    mapping = np.zeros(len(lanes), dtype=np.int32)
    for i, lid in enumerate(lanes):
        base_edge = lid.rsplit('_', 1)[0]       # e.g. "edgeID_0" → "edgeID"
        mapping[i] = segment_id.get(base_edge, -1)

    # --- 여기서부터 리인덱싱 및 검증 ---
    # 6) 빈 클러스터(사용되지 않은 segment ID) 제거하고
    #    0 이상 ID만 0부터 연속된 ID로 재할당
    used_ids = sorted({sid for sid in mapping.tolist() if sid >= 0})
    new_id_map = {old: new for new, old in enumerate(used_ids)}
    mapping_compressed = np.array(
        [new_id_map[sid] if sid >= 0 else -1 for sid in mapping],
        dtype=np.int32
    )

    # 7) 결과 저장 및 요약 출력
    np.save(args.output, mapping_compressed)
    print(f"저장 완료: '{args.output}'")
    print(f"총 lanes: {len(lanes)}")
    print(f"원본 segment 개수: {seg_counter}")
    print(f"사용된 segment (압축 후) 개수: {len(used_ids)}")
    print(f"ID 범위: 0 ~ {len(used_ids)-1}")
    # 검증
    print("mapping shape:", mapping_compressed.shape)
    print("unique IDs:", len(used_ids))
    print("예시 (first 5):", mapping_compressed[:5].tolist())

if __name__ == '__main__':
    main()