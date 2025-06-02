#!/usr/bin/env python3
# inspect_adjacency.py
# -*- coding: utf-8 -*-

"""
3_tensor/adjacency의 인접행렬 A.npy와 lanes.txt를 검사 및 시각화합니다.
- 노드 수, 엣지 수 출력
- lanes_list.txt, lane_pairs.txt 생성
- NetworkX로 그래프 연결 상태 플롯
- SUMO net XML에서 위치 정보 추출하여 노드 위치 지정
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from lxml import etree
import sys

def inspect_adjacency(net_xml_path: Path, adj_path: Path, edges_path: Path, out_dir: Path):
    # 0) net_xml_path가 None이면 위치 정보 없이 진행
    pos_map = {}
    if net_xml_path is not None and net_xml_path.exists():
        # 1) SUMO net XML 파싱: 각 lane의 위치 정보를 추출하여 pos_map에 저장
        # 주의: :internal edge 제외, 첫 lane의 shape attribute에서 중간 좌표 계산
        print(f"▶ SUMO net XML 파싱 중: {net_xml_path}")
        tree = etree.parse(str(net_xml_path))
        root = tree.getroot()
        for edge in root.findall("edge"):
            eid = edge.get("id")
            if eid is None or ":internal" in eid:
                continue
            for lane in edge.findall("lane"):
                lid = lane.get("id")
                if lid is None:
                    continue
                shape = lane.get("shape")
                if not shape:
                    continue
                points = shape.split()
                xs = []
                ys = []
                for pt in points:
                    try:
                        x_str, y_str = pt.split(",")
                        xs.append(float(x_str))
                        ys.append(float(y_str))
                    except Exception:
                        continue
                if xs and ys:
                    mean_x = sum(xs) / len(xs)
                    mean_y = sum(ys) / len(ys)
                    pos_map[lid] = (mean_x, mean_y)
        print(f"  • 위치 정보 추출 완료: {len(pos_map)}개 lane 위치 확보")
    else:
        print("▶ net_xml_path가 없거나 존재하지 않아 위치 정보 추출을 건너뜁니다.")
        net_xml_path = None  # 명확히 None 처리

    # 2) 인접행렬 A 및 lanes 리스트 로드
    A = np.load(adj_path)
    lanes = [line.strip() for line in edges_path.read_text(encoding='utf-8').splitlines()]

    # 3) 노드/엣지 수 계산 및 출력
    N = A.shape[0]
    total_links = int(A.sum())
    non_self_links = total_links - N  # self-loop 제외
    print(f"▶ 노드 수 N: {N}")
    print(f"▶ 엣지 수 (self-loop 제외): {non_self_links}")

    # 4) NetworkX 그래프 생성 (방향성)
    # Replace automatic creation from adjacency matrix with manual node/edge addition
    G = nx.DiGraph()
    # Add nodes by string ID
    for lid in lanes:
        G.add_node(lid)
    # Add edges using adjacency matrix with weights
    for i in range(N):
        for j in range(N):
            if A[i, j] > 0:
                G.add_edge(lanes[i], lanes[j], weight=A[i, j])

    # 5) 연결 컴포넌트 수 계산 및 출력 (약한 연결 요소)
    num_weakly_cc = nx.number_weakly_connected_components(G)
    print(f"▶ 약한 연결 컴포넌트 수: {num_weakly_cc}")

    # 6) 고립 노드(자기 자신만 연결된 노드) 탐색 및 출력
    isolated_nodes = []
    for i in range(N):
        # 자기 자신으로 향하는 self-loop만 있고 다른 연결이 없는 경우
        outgoing = A[i, :].sum() - A[i, i]
        incoming = A[:, i].sum() - A[i, i]
        if outgoing == 0 and incoming == 0:
            isolated_nodes.append(lanes[i])
    print(f"▶ 고립 노드 수: {len(isolated_nodes)}")
    if isolated_nodes:
        sample_isolated = isolated_nodes[:5]
        print(f"  • 고립 노드 샘플: {sample_isolated}")

    # 7) lanes.txt와 net XML 간 ID 차이 확인
    lanes_set = set(lanes)
    pos_keys_set = set(pos_map.keys())
    missing_in_xml = lanes_set - pos_keys_set
    missing_in_edges = pos_keys_set - lanes_set
    print(f"▶ lanes.txt에 있으나 net XML에 없는 lane ID 수: {len(missing_in_xml)}")
    if missing_in_xml:
        sample_missing_xml = list(missing_in_xml)[:5]
        print(f"  • 샘플: {sample_missing_xml}")
    print(f"▶ net XML에 있으나 lanes.txt에 없는 lane ID 수: {len(missing_in_edges)}")
    if missing_in_edges:
        sample_missing_edges = list(missing_in_edges)[:5]
        print(f"  • 샘플: {sample_missing_edges}")

    # 8) lanes 리스트 저장
    lanes_out = out_dir / "lanes_list.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(lanes_out, 'w', encoding='utf-8') as f:
        for nid in lanes:
            f.write(nid + "\n")
    print(f"  • lanes 리스트 저장: {lanes_out}")

    # 9) lane 페어 리스트 저장 (self-loop 제외)
    lane_pairs_out = out_dir / "lane_pairs.txt"
    with open(lane_pairs_out, 'w', encoding='utf-8') as f:
        for i in range(N):
            for j in range(N):
                if A[i, j] and i != j:
                    f.write(f"{lanes[i]}\t{lanes[j]}\n")
    print(f"  • lane 페어 리스트 저장: {lane_pairs_out}")

    # 10) 그래프 시각화
    print("▶ 그래프 시각화 중… (노드 크기 작게, 엣지 가늘게 표시)")
    plt.figure(figsize=(8, 8))

    # 노드 위치 지정: pos_map에 없는 노드는 spring_layout으로 보완
    pos = {}
    missing_pos_nodes = []
    for lid in lanes:
        if lid in pos_map:
            pos[lid] = pos_map[lid]
        else:
            missing_pos_nodes.append(lid)
    if missing_pos_nodes:
        # spring_layout을 임시 그래프에서 계산
        subgraph = G.subgraph(missing_pos_nodes)
        spring_pos = nx.spring_layout(subgraph, seed=42)
        pos.update(spring_pos)

    # 노드 색상: 고립 노드 빨강, 나머지 파랑
    node_colors = []
    isolated_set = set(isolated_nodes)
    for lid in lanes:
        if lid in isolated_set:
            node_colors.append('red')
        else:
            node_colors.append('blue')

    # 노드 그리기
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=4, width=0.5)

    # 범례 추가
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label='Isolated Nodes')
    blue_patch = mpatches.Patch(color='blue', label='Connected Nodes')
    plt.legend(handles=[red_patch, blue_patch], loc='best')

    plt.title(f"Lane‐level Adjacency Graph (N={N}, E={non_self_links})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 명령행 인자 처리: 
    # 1) net_xml_path adj_path edges_path  (3개 인자)
    # 2) adj_path edges_path                (2개 인자, 위치 정보 없이 시각화)
    args = sys.argv[1:]
    if len(args) == 3:
        net_xml_path = Path(args[0])
        adj_path = Path(args[1])
        edges_path = Path(args[2])
    elif len(args) == 2:
        net_xml_path = None
        adj_path = Path(args[0])
        edges_path = Path(args[1])
    else:
        print("Usage: python inspect_adjacency.py [net_xml] adj_path edges_path")
        exit(1)

    base = adj_path.parent
    if not adj_path.exists() or not edges_path.exists():
        print("❌ 파일을 찾을 수 없습니다:", adj_path, edges_path)
        exit(1)
    if net_xml_path is not None and not net_xml_path.exists():
        print("❌ net XML 파일을 찾을 수 없습니다:", net_xml_path)
        exit(1)

    inspect_adjacency(net_xml_path, adj_path, edges_path, base)