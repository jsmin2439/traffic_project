#!/usr/bin/env python3
# build_adjacency.py (차로 단위 버전)
# -*- coding: utf-8 -*-

"""
1_lake/net/bucheon.net.xml 에서
  • lane 단위 인접행렬 A (N×N, self-loop 포함)
  • lane 리스트 lanes (차로 ID, 길이 N)
를 추출하여 numpy 및 sparse 파일로 저장하고, 간단히 검증합니다.

[주요 수정 포인트]
- parse_adjacency: edge → lane 단위로 노드 수집
- parse_adjacency: connection 순회 시 “edge의 모든 lane ↔ edge의 모든 lane” 쌍으로 가중치 누적
- main: 실제 고립된 lane(연결 없는 lane)을 필터링하여 제거
- main: 출력 파일명을 lanes.txt 로 변경
"""

import sys
import platform
import warnings
from lxml import etree            # XML 파싱 (lxml 사용)
import numpy as np                # 수치 연산
from pathlib import Path
import scipy.sparse as sp         # 희소 행렬 저장

def parse_adjacency(net_xml_path: str, directed=True):
    """
    lane 단위로 인접행렬을 생성합니다.

    Args:
      net_xml_path: SUMO 네트워크 XML 파일 경로 (예: '1_lake/net/bucheon.net.xml')
      directed: True면 방향 그래프, False면 무향 그래프

    Returns:
      - A: np.ndarray, shape=(N, N) 인접행렬 (self-loop 포함된 상태)
      - lanes: List[str], 길이 N (lane ID 리스트, 원본 순서 유지)
    """

    # 1) XML 파싱
    tree = etree.parse(net_xml_path)

    # 2) lane ID 리스트 수집 (internal/junction 제외)
    lanes = []
    for edge in tree.findall('.//edge'):
        eid  = edge.get('id')
        func = (edge.get('function') or '').lower()
        # ':'로 시작하거나 'internal' 기능(edge)인 경우 스킵
        if eid.startswith(':') or func == 'internal':
            continue
        # edge 요소 아래의 모든 lane 태그에서 id 수집
        for lane in edge.findall('lane'):
            lanes.append(lane.get('id'))

    N = len(lanes)
    lane_index = {lid: idx for idx, lid in enumerate(lanes)}

    # 3) lane-level 연결 가중치 누적
    weight_dict = {}
    for conn in tree.findall('.//connection'):
        fe = conn.get('from')  # from-edge ID
        te = conn.get('to')    # to-edge ID
        # internal/junction 연결 제외
        if fe.startswith(':') or te.startswith(':'):
            continue

        # from-edge에 속한 모든 lane ID 목록
        from_edge_elem = tree.find(f".//edge[@id='{fe}']")
        if from_edge_elem is None:
            continue
        from_lanes = [l.get('id') for l in from_edge_elem.findall('lane')]

        # to-edge에 속한 모든 lane ID 목록
        to_edge_elem = tree.find(f".//edge[@id='{te}']")
        if to_edge_elem is None:
            continue
        to_lanes = [l.get('id') for l in to_edge_elem.findall('lane')]

        # “edge의 모든 lane ↔ edge의 모든 lane” 쌍으로 가중치 누적
        for fl in from_lanes:
            for tl in to_lanes:
                i = lane_index.get(fl)
                j = lane_index.get(tl)
                if i is None or j is None:
                    continue
                weight_dict[(i, j)] = weight_dict.get((i, j), 0) + 1

    # 4) 인접행렬 A 생성
    A = np.zeros((N, N), dtype=np.float32)
    for (i, j), w in weight_dict.items():
        A[i, j] = w

    # 5) directed=False인 경우 무향 처리 (A = max(A, A^T))
    if not directed:
        A = np.maximum(A, A.T)

    # 6) self-loop 추가 (만약 대각 원소가 0인 경우)
    for idx in range(N):
        if A[idx, idx] == 0:
            A[idx, idx] = 1.0

    return A, lanes

def get_normalized_adjacency(A: np.ndarray):
    """
    정규화된 인접행렬을 반환합니다:
      Â = A + I (self-loop 포함 가정)
      D = diag(degree(Â))
      A_norm = D^{-1/2} Â D^{-1/2}

    Args:
      A: np.ndarray, shape=(N, N), self-loop 포함 여부 상관 없음. 

    Returns:
      A_norm: np.ndarray, shape=(N, N)
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input A must be numpy ndarray.")

    # degree 벡터
    deg = A.sum(axis=1)
    # D^{-1/2} 계산 (0인 경우 0으로 처리)
    with np.errstate(divide='ignore'):
        d_inv_sqrt = 1.0 / np.sqrt(deg)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    # D^{-1/2} 행렬
    D_inv_sqrt = np.diag(d_inv_sqrt)
    # 정규화된 adjacency
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm

def main():
    # 0) 버전 정보 출력
    print(f"Python version: {platform.python_version()}")
    try:
        import lxml
        print(f"lxml version: {lxml.__version__}")
    except ImportError:
        warnings.warn("lxml not installed. XML 파싱 시 오류가 발생할 수 있습니다.")

    # 1) 인수 개수 확인
    if len(sys.argv) != 2:
        print("Usage: python build_adjacency.py path/to/net.xml")
        sys.exit(1)

    net_xml = sys.argv[1]
    out_dir = Path("3_tensor/adjacency")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) lane 단위 파싱 수행
    print("▶ Parsing adjacency (lane-level)...")
    A, lanes = parse_adjacency(net_xml, directed=True)
    print(f"  • 수집된 raw lane 수: {len(lanes)}")

    # 3) 실제 연결된 lane만 필터링
    print("▶ 실제 고립 lane 필터링 중...")
    tree_root = etree.parse(net_xml).getroot()
    connected = set()
    # connection 태그 순회
    for conn in tree_root.findall('.//connection'):
        fe, te = conn.get('from'), conn.get('to')
        if fe.startswith(':') or te.startswith(':'):
            continue

        # from-edge의 lane들
        from_edge_elem = tree_root.find(f".//edge[@id='{fe}']")
        if from_edge_elem is not None:
            for lane_elem in from_edge_elem.findall('lane'):
                lid = lane_elem.get('id')
                if lid in lanes:
                    connected.add(lid)

        # to-edge의 lane들
        to_edge_elem = tree_root.find(f".//edge[@id='{te}']")
        if to_edge_elem is not None:
            for lane_elem in to_edge_elem.findall('lane'):
                lid = lane_elem.get('id')
                if lid in lanes:
                    connected.add(lid)

    # 실제 연결된 lane만 keep
    keep_indices = [i for i, lid in enumerate(lanes) if lid in connected]
    removed_lanes = [lid for i, lid in enumerate(lanes) if lid not in connected]
    if removed_lanes:
        print(f"  • 제거된 실제 고립 lane 수: {len(removed_lanes)}")
    # 행렬과 리스트를 인덱스 기준으로 필터링
    A = A[np.ix_(keep_indices, keep_indices)]
    lanes = [lanes[i] for i in keep_indices]
    print(f"  • 필터 후 남은 lane 수: {len(lanes)}")

    # 4) 정규화된 인접행렬 계산
    print("▶ 인접행렬 정규화 중...")
    A_norm = get_normalized_adjacency(A)

    # 5) 결과 저장
    print("▶ 결과 저장 중...")
    np.save(out_dir / "A.npy", A_norm)                  # NumPy 바이너리
    A_sparse = sp.csr_matrix(A_norm)                    
    sp.save_npz(out_dir / "A.npz", A_sparse)             # Sparse 형식
    # lane ID를 파일로 저장 (lanes.txt)
    with open(out_dir / "lanes.txt", "w", encoding="utf-8") as f:
        for lid in lanes:
            f.write(f"{lid}\n")

    # 6) 최종 검증 정보 출력
    N_final = A_norm.shape[0]
    nonzero_links = int(np.sum(A_norm > 0)) - N_final  # self-loop 제외
    print(f"[완료] 저장된 파일: {out_dir}/A.npy, A.npz, lanes.txt")
    print(f"  • A_norm shape: {A_norm.shape}")
    print(f"  • self-loop 제외 nonzero link 수: {nonzero_links}")
    print(f"  • Sample lanes: {lanes[:5]}")

if __name__ == "__main__":
    main()