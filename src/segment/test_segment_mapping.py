#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_mapping.py

생성된 lane_to_segment_id .npy 파일을 검증하고,
클러스터별 lane 개수 통계를 출력합니다.
"""
import numpy as np

# 1) 매핑 파일 로드
mapping = np.load('lane_to_segment_id_by_edge.npy')
lanes   = open('lanes.txt').read().splitlines()

assert mapping.ndim == 1, "매핑 배열이 일차원이 아닙니다."
assert mapping.shape[0] == len(lanes), "lanes.txt 길이와 매핑 길이 불일치"

# 2) 기본 통계 출력
print(f"Mapping shape: {mapping.shape}")
unique_ids = np.unique(mapping)
unique_ids = unique_ids[unique_ids >= 0]  # -1 제외
print(f"Unique segment count: {unique_ids.size}")
print(f"Min segment ID: {mapping.min()}  Max segment ID: {mapping.max()}")

# 3) 클러스터별 lane 개수 계산
#    np.bincount를 쓰려면 음수가 없어야 하므로 -1 제거
counts = np.bincount(mapping[mapping >= 0])
#   counts[i] = segment i에 할당된 lane 수

print("\nCluster size statistics:")
print(f"  Max lanes in one cluster : {counts.max()}")
print(f"  Min lanes in one cluster : {counts.min()}")
print(f"  Mean lanes per cluster   : {counts.mean():.2f}")

# 4) 클러스터별 개수 분포 히스토그램 (간단한 텍스트 버전)
print("\nCluster size distribution (size: count of clusters):")
from collections import Counter
dist = Counter(counts)
for size, num in sorted(dist.items()):
    print(f"  {size:3d} lanes  → {num:3d} clusters")

# 5) 일부 예시 출력
print("\n첫 5개 매핑 예시:")
for i in range(5):
    print(f"  {lanes[i]} -> {mapping[i]}")