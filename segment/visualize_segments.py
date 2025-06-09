# cluster_by_edge.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_by_edge.py

SUMO .net.xml 파일에서 edge→lane 매핑을 추출하여,
lanes.txt 순서대로 lane→segment(edge) ID 배열을 생성합니다.

Usage:
  python cluster_by_edge.py \
    --net bucheon.net.xml \
    --lanes lanes.txt \
    --output lane_to_segment_id_by_edge.npy \
    [--merge_window N]

옵션:
  --merge_window N : N개의 연속 edge를 하나의 segment로 병합 (기본값: 1)
                     예: N=3 → 인접 3개 edge를 같은 cluster로 묶음

출력:
  - NumPy .npy 파일: 각 lane 인덱스(i)에 대응하는 segment(edge) 인덱스
  - 콘솔 로그:
      * 총 lane 수
      * 병합 전 고유 segment(edge) 수
      * 병합 후 segment 수 (merge_window > 1인 경우)
      * 저장된 매핑 shape 및 고유 ID 수 검증
"""
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--net',     required=True, help='SUMO .net.xml 파일')
    p.add_argument('--lanes',   required=True, help='lanes.txt (순서대로 lane ID)')
    p.add_argument('--mapping', required=True, help='lane_to_segment_id.npy')
    return p.parse_args()


def main():
    args = parse_args()

    # 1) lanes, mapping 로드
    lanes   = [l.strip() for l in open(args.lanes) if l.strip()]
    mapping = np.load(args.mapping)
    n_segs  = int(mapping.max()) + 1

    # 2) .net.xml 에서 lane별 좌표(shape) 읽기
    tree = ET.parse(args.net)
    root = tree.getroot()
    lane_shapes = {}
    for lane in root.findall('.//lane'):
        lid = lane.get('id')
        shape_str = lane.get('shape', '')
        if not shape_str:
            continue
        pts = np.array([tuple(map(float, xy.split(','))) for xy in shape_str.split()])
        lane_shapes[lid] = pts

    # 3) 세그먼트별로 lane 모아서 centroid 계산
    seg2lines = {sid: [] for sid in range(n_segs)}
    for idx, lid in enumerate(lanes):
        sid = int(mapping[idx])
        pts = lane_shapes.get(lid)
        if sid < 0 or pts is None:
            continue
        seg2lines[sid].append(pts)

    # 4) 그리기: 연속적인 컬러맵과 라벨 개선
    cmap = get_cmap('viridis', n_segs)
    fig, ax = plt.subplots(figsize=(12,12))
    legend_patches = []
    for sid, lines in seg2lines.items():
        if not lines:
            continue
        color = cmap(sid)
        for pts in lines:
            ax.plot(pts[:,0], pts[:,1], color=color, linewidth=1.0, alpha=0.6)
        # 세그먼트 중심: 경계 박스 중심으로 계산
        all_pts = np.vstack(lines)
        min_xy = all_pts.min(axis=0)
        max_xy = all_pts.max(axis=0)
        centroid = (min_xy + max_xy) / 2
        ax.text(
            centroid[0], centroid[1], str(sid),
            fontsize=7, fontweight='bold', ha='center', va='center',
            color='white',
            bbox=dict(facecolor=color, edgecolor='black', boxstyle='round', alpha=0.8)
        )
        if sid < 10:  # 주요 10개 세그먼트만 범례에 표시
            legend_patches.append(mpatches.Patch(color=color, label=f'SID {sid}'))

    ax.set_aspect('equal', 'box')
    ax.axis('off')
    ax.set_title(f'{n_segs} Segments Visualization', fontsize=16)

    # 5) 범례 및 컬러바
    if legend_patches:
        ax.legend(handles=legend_patches, title='Top 10 Segments', loc='upper right', fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_segs-1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Segment ID', fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
    main()
