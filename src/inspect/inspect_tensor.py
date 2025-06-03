#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tensor_inspect.py

생성된 input_tensor 및 normalized_tensor(.npy) 파일에 대해
아래 정보를 상세히 추출하여 CSV로 저장하는 스크립트입니다.

• 각 날짜별 input_tensor / normalized_tensor 기본 요약
  - shape, dtype, 전체 요소 수
  - NaN 개수, Inf 개수, 음수 개수, 양수 개수, 0 개수
  - 채널 수

• 각 채널별 통계 (min, max, mean, std, median, q1, q3, nonzero_count, zero_count)

출력 파일:
  - 3_tensor/tensor_summary.csv
  - 3_tensor/tensor_channel_stats.csv
"""

import os
import numpy as np
import pandas as pd

# 1) 검사할 날짜 리스트 및 기본 경로 설정
DATE_LIST = [
    20220810, 20220811, 20220812, 20220813, 20220814, 20220815,
    20220819, 20220820, 20220821, 20220822, 20220823,
    20220906, 20220907, 20220909, 20220910, 20220911,
    20220912, 20220913, 20220914, 20220915, 20220916,
    20220926, 20220928, 20220930, 20221001, 20221002,
    20221026, 20221027, 20221028, 20221030, 20221031
]
BASE_DIR = '3_tensor/'  # 각 날짜별 폴더가 있는 디렉토리
FIELDS = ['input_tensor', 'normalized_tensor']

# 2) 결과 저장용 리스트 초기화
summary_records = []
channel_stats_list = []

# 3) 날짜별/필드별로 텐서 로드 및 통계 계산
for date in DATE_LIST:
    date_str = str(date)
    date_dir = os.path.join(BASE_DIR, date_str)
    for field in FIELDS:
        # .npy 파일 경로
        npy_path = os.path.join(date_dir, f'{field}_{date}.npy')
        if not os.path.exists(npy_path):
            print(f"⚠️ 경고: 파일이 존재하지 않습니다: {npy_path}")
            continue
        
        # 3-1) 텐서 로드
        tensor = np.load(npy_path)  # ex) shape=(288, 1370, 8)
        shape = tensor.shape
        dtype = tensor.dtype
        
        # 3-2) 전체 요소 수 및 NaN/Inf/음수/양수/0 개수
        total_elements = tensor.size
        nan_count = int(np.isnan(tensor).sum())
        inf_count = int(np.isinf(tensor).sum())
        neg_count = int((tensor < 0).sum())
        pos_count = int((tensor > 0).sum())
        zero_count = int((tensor == 0).sum())
        
        # 3-3) 채널 수 및 채널별 통계
        num_channels = shape[2]
        for ch in range(num_channels):
            vals = tensor[:, :, ch].ravel()
            ch_min = float(np.min(vals))
            ch_max = float(np.max(vals))
            ch_mean = float(np.mean(vals))
            ch_std = float(np.std(vals))
            ch_median = float(np.median(vals))
            ch_q1 = float(np.percentile(vals, 25))
            ch_q3 = float(np.percentile(vals, 75))
            ch_nonzero = int(np.count_nonzero(vals))
            ch_zero = int(vals.size - ch_nonzero)
            
            channel_stats_list.append({
                'date': date,
                'field': field,
                'channel': ch,
                'min': ch_min,
                'max': ch_max,
                'mean': ch_mean,
                'std': ch_std,
                'median': ch_median,
                'q1': ch_q1,
                'q3': ch_q3,
                'nonzero_count': ch_nonzero,
                'zero_count': ch_zero
            })
        
        # 3-4) 요약 레코드 추가
        summary_records.append({
            'date': date,
            'field': field,
            'shape': shape,
            'dtype': str(dtype),
            'total_elements': total_elements,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'neg_count': neg_count,
            'pos_count': pos_count,
            'zero_count': zero_count,
            'num_channels': num_channels
        })

# 4) DataFrame으로 정리
summary_df = pd.DataFrame(summary_records)
channel_stats_df = pd.DataFrame(channel_stats_list)

# 5) 출력 결과 예시 (콘솔에 일부만 출력)
print("\n▶ 텐서 전체 요약 (일부 예시):")
print(summary_df.head(6).to_string(index=False))

print("\n▶ 채널별 상세 통계 (일부 예시):")
print(channel_stats_df.head(8).to_string(index=False))

# 6) CSV로 저장
summary_df.to_csv(os.path.join(BASE_DIR, 'tensor_summary.csv'), index=False)
channel_stats_df.to_csv(os.path.join(BASE_DIR, 'tensor_channel_stats.csv'), index=False)

print(f"\n✅ 요약 정보가 '{os.path.join(BASE_DIR, 'tensor_summary.csv')}'에 저장되었습니다.")
print(f"✅ 채널별 통계가 '{os.path.join(BASE_DIR, 'tensor_channel_stats.csv')}'에 저장되었습니다.")