import os
import glob
import pandas as pd
import numpy as np

def resample_packets(df):
    for direction in ["upload", "download"]:
        packet_col = direction + "packets_diff"
        bytes_col = direction + "bytes_diff"

        packet_col_idx = df.columns.get_loc(packet_col)
        bytes_col_idx = df.columns.get_loc(bytes_col)

        for i in range(len(df)):
            packet_val = df.iloc[i, packet_col_idx]
            bytes_val = df.iloc[i, bytes_col_idx]

            if packet_val == 0 and bytes_val != 0:
                val = bytes_val
                df.iloc[i, bytes_col_idx] = 0

                j = i - 1
                while j >= 0:
                    if df.iloc[j, packet_col_idx] != 0:
                        df.iloc[j, bytes_col_idx] += val
                        break
                    j -= 1
            
            if packet_val != 0 and bytes_val == 0:
                val = packet_val
                df.iloc[i, packet_col_idx] = 0

                j = i + 1

                while j < len(df):
                    if df.iloc[j, bytes_col_idx] != 0:
                        df.iloc[j, packet_col_idx] += val
                        break
                    j += 1


    return df






# 기본 폴더 설정
base_input_folder = r"C:\Users\chomdee\Desktop\WF\datas"
base_output_folder = r"C:\Users\chomdee\Desktop\WF\pre_datas"
input_pattern = os.path.join(base_input_folder, "**", "*.csv")
os.makedirs(base_output_folder, exist_ok=True)

MAX_DIFF_THRESHOLD = None  # 예: 10000 (필요 시 상한값 지정)

def process_file(file_path):
    # CSV 파일 읽기 및 timestamp를 datetime으로 변환
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 필요한 숫자형 컬럼 채우기
    num_cols = ['uploadbytes', 'downloadbytes', 'uploadpackets', 'downloadpackets']
    df[num_cols] = df[num_cols].fillna(0)
    
    # flag 컬럼이 있다면 forward fill 적용
    if 'flag' in df.columns:
        df['flag'] = df['flag'].fillna(method='ffill')
    
    # 모든 결측값 0으로 채우기
    df = df.fillna(0)
    
    # 만약 연속된 row에서 num_cols 값이 모두 0이면 이전 row의 값 복사
    for i in range(1, len(df)):
        if (df.loc[i, num_cols] == 0).all():
            df.loc[i, num_cols] = df.loc[i-1, num_cols]
    
    # 원본 index 보존 (transition 처리를 위해)
    df['orig_index'] = df.index
    
    # transition 컬럼 추가: flag가 0에서 1로 전환되는 row 식별
    df['transition'] = (df['flag'] == 1) & (df['flag'].shift(1) == 0)
    
    # 인덱스 재설정 (원본 index와 transition 컬럼은 보존됨)
    df.reset_index(drop=True, inplace=True)
    
    # flag==1인 row만 추출하여 df_flag1 생성
    df_flag1 = df[df['flag'] == 1].copy().reset_index(drop=True)
    
    # 각 숫자형 컬럼에 대해 _adj 컬럼 생성 (원본 값을 그대로 사용)
    for col in num_cols:
        df_flag1[col + "_adj"] = df_flag1[col]
    
    # 각 컬럼별로 diff 계산: diff = 현재 row의 _adj 값 - 바로 이전 row의 _adj 값
    for col in num_cols:
        diff_col = col + "_diff"
        df_flag1[diff_col] = df_flag1[col + "_adj"] - df_flag1[col + "_adj"].shift(1)
        df_flag1[diff_col] = df_flag1[diff_col].fillna(0)
        df_flag1[diff_col] = df_flag1[diff_col].clip(lower=0)
        if MAX_DIFF_THRESHOLD is not None:
            df_flag1[diff_col] = df_flag1[diff_col].clip(upper=MAX_DIFF_THRESHOLD)
        # 모든 diff 값을 정수형으로 변환
        df_flag1[diff_col] = df_flag1[diff_col].round().astype(int)
    
    # 전환(row)인 경우에 대해 처리:
    # 만약 해당 row의 downloadbytes와 downloadpackets가 이전 row(원본에서 flag==0)와 동일하면,
    # 해당 row의 download 관련 diff 값을 0으로 설정
    for i, row in df_flag1.iterrows():
        if row['transition']:
            orig_idx = row['orig_index']
            prev_row = df[df['orig_index'] == orig_idx - 1]
            if not prev_row.empty:
                prev_row = prev_row.iloc[0]
                if (prev_row['downloadbytes'] == row['downloadbytes']) and (prev_row['downloadpackets'] == row['downloadpackets']):
                    df_flag1.at[i, 'downloadbytes_diff'] = 0
                    df_flag1.at[i, 'downloadpackets_diff'] = 0
    
    # transition와 orig_index 컬럼 삭제
    df_flag1.drop(columns=['transition', 'orig_index'], inplace=True)
    
    # 최종 전처리 후, row 수가 1500개가 아니면 마지막 row를 복사해서 1500개로 맞춤
    target_rows = 1500
    current_rows = len(df_flag1)
    if current_rows < target_rows:
        n_missing = target_rows - current_rows
        last_row = df_flag1.iloc[[-1]].copy()
        pad_df = pd.concat([last_row] * n_missing, ignore_index=True)
        df_flag1 = pd.concat([df_flag1, pad_df], ignore_index=True)
    elif current_rows > target_rows:
        df_flag1 = df_flag1.iloc[:target_rows].copy()
    
    # timestamp를 0초부터 30초까지, 총 1500 row로 재설정 (0.02초 간격) 및 소수점 2자리 반올림
    df_flag1['timestamp'] = np.round(np.linspace(0, 30, num=target_rows, endpoint=False), 2)

    df_flag1 = resample_packets(df_flag1)
    
    return df_flag1

# 모든 CSV 파일 처리
csv_files = glob.glob(input_pattern, recursive=True)

for file_path in csv_files:
    try:
        processed_df = process_file(file_path)
        relative_path = os.path.relpath(file_path, base_input_folder)
        output_dir = os.path.join(base_output_folder, os.path.dirname(relative_path))
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{name}_preprocessed{ext}")
        processed_df.to_csv(output_path, index=False)
        print(f"Processed and saved: {output_path}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
