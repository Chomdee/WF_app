# 동일한 시간대에서 upload, download 에 대한 패킷이 여러 개 들어왔을 때,
# 이 패킷들에 대한 sequence를 랜덤하게 셔플하여 데이터를 증강하는 코드


import os
import pandas as pd
import random
from pathlib import Path

def get_packets(df):
    return df[(df["uploadpackets_diff"] != 0) | (df["downloadpackets_diff"] != 0)]

def processing(df):
    all_packets = []

    timestamp_col_idx = df.columns.get_loc("timestamp")
    uploadbytes_col_idx = df.columns.get_loc("uploadbytes_diff")
    uploadpackets_col_idx = df.columns.get_loc("uploadpackets_diff")
    downloadbytes_col_idx = df.columns.get_loc("downloadbytes_diff")
    downloadpackets_col_idx = df.columns.get_loc("downloadpackets_diff")

    for row in range(len(df)):
        timestamp = df.iloc[row, timestamp_col_idx]
        uploadpacket = df.iloc[row, uploadpackets_col_idx]
        uploadbyte = df.iloc[row, uploadbytes_col_idx]
        downloadpacket = df.iloc[row, downloadpackets_col_idx]
        downloadbyte = df.iloc[row, downloadbytes_col_idx]

        packets = []

        if uploadpacket != 0:
            quo = uploadbyte // uploadpacket
            rem = uploadbyte % uploadpacket
            packets.extend([(timestamp, quo, 1)] * uploadpacket if rem == 0 else [(timestamp, quo, 1)] * (uploadpacket - 1) + [(timestamp, quo + rem, 1)])

        if downloadpacket != 0:
            quo = downloadbyte // downloadpacket
            rem = downloadbyte % downloadpacket
            packets.extend([(timestamp, quo, 0)] * downloadpacket if rem == 0 else [(timestamp, quo, 0)] * (downloadpacket - 1) + [(timestamp, quo + rem, 0)])

        random.shuffle(packets)
        all_packets.extend(packets)

    return all_packets

#  입력/출력 폴더
INPUT_DIR = r"C:\Users\chomdee\Desktop\WF\pre_datas"
OUTPUT_DIR = r"C:\Users\chomdee\Desktop\WF\pre_datas_augmented"

for file_path in Path(INPUT_DIR).rglob("*.csv"):
    try:
        df = pd.read_csv(file_path)
        packets_df = get_packets(df)
        relative_path = file_path.relative_to(INPUT_DIR)
        file_stem = file_path.stem  

        for i in range(5):
            packets = processing(packets_df)
            df_out = pd.DataFrame(packets, columns=["timestamp", "bytes", "direction"])
            df_out = df_out.sort_values("timestamp")

            # 저장 경로
            output_subdir = Path(OUTPUT_DIR) / relative_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)

            output_file = output_subdir / f"{file_stem}_augmented_{i+1}.csv"
            df_out.to_csv(output_file, index=False)

        print(f"[DONE] Processed {file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
