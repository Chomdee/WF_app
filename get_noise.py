import os
import pandas as pd
import random

# ---- 설정 ----
INPUT_FOLDER_PATH = r"C:\Users\chomdee\Desktop\WF\pre_tor_100_ten_packet_seq_augmentation\dropbox"
OUTPUT_FOLDER_PATH = r"C:\Users\chomdee\Desktop\WF\pre_tor_plus_noise\dropbox"
NOISE_FOLDER_PATH = r"C:\Users\chomdee\Desktop\WF\raw\dropbox"

os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)

# ---- 유틸 함수: 노이즈 추출 ----
def noise_sequence(df):
    noises = set()
    for _, row in df.iterrows():
        if row['uploadpackets_diff'] > 0:
            q = row['uploadbytes_diff'] // row['uploadpackets_diff']
            r = row['uploadbytes_diff'] % row['uploadpackets_diff']
            noises.add((q, 1))
            noises.add((q + r, 1))
        if row['downloadpackets_diff'] > 0:
            q = row['downloadbytes_diff'] // row['downloadpackets_diff']
            r = row['downloadbytes_diff'] % row['downloadpackets_diff']
            noises.add((q, 0))
            noises.add((q + r, 0))
    return list(noises)

# ---- 유틸 함수: resample 패킷 ----
def resample_packets(df):
    for direction in ["upload", "download"]:
        p_col = direction + "packets_diff"
        b_col = direction + "bytes_diff"
        for i in range(len(df)):
            p = df.at[i, p_col]
            b = df.at[i, b_col]
            if p == 0 and b != 0:
                val = b
                df.at[i, b_col] = 0
                for j in range(i-1, -1, -1):
                    if df.at[j, p_col] != 0:
                        df.at[j, b_col] += val
                        break
            elif p != 0 and b == 0:
                val = p
                df.at[i, p_col] = 0
                for j in range(i+1, len(df)):
                    if df.at[j, b_col] != 0:
                        df.at[j, p_col] += val
                        break
    return df

# ---- 노이즈용 파일 리스트 확보 ----
noise_candidates = [os.path.join(NOISE_FOLDER_PATH, f) for f in os.listdir(NOISE_FOLDER_PATH) if f.endswith(".csv")]
if not noise_candidates:
    raise FileNotFoundError("노이즈용 CSV 파일이 없습니다.")

# ---- 입력 파일 순회 처리 ----
input_files = [f for f in os.listdir(INPUT_FOLDER_PATH) if f.endswith(".csv")]

for file_name in input_files:
    input_path = os.path.join(INPUT_FOLDER_PATH, file_name)
    df = pd.read_csv(input_path)

    # 랜덤 noise 파일 선택
    noise_path = random.choice(noise_candidates)
    noise_df = pd.read_csv(noise_path)
    noise_df = noise_df[noise_df["flag"] == 0].reset_index(drop=True)

    num_cols = ['uploadbytes', 'downloadbytes', 'uploadpackets', 'downloadpackets']
    for col in num_cols:
        diff_col = col + "_diff"
        noise_df[diff_col] = (noise_df[col] - noise_df[col].shift(1)).fillna(0).clip(lower=0).round().astype(int)

    noise_df = resample_packets(noise_df)
    noise_df = noise_df[(noise_df["uploadpackets_diff"] != 0) | (noise_df["downloadpackets_diff"] != 0)]

    noises = noise_sequence(noise_df)

    # 노이즈 추가
    while len(df) <= 2500:
        byte, direction = random.choice(noises)
        last_ts = df['timestamp'].iloc[-1] if not df.empty else 0
        new_row = {
            'timestamp':0,
            'bytes': byte,
            'direction': direction,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # 저장
    name, ext = os.path.splitext(file_name)
    output_name = name + "_noise" + ext
    output_path = os.path.join(OUTPUT_FOLDER_PATH, output_name)

    df['index'] = range(len(df))
    df.to_csv(output_path, index=False)
    print(f"✔ Saved: {output_path}")
