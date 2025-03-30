import os
import pandas as pd
import random

# 사이트 목록
# sites = [ "adobe", "apple", "archive", "chatgpt", "data", "data_europa_eu", "data_globalchange", "data_gov", "data_who_int", 
        #  "data_worldbank", "dzen", "flickr", "github", "kaggle", "m_soundcloud", "mail", "medium", "news_ycombinator", "open_canada", "telegram", "temu", "wordpress"]
sites = [ "open_canada"]

# 폴더 경로 설정
input_folder_root = r"C:\Users\chomdee\Desktop\WF\pre_datas_augmented"
output_folder_root = r"C:\Users\chomdee\Desktop\WF\pre_datas_augmented_with_front_and_back_noise"
noise_folder_root = r"C:\Users\chomdee\Desktop\WF\raw"

# 유틸 함수: 노이즈 시퀀스 생성
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

# 유틸 함수: 패킷 재분배
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

# ---- 전체 사이트 반복 처리 ----
for site in sites:
    input_folder = os.path.join(input_folder_root, site)
    output_folder = os.path.join(output_folder_root, site)
    noise_folder = os.path.join(noise_folder_root, site)

    os.makedirs(output_folder, exist_ok=True)

    # 노이즈용 파일 리스트 확보
    noise_candidates = [os.path.join(noise_folder, f) for f in os.listdir(noise_folder) if f.endswith(".csv")]
    if not noise_candidates:
        print(f"[SKIP] No noise files in {noise_folder}")
        continue

    input_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    for file_name in input_files:
        input_path = os.path.join(input_folder, file_name)
        df = pd.read_csv(input_path)

        # 랜덤 노이즈 파일 선택
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

        if not noises:
            print(f"[SKIP] No noise generated from {noise_path}")
            continue

        # 노이즈 추가
        while len(df) <= 5000:
            byte, direction = random.choice(noises)
            new_row = {
                'timestamp': 0,
                'bytes': byte,
                'direction': direction,
            }

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            

        # index 컬럼 추가 및 저장
        df['index'] = range(len(df))
        name, ext = os.path.splitext(file_name)
        output_name = name + "_noise" + ext
        output_path = os.path.join(output_folder, output_name)
        df.to_csv(output_path, index=False)
        print(f"[✔] Saved: {output_path}")
