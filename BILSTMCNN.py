import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ✅ 1. 데이터 로드 및 전처리
root_folder = r"pre_datas_augmented_with_noise"
SEQ_LEN = 1000  # 입력 시퀀스 길이

# 폴더 → 클래스 인덱스 매핑
subfolders = sorted([os.path.join(root_folder, d) for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])
folder_labels = {os.path.basename(folder): idx for idx, folder in enumerate(subfolders)}
print("폴더별 true_label 매핑:", folder_labels)

# 사용할 컬럼
required_columns = ['timestamp', 'bytes', 'direction', 'index']
feature_cols = ['bytes', 'direction']  # timestamp 제외

# 데이터 저장용 리스트
X_list = []
y_list = []

for folder in subfolders:
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"파일 읽기 실패: {csv_file}, 에러: {e}")
            continue

        if not all(col in df.columns for col in required_columns):
            print(f"필요한 컬럼이 누락된 파일: {csv_file}")
            continue

        df = df[required_columns]
        seq = df[feature_cols].values

        # 시퀀스 길이 고정 (패딩 또는 잘라내기)
        if len(seq) >= SEQ_LEN:
            seq = seq[:SEQ_LEN]
        else:
            pad_len = SEQ_LEN - len(seq)
            pad = np.zeros((pad_len, len(feature_cols)))
            seq = np.vstack([seq, pad])

        X_list.append(seq)
        folder_name = os.path.basename(folder)
        y_list.append(folder_labels[folder_name])

# 리스트 → numpy 배열
X = np.array(X_list)
y = np.array(y_list)

print("총 샘플 수:", len(X), "입력 shape:", X.shape)

# 표준화
scaler = StandardScaler()
X_reshaped = X.reshape(-1, X.shape[-1])  # (전체 샘플 * 시퀀스 길이, feature 수)
X_scaled = scaler.fit_transform(X_reshaped).reshape(X.shape)

# ✅ 2. train/test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ 3. PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ✅ 4. BiCNN-LSTM 모델
class BiCNNLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=17, kernel_size=3, num_filters=64, lstm_layers=2):
        super(BiCNNLSTM, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=hidden_dim, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, seq_len, features) → (batch, features, seq_len)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        x = x.permute(0, 2, 1)  # → (batch, seq_len, features)
        _, (hn, _) = self.lstm(x)
        x = torch.cat((hn[-2], hn[-1]), dim=1)

        return self.fc(x)

# ✅ 5. 학습 및 평가

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))  # GPU 이름 출력



model = BiCNNLSTM(input_dim=len(feature_cols), output_dim=len(folder_labels)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# 학습 루프
EPOCHS = 100
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    y_true_train, y_pred_train = [], []

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        y_true_train.extend(y_batch.cpu().numpy())
        y_pred_train.extend(predicted.cpu().numpy())

    train_acc = accuracy_score(y_true_train, y_pred_train)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}")

# ✅ 평가 (Test)
model.eval()
y_pred, y_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)

        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(y_batch.numpy())

accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", conf_matrix)

# ✅ 평가 (Train)
train_y_pred, train_y_true = [], []
with torch.no_grad():
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)

        train_y_pred.extend(predicted.cpu().numpy())
        train_y_true.extend(y_batch.numpy())

train_accuracy = accuracy_score(train_y_true, train_y_pred)
print(f"Train Accuracy: {train_accuracy:.4f}")
