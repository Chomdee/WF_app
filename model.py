import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from imblearn.over_sampling import SMOTE
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_MAPPING = {}

def get_label_from_filepath(file_path):
    return os.path.basename(os.path.dirname(file_path)).lower()

# ✅ 수정된 feature 로딩 함수 (timestamp 제외)
def load_and_aggregate(file_path):
    df = pd.read_csv(file_path)
    if len(df) < 2500:
        raise ValueError(f"File {file_path} has less than 2500 rows: {len(df)}")
    
    df = df.iloc[:2500].copy()

    # timestamp를 제외하고 사용할 컬럼만
    feature_cols = ['bytes', 'direction', 'index']
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in file {file_path}: {missing}")

    features = df[feature_cols].values  
    return features.flatten()           

class AggregatedDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.features[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.3):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def stratified_split_per_class(X, y, test_size=0.2, random_state=42):
    unique_classes = np.unique(y)
    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
    for cls in unique_classes:
        idx = np.where(y == cls)[0]
        X_cls = X[idx]
        y_cls = y[idx]
        if len(y_cls) < 2:
            X_train_list.append(X_cls)
            y_train_list.append(y_cls)
        else:
            X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
                X_cls, y_cls, test_size=test_size, random_state=random_state, stratify=None
            )
            X_train_list.append(X_train_cls)
            X_test_list.append(X_test_cls)
            y_train_list.append(y_train_cls)
            y_test_list.append(y_test_cls)
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    if X_test_list:
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
    else:
        X_test, y_test = np.array([]), np.array([])
    return X_train, X_test, y_train, y_test

def load_all_data(preprocessed_dir):
    file_paths = glob.glob(os.path.join(preprocessed_dir, "**", "*_noise.csv"), recursive=True)
    X_list, y_list = [], []
    for file_path in file_paths:
        folder = get_label_from_filepath(file_path)
        if folder not in CLASS_MAPPING:
            CLASS_MAPPING[folder] = len(CLASS_MAPPING)
        label = CLASS_MAPPING[folder]
        try:
            features = load_and_aggregate(file_path)
            X_list.append(features)
            y_list.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return np.array(X_list), np.array(y_list)

def train_nearest_centroid(X_train, y_train):
    centroids = {}
    for cls in np.unique(y_train):
        centroids[cls] = np.mean(X_train[y_train == cls], axis=0)
    return centroids

def predict_nearest_centroid(X, centroids):
    preds = []
    for x in X:
        distances = {cls: np.linalg.norm(x - centroid) for cls, centroid in centroids.items()}
        pred = min(distances, key=distances.get)
        preds.append(pred)
    return np.array(preds)

def main():
    preprocessed_dir = r"C:\Users\chomdee\Desktop\WF\pre_tor_plus_noise"
    print("[INFO] Loading features...")
    X, y = load_all_data(preprocessed_dir)
    print(f"[INFO] Data shape: {X.shape}, Labels: {y.shape}")
    print(f"[INFO] CLASS_MAPPING: {CLASS_MAPPING}")

    X_train, X_test, y_train, y_test = stratified_split_per_class(X, y, test_size=0.2)
    print(f"[INFO] Train: {len(y_train)}, Test: {len(y_test)}")

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_dir = r"C:\Users\chomdee\Desktop\WF\scaler"
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(scaler_dir, "scaler_noise.pkl"))
    print("[INFO] Scaler saved.")

    print(f"[INFO] Before SMOTE: {Counter(y_train)}")
    smote = SMOTE(random_state=42)
    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
    print(f"[INFO] After SMOTE: {Counter(y_train)}")

    input_size = X_train_scaled.shape[1]  # 4500
    hidden_size = 64
    output_size = len(CLASS_MAPPING)
    model = MLPModel(input_size, hidden_size, output_size, dropout=0.3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_loader = DataLoader(AggregatedDataset(X_train_scaled, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(AggregatedDataset(X_test_scaled, y_test), batch_size=32, shuffle=False)

    best_acc = 0.0
    for epoch in range(200):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        avg_loss = running_loss / len(train_loader)

        model.eval()
        preds, true_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy())
                true_labels.extend(y_batch.cpu().numpy())

        acc = accuracy_score(true_labels, preds)
        macro_f1 = f1_score(true_labels, preds, average='macro')
        print(f"Epoch [{epoch+1}/200] Loss: {avg_loss:.4f}, Acc: {acc:.4f}, Macro F1: {macro_f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            model_dir = r"C:\Users\chomdee\Desktop\WF\model"
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model_bytes_direction_index.pth"))
            print(f"[INFO] New best model saved (Acc: {best_acc:.4f})")

    print(f"[FINAL] Best MLP Test Accuracy: {best_acc:.4f}")

    # Zaccord 방식
    centroids = train_nearest_centroid(X_train_scaled, y_train)
    z_preds = predict_nearest_centroid(X_test_scaled, centroids)
    z_acc = accuracy_score(y_test, z_preds)
    z_macro_f1 = f1_score(y_test, z_preds, average='macro')
    print(f"[FINAL] Zaccord Accuracy: {z_acc:.4f}, Macro F1: {z_macro_f1:.4f}")

if __name__ == "__main__":
    main()
