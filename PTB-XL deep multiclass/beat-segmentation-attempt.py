import os
import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from scipy.signal import butter, lfilter, find_peaks
from tqdm import tqdm
from sklearn.model_selection import KFold
import ast
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Preprocessing Functions
# -------------------------
def butter_bandpass_filter(signal, lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal)
    return y

def pan_tompkins_detector(signal, fs):
    signal = butter_bandpass_filter(signal, 0.5, 45, fs, order=1)
    diff = np.diff(signal)
    squared = diff ** 2
    window_size = int(0.150 * fs)
    mwa = np.convolve(squared, np.ones(window_size) / window_size, mode='same')
    peaks, _ = find_peaks(mwa, distance=fs*0.2, height=np.mean(mwa))
    return peaks

# Update the segmentation function to allow variable beat lengths
def segment_beats(signal, r_peaks, fs=100, Lf=25, Lk=55):
    beat_length = Lf + Lk
    beats = []
    for r in r_peaks:
        start = max(r - Lf, 0)
        end = min(r + Lk, len(signal))
        beat = signal[start:end]
        if len(beat) < beat_length:
            pad = np.zeros(beat_length)
            pad[:len(beat)] = beat
            beat = pad
        beats.append(beat)
    return np.array(beats)

# -------------------------
# Load and Process PTB-XL
# -------------------------
def load_ptbxl_data(base_path):
    df = pd.read_csv(os.path.join(base_path, 'ptbxl_database.csv'))
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

    agg_df = pd.read_csv(os.path.join(base_path, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))  # unique classes

    df['diagnostic_superclass'] = df['scp_codes'].apply(aggregate_diagnostic)
    df = df[df['diagnostic_superclass'].map(len) == 1]
    df['label'] = df['diagnostic_superclass'].apply(lambda x: x[0])

    label_encoder = LabelEncoder()
    df['label_idx'] = label_encoder.fit_transform(df['label'])
    return df[['filename_lr', 'label_idx']], label_encoder


# -------------------------
# Dataset
# -------------------------
class PTBXLBeatDataset(Dataset):
    def __init__(self, df, base_path, max_beats=20, Lf=25, Lk=55):
        self.df = df
        self.base_path = base_path
        self.fs = 100
        self.max_beats = max_beats
        self.Lf = Lf
        self.Lk = Lk

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.base_path, row['filename_lr'])
        signal = load_record(path)
        r_peaks = pan_tompkins_detector(signal, self.fs)
        beats = segment_beats(signal, r_peaks, self.fs, self.Lf, self.Lk)
        if len(beats) > self.max_beats:
            beats = beats[:self.max_beats]
        else:
            pad = np.zeros((self.max_beats, beats.shape[1]))
            pad[:len(beats)] = beats
            beats = pad
        beats = torch.tensor(beats).float().unsqueeze(1)
        label = torch.tensor(row['label_idx']).long()
        return beats, label


# -------------------------
# Model
# -------------------------
class BeatFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.gru = nn.GRU(256, 128, batch_first=True, bidirectional=True)

    def forward(self, x):
        B, N, C, L = x.shape
        x = x.view(B * N, C, L)
        x = self.cnn(x)
        x = x.view(B * N, x.shape[1], -1).permute(0, 2, 1)
        _, h = self.gru(x)
        h = torch.cat([h[0], h[1]], dim=1)
        h = h.view(B, N, -1)
        return h

# Add visualization of attention weights
def visualize_attention_weights(signal, weights, title="Attention Weights Visualization"):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal, label="ECG Signal")
    plt.title("ECG Signal")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.bar(range(len(weights)), weights, label="Attention Weights")
    plt.title("Attention Weights")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Update the AttentionFusion class to return weights for visualization
class AttentionFusion(nn.Module):
    def __init__(self, input_dim=256, attn_dim=128):
        super().__init__()
        self.fc = nn.Linear(input_dim, attn_dim)
        self.attn = nn.Linear(attn_dim, 1)

    def forward(self, x):
        B, N, D = x.shape
        x = x.view(-1, D)
        x = self.fc(x)
        attention_weights = self.attn(x)
        weights = F.softmax(attention_weights, dim=0).view(B, N)
        weighted_sum = (x.view(B, N, -1) * weights.unsqueeze(-1)).sum(dim=1)
        return weighted_sum, weights

# Update BLFNet to handle attention weights
class BLFNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.extractor = BeatFeatureExtractor()
        self.attn = AttentionFusion(256)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.extractor(x)
        fused, weights = self.attn(features)
        out = self.classifier(fused)
        return out, weights


# -------------------------
# Training and Evaluation
# -------------------------
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)

        # One-hot encode the labels to match the output shape
        y_one_hot = F.one_hot(y, num_classes=model.classifier.out_features).float()

        optimizer.zero_grad()
        out, _ = model(x)
        loss = criterion(out, y_one_hot)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc

# Update evaluation to include attention visualization
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    all_probs = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_one_hot = F.one_hot(y, num_classes=model.classifier.out_features).float()
            out, weights = model(x)
            loss = criterion(out, y_one_hot)
            total_loss += loss.item()
            probs = torch.sigmoid(out).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_one_hot.cpu().numpy())

            # Visualize attention weights for the first batch
            if len(all_preds) == 0:
                visualize_attention_weights(x[0].cpu().numpy().flatten(), weights[0].cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs, average='macro')
    return total_loss / len(loader), auc


# -------------------------
# Main
# -------------------------
data_path = "/kaggle/input/ptb-xl-dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
df, label_encoder = load_ptbxl_data(data_path)

# Split the dataset into training/validation and test sets
# Use one part of the dataset as the test set, and the remaining 9 parts for cross-validation

def split_dataset(df, test_split=0.1):
    test_size = int(len(df) * test_split)
    test_df = df.iloc[:test_size]
    train_val_df = df.iloc[test_size:]
    return train_val_df, test_df

# Update the main script to include a test set
train_val_df, test_df = split_dataset(df)

# Define the test dataset and loader
test_dataset = PTBXLBeatDataset(test_df, data_path)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = BLFNet(num_classes=5).to(device)

# Replace CrossEntropyLoss with BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()

# Add L2 regularization to the optimizer
l2_lambda = 1e-4
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=l2_lambda)

# Perform 9-fold cross-validation on the training/validation set
kf = KFold(n_splits=9)

# Update the training loop to include proper evaluation and metrics
for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_df)):
    print(f"Training Fold {fold + 1}")
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]
    train_dataset = PTBXLBeatDataset(train_df, data_path)
    val_dataset = PTBXLBeatDataset(val_df, data_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    for epoch in range(22):  # 22 epochs
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_auc = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

# After cross-validation, evaluate the final model on the test set
evaluate_final_model(model, test_loader, criterion)
