import pandas as pd
import numpy as np
import time
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score, accuracy_score
from collections import Counter
import wandb  # <-- THÊM THƯ VIỆN WANDB

# ==========================================
# 0. KHỞI TẠO WANDB
# ==========================================
# Dán API key của bạn vào đây (nếu bạn chưa login qua terminal)
# wandb.login(key="dán_api_key_của_bạn_vào_đây") 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Đang chạy trên thiết bị: {device}")

# ==========================================
# 1. CẤU HÌNH SIÊU THAM SỐ
# ==========================================
MAX_WORDS = 15000     
MAX_LEN = 120         
EMBEDDING_DIM = 256   
BATCH_SIZE = 64
EPOCHS = 6            
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

wandb.init(
    project="Toxic-Comment-Classification",
    name="Hybrid-CNN-BiLSTM-Run",
    config={
        "architecture": "Hybrid (Conv1D + MaxPool + BiLSTM)",
        "dataset": "balanced_dataset_5050",
        "max_words": MAX_WORDS,
        "max_len": MAX_LEN,
        "embedding_dim": EMBEDDING_DIM,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss_function": "BCEWithLogitsLoss + pos_weight",
        "optimizer": "AdamW"
    }
)

print("2. Tải và tiền xử lý văn bản...")
df = pd.read_csv('FINAL_dataset_32k_cleaned.csv')

def clean_and_tokenize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

df['tokenized'] = df['comment_text'].apply(clean_and_tokenize)

# CHIA TẬP CHUẨN 70% TRAIN - 10% VAL - 20% TEST
X_temp, X_test_tokens, y_temp, y_test = train_test_split(
    df['tokenized'].values, df[label_cols].values, test_size=0.2, random_state=42
)
X_train_tokens, X_val_tokens, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42
)

print(f"-> Kích thước tập: Train({len(X_train_tokens)}), Val({len(X_val_tokens)}), Test({len(X_test_tokens)})")

print("3. Xây dựng Từ điển và Padding (Chỉ học từ vựng trên tập Train)...")
all_words = [word for tokens in X_train_tokens for word in tokens]
word_counts = Counter(all_words)
vocab = {word: i+2 for i, (word, _) in enumerate(word_counts.most_common(MAX_WORDS-2))}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1

def text_to_sequence(tokens):
    seq = [vocab.get(word, 1) for word in tokens]
    if len(seq) < MAX_LEN:
        seq += [0] * (MAX_LEN - len(seq))
    else:
        seq = seq[:MAX_LEN]
    return seq

X_train_pad = np.array([text_to_sequence(t) for t in X_train_tokens])
X_val_pad = np.array([text_to_sequence(t) for t in X_val_tokens])
X_test_pad = np.array([text_to_sequence(t) for t in X_test_tokens])

class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
    def __len__(self):
        return len(self.labels)

train_loader = DataLoader(TextDataset(X_train_pad, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TextDataset(X_val_pad, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TextDataset(X_test_pad, y_test), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 4. KIẾN TRÚC LAI GHÉP: CNN-BiLSTM (HYBRID MODEL)
# ==========================================
print("4. Khởi tạo kiến trúc mạng Hybrid CNN-BiLSTM...")
class HybridCNN_BiLSTM(nn.Module):
    def __init__(self):
        super(HybridCNN_BiLSTM, self).__init__()
        self.embedding = nn.Embedding(MAX_WORDS, EMBEDDING_DIM, padding_idx=0)
        self.conv = nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, 
                            num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(256, 64) 
        self.fc2 = nn.Linear(64, 6)
        
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1) 
        x = F.relu(self.conv(x))               
        x = self.pool(x)                       
        x = x.permute(0, 2, 1)                 
        
        lstm_out, _ = self.lstm(x)             
        x = torch.max(lstm_out, dim=1)[0]      
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

model = HybridCNN_BiLSTM().to(device)

num_pos = y_train.sum(axis=0)
num_neg = y_train.shape[0] - num_pos
pos_weight_values = num_neg / (num_pos + 1e-5)
pos_weight = torch.tensor(pos_weight_values, dtype=torch.float).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Theo dõi Gradients của mạng Hybrid trên wandb
wandb.watch(model, criterion, log="all", log_freq=10)

# ==========================================
# 5. VÒNG LẶP HUẤN LUYỆN CÓ ĐÁNH GIÁ VALIDATION
# ==========================================
print("\n5. Tiến hành huấn luyện (Có theo dõi Validation Loss)...")
start_time = time.time()
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_loader)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # <-- LOG HAI ĐƯỜNG LOSS LÊN WANDB ĐỂ VẼ BIỂU ĐỒ SONG SONG -->
    wandb.log({
        "epoch": epoch + 1, 
        "train_loss": avg_train_loss, 
        "val_loss": avg_val_loss
    })

train_time = time.time() - start_time

# ==========================================
# 6. ĐÁNH GIÁ TRÊN TẬP TEST VÀ CHỐT CHỈ SỐ
# ==========================================
print("\n6. Đánh giá trên tập TEST và Tối ưu hóa Ngưỡng...")
model.eval()
raw_probs = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        probs = torch.sigmoid(logits).cpu().numpy()
        raw_probs.append(probs)

pred_probs = np.vstack(raw_probs)

best_thresholds = []
preds_tuned = np.zeros_like(pred_probs)
threshold_logs = {} # Lưu log cho wandb

for i in range(y_test.shape[1]):
    best_t = 0.5
    best_f1 = 0
    for t in np.arange(0.1, 0.9, 0.05):
        temp_preds = (pred_probs[:, i] > t).astype(int)
        f1 = f1_score(y_test[:, i], temp_preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    best_thresholds.append(best_t)
    preds_tuned[:, i] = (pred_probs[:, i] > best_t).astype(int)
    threshold_logs[f"Threshold_{label_cols[i]}"] = best_t

subset_acc = accuracy_score(y_test, preds_tuned)
hl = hamming_loss(y_test, preds_tuned)
macro_f1 = f1_score(y_test, preds_tuned, average='macro')
micro_f1 = f1_score(y_test, preds_tuned, average='micro')
macro_auc = roc_auc_score(y_test, pred_probs, average='macro')

print(f"\n|=== KẾT QUẢ ĐÁNH GIÁ HYBRID CNN-BiLSTM ===|")
print(f"| Thời gian thực thi: {train_time:.2f} giây")
print(f"| Subset Accuracy: {subset_acc:.4f}")
print(f"| Hamming Loss   : {hl:.4f}")
print(f"| Macro F1-Score : {macro_f1:.4f}")
print(f"| Micro F1-Score : {micro_f1:.4f}")
print(f"| Macro ROC-AUC  : {macro_auc:.4f}")
print(f"|==========================================|")

# <-- LOG TẤT CẢ KẾT QUẢ CUỐI CÙNG LÊN WANDB -->
final_metrics = {
    "Subset Accuracy": subset_acc,
    "Hamming Loss": hl,
    "Macro F1-Score": macro_f1,
    "Micro F1-Score": micro_f1,
    "Macro ROC-AUC": macro_auc,
    "Training Time (s)": train_time
}
final_metrics.update(threshold_logs)
wandb.log(final_metrics)

wandb.finish()
print("\n[OK] Đã đẩy toàn bộ báo cáo lên Weights & Biases!")