"""
Assignment: SMS Spam vs. Ham Classification Using a Convolutional Neural Network (CNN)
Deep Learning / Natural Language Processing Course
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import string
import random

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")

# ============================================================
# 1. Data Loading
# ============================================================

print("\n1. Loading dataset...")
url = ("https://raw.githubusercontent.com/mohitgupta-1O1/"
       "Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv")
df = pd.read_csv(url, encoding='latin-1')
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['label'].value_counts().to_string()}")

# ============================================================
# 2. Preprocessing and Train/Test Split
# ============================================================

print("\n2. Preprocessing and splitting...")

def tokenize(text):
    return text.lower().split()

df['tokens'] = df['message'].apply(tokenize)

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df['label_num']
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Training samples: {len(train_df)}")
print(f"Test samples:     {len(test_df)}")

# ============================================================
# 3. Vocabulary Construction (Training Data Only)
# ============================================================

print("\n3. Building vocabulary from training data...")

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

word_counts = Counter()
for tokens in train_df['tokens']:
    word_counts.update(tokens)

word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for word, _ in word_counts.most_common():
    word2idx[word] = len(word2idx)

vocab_size = len(word2idx)
PAD_IDX = word2idx[PAD_TOKEN]
UNK_IDX = word2idx[UNK_TOKEN]

print(f"Vocabulary size: {vocab_size}")

# ============================================================
# 4. Encoding, Padding/Truncating
# ============================================================

MAX_LEN = 50

def encode_and_pad(tokens, word2idx, max_len):
    """Convert tokens to indices, then pad or truncate to max_len."""
    indices = [word2idx.get(t, UNK_IDX) for t in tokens]
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices = indices + [PAD_IDX] * (max_len - len(indices))
    return indices

# ============================================================
# 5. Custom PyTorch Dataset and DataLoader
# ============================================================

class SpamDataset(Dataset):
    def __init__(self, dataframe, word2idx, max_len):
        self.labels = dataframe['label_num'].values
        self.texts = [
            encode_and_pad(tokens, word2idx, max_len)
            for tokens in dataframe['tokens']
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.texts[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )


BATCH_SIZE = 64

train_dataset = SpamDataset(train_df, word2idx, MAX_LEN)
test_dataset = SpamDataset(test_df, word2idx, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n4. DataLoaders ready — Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# ============================================================
# 6. CNN Model Architecture
# ============================================================

class SpamCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes,
                 dropout_rate, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Multiple Conv1d layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(n_filters * len(filter_sizes), 1)

    def forward(self, text):
        embedded = self.embedding(text)            # [batch, MAX_LEN, emb_dim]
        embedded = embedded.permute(0, 2, 1)       # [batch, emb_dim, MAX_LEN]

        conved = []
        for conv in self.convs:
            c = F.relu(conv(embedded))              # [batch, n_filters, L_out]
            c = F.max_pool1d(c, c.shape[2])         # [batch, n_filters, 1]
            c = c.squeeze(2)                        # [batch, n_filters]
            conved.append(c)

        cat = torch.cat(conved, dim=1)             # [batch, n_filters * len(filter_sizes)]
        cat = self.dropout(cat)
        out = self.fc(cat)                          # [batch, 1]
        return out


# Hyperparameters
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

model = SpamCNN(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    n_filters=N_FILTERS,
    filter_sizes=FILTER_SIZES,
    dropout_rate=DROPOUT_RATE,
    pad_idx=PAD_IDX
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n5. Model created — {total_params:,} parameters")
print(model)

# ============================================================
# 7. Training and Evaluation Functions
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(texts).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = torch.round(torch.sigmoid(logits))
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)

            logits = model(texts).squeeze(1)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            preds = torch.round(torch.sigmoid(logits))
            correct += (preds == labels).sum().item()
            total += len(labels)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels

# ============================================================
# 8. Training Loop
# ============================================================

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\n6. Training for {NUM_EPOCHS} epochs (Adam, lr={LEARNING_RATE}, BCEWithLogitsLoss)")
print("=" * 65)

train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(1, NUM_EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    te_loss, te_acc, te_preds, te_labels = evaluate(model, test_loader, criterion, DEVICE)

    train_losses.append(tr_loss)
    train_accs.append(tr_acc)
    test_losses.append(te_loss)
    test_accs.append(te_acc)

    print(f"  Epoch {epoch:2d}/{NUM_EPOCHS} | "
          f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
          f"Test Loss: {te_loss:.4f}  Acc: {te_acc:.4f}")

print("=" * 65)
print(f"Final Test Accuracy: {test_accs[-1]*100:.2f}%")

# ============================================================
# 9. Results Summary
# ============================================================

print(f"\n{'='*65}")
print("RESULTS SUMMARY")
print(f"{'='*65}")
print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Train Acc':>10} | {'Test Loss':>10} | {'Test Acc':>9}")
print("-" * 65)
for ep in range(NUM_EPOCHS):
    print(f"{ep+1:>6} | {train_losses[ep]:>11.4f} | {train_accs[ep]*100:>9.2f}% | "
          f"{test_losses[ep]:>10.4f} | {test_accs[ep]*100:>8.2f}%")
print("-" * 65)

# ============================================================
# 10. Confusion Matrix and Classification Report
# ============================================================

print(f"\n{'='*55}")
print("CONFUSION MATRIX")
print(f"{'='*55}")
cm = confusion_matrix(te_labels, te_preds)
tn, fp, fn, tp = cm.ravel()
print(f"                Predicted Ham  Predicted Spam")
print(f"  Actual Ham       {tn:>5}          {fp:>5}")
print(f"  Actual Spam      {fn:>5}          {tp:>5}")
print(f"\n  True Negatives (Ham correct):   {tn}")
print(f"  False Positives (Ham as Spam):  {fp}")
print(f"  False Negatives (Spam as Ham):  {fn}")
print(f"  True Positives (Spam correct):  {tp}")

print(f"\n{'='*55}")
print("CLASSIFICATION REPORT")
print(f"{'='*55}")
print(classification_report(
    te_labels, te_preds,
    target_names=['Ham (0)', 'Spam (1)'],
    digits=4
))

# ============================================================
# 11. Training Curves Plot
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, NUM_EPOCHS + 1)

ax1.plot(epochs_range, train_losses, 'b-o', markersize=5, label='Train Loss')
ax1.plot(epochs_range, test_losses, 'r-s', markersize=5, label='Test Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training vs. Test Loss', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, [a*100 for a in train_accs], 'b-o', markersize=5, label='Train Acc')
ax2.plot(epochs_range, [a*100 for a in test_accs], 'r-s', markersize=5, label='Test Acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training vs. Test Accuracy', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/malinisenthilkumar/Documents/Meharry/nlp/assignment/cnn_training_curves.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Training curves saved to cnn_training_curves.png")

# ============================================================
# 12. Inference Function
# ============================================================

def predict_message(text, model, word2idx, max_len, device):
    """Classify a raw text message as Spam or Ham with confidence."""
    model.eval()
    tokens = text.lower().split()
    encoded = encode_and_pad(tokens, word2idx, max_len)
    tensor = torch.tensor([encoded], dtype=torch.long).to(device)

    with torch.no_grad():
        logit = model(tensor).squeeze()
        prob = torch.sigmoid(logit).item()

    label = 'Spam' if prob >= 0.5 else 'Ham'
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, confidence, prob

# ============================================================
# 13. Custom Test Messages
# ============================================================

custom_messages = [
    "Congratulations! You've been selected to win a free iPhone. Click here to claim.",
    "Hey, are we still on for our study session at the library this evening?",
    "URGENT: Your bank account has been locked. Verify your password immediately.",
    "Can you please pick up some coffee and groceries on your way home?"
]

print(f"\n{'='*70}")
print("CUSTOM MESSAGE PREDICTIONS")
print(f"{'='*70}")
for msg in custom_messages:
    label, confidence, spam_prob = predict_message(msg, model, word2idx, MAX_LEN, DEVICE)
    print(f"\n  Message:    \"{msg}\"")
    print(f"  Prediction: {label} (confidence: {confidence:.2%}, spam probability: {spam_prob:.4f})")
print(f"\n{'='*70}")
print("\nDone.")
