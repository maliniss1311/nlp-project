"""
Lab Assignment: Spam vs. Ham Text Classification with Feedforward Neural Networks
Natural Language Processing Course - 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import random
from sklearn.model_selection import train_test_split

nltk.download('stopwords', quiet=True)

# ============================================================
# 1. Data Loading, Cleaning, and Explicit Train/Test Splitting
# ============================================================

def load_and_split_data():
    url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})

    stop_words = set(stopwords.words('english'))
    punctuations = set(string.punctuation)

    processed_data = []
    for msg, label in zip(df['message'], df['label_num']):
        tokens = [w.lower() for w in msg.split()
                  if w.lower() not in stop_words and w not in punctuations]
        processed_data.append((tokens, label))

    # Explicit 80/20 train-test split
    train_data, test_data = train_test_split(
        processed_data, test_size=0.2, random_state=42
    )
    return train_data, test_data


print("Loading and splitting data...")
train_dataset, test_dataset = load_and_split_data()
print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# ============================================================
# 2. Vocabulary Construction (Training Data Only)
# ============================================================

def build_vocab(train_data):
    word2idx = {'<UNK>': 0}
    idx = 1
    for tokens, _ in train_data:
        for token in tokens:
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1
    return word2idx


word2idx = build_vocab(train_dataset)
vocab_size = len(word2idx)
print(f"Vocabulary size (from training set only): {vocab_size}")

# ============================================================
# 3. Numericalization: Convert tokens to index tensors
# ============================================================

def encode_message(tokens, word2idx, max_len=None):
    indices = [word2idx.get(t, word2idx['<UNK>']) for t in tokens]
    if len(indices) == 0:
        indices = [word2idx['<UNK>']]
    return indices


def collate_batch(batch, word2idx):
    """Pad sequences in a batch to the same length and return tensors."""
    encoded = [encode_message(tokens, word2idx) for tokens, _ in batch]
    labels = [label for _, label in batch]

    max_len = max(len(seq) for seq in encoded)
    padded = [seq + [word2idx['<UNK>']] * (max_len - len(seq)) for seq in encoded]

    x = torch.tensor(padded, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float)
    return x, y


# ============================================================
# 4. Model Architecture: Feedforward Neural Network
# ============================================================

class SpamClassifierFNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1):
        super(SpamClassifierFNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Input features to fc1 must equal embedding_dim due to mean pooling
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        doc_vector = embedded.mean(dim=1)  # Mean pooling across sequence length
        out = self.fc1(doc_vector)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# ============================================================
# 5. Training and Evaluation Functions
# ============================================================

def train_one_epoch(model, train_data, word2idx, optimizer, criterion, batch_size=64):
    model.train()
    random.shuffle(train_data)
    total_loss = 0.0
    correct = 0
    total = 0

    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i + batch_size]
        x, y = collate_batch(batch, word2idx)

        optimizer.zero_grad()
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch)
        preds = (logits >= 0.0).float()
        correct += (preds == y).sum().item()
        total += len(batch)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, test_data, word2idx, criterion, batch_size=64):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            x, y = collate_batch(batch, word2idx)

            logits = model(x).squeeze(1)
            loss = criterion(logits, y)

            total_loss += loss.item() * len(batch)
            preds = (logits >= 0.0).float()
            correct += (preds == y).sum().item()
            total += len(batch)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ============================================================
# 6. Experimental Evaluation: Epoch Sweep [5, 10, 20, 40]
# ============================================================

# Hyperparameters
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCH_CONFIGS = [5, 10, 20, 40]

print("\n" + "=" * 65)
print("Experimental Evaluation: Training across epoch configurations")
print("=" * 65)
print(f"Embedding dim: {EMBEDDING_DIM}, Hidden dim: {HIDDEN_DIM}, "
      f"LR: {LEARNING_RATE}, Batch size: {BATCH_SIZE}")
print("=" * 65)

results = {}

for num_epochs in EPOCH_CONFIGS:
    print(f"\n--- Training with {num_epochs} epochs ---")

    # Re-initialize model and optimizer for each experiment
    torch.manual_seed(42)
    model = SpamClassifierFNN(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_dataset, word2idx, optimizer, criterion, BATCH_SIZE
        )
        te_loss, te_acc = evaluate(
            model, test_dataset, word2idx, criterion, BATCH_SIZE
        )

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        test_losses.append(te_loss)
        test_accs.append(te_acc)

        if epoch % max(1, num_epochs // 5) == 0 or epoch == num_epochs:
            print(f"  Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | "
                  f"Test Loss: {te_loss:.4f}, Test Acc: {te_acc:.4f}")

    results[num_epochs] = {
        'final_train_loss': train_losses[-1],
        'final_train_acc': train_accs[-1],
        'final_test_loss': test_losses[-1],
        'final_test_acc': test_accs[-1],
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
    }

# ============================================================
# 7. Summary Table
# ============================================================

print("\n" + "=" * 65)
print("SUMMARY: Training vs. Test Accuracy Across Epoch Configurations")
print("=" * 65)
print(f"{'Epochs':>8} | {'Train Loss':>12} | {'Train Acc':>10} | "
      f"{'Test Loss':>12} | {'Test Acc':>10}")
print("-" * 65)
for ep in EPOCH_CONFIGS:
    r = results[ep]
    print(f"{ep:>8} | {r['final_train_loss']:>12.4f} | {r['final_train_acc']:>10.4f} | "
          f"{r['final_test_loss']:>12.4f} | {r['final_test_acc']:>10.4f}")
print("-" * 65)

# ============================================================
# 8. Overfitting Analysis Output
# ============================================================

print("\n" + "=" * 65)
print("OVERFITTING ANALYSIS")
print("=" * 65)

best_test_ep = max(EPOCH_CONFIGS, key=lambda e: results[e]['final_test_acc'])
worst_gap_ep = max(EPOCH_CONFIGS,
                   key=lambda e: results[e]['final_train_acc'] - results[e]['final_test_acc'])

print(f"Best test accuracy: {results[best_test_ep]['final_test_acc']:.4f} "
      f"(at {best_test_ep} epochs)")
print(f"Largest train-test gap: "
      f"{results[worst_gap_ep]['final_train_acc'] - results[worst_gap_ep]['final_test_acc']:.4f} "
      f"(at {worst_gap_ep} epochs)")
print(f"  -> Train Acc: {results[worst_gap_ep]['final_train_acc']:.4f}, "
      f"Test Acc: {results[worst_gap_ep]['final_test_acc']:.4f}")

if results[40]['final_train_acc'] > results[40]['final_test_acc'] + 0.05:
    print("\nObservation: At 40 epochs, the model shows clear signs of overfitting.")
    print("Training accuracy continues to climb while test accuracy plateaus or drops,")
    print("demonstrating the 'learning by heart' phenomenon where the network memorizes")
    print("training data noise rather than learning generalizable patterns.")
else:
    print("\nThe model shows relatively stable generalization across epoch settings.")

print("\nDone.")
