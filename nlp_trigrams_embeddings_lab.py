"""
NLP Lab: Trigrams and Word Embeddings
A Guided Introduction to Statistical and Neural NLP
Jurafsky & Martin, Ch. 2, 3, & 6
"""

import nltk
import re
from collections import Counter
import math
from gensim.models import Word2Vec

nltk.download('webtext', quiet=True)
from nltk.corpus import webtext

# =============================================================================
# Task 1: Preprocessing
# =============================================================================

def preprocess(raw_text):
    """Convert raw corpus into a list of tokenized sentences."""
    sentences = []
    for line in raw_text.split('\n'):
        line = line.strip().lower()
        tokens = re.findall(r'\w+', line)
        if tokens:
            sentences.append(tokens)
    return sentences

raw = webtext.raw('wine.txt')
sentences = preprocess(raw)

print("=" * 60)
print("TASK 1: Preprocessing")
print("=" * 60)
print(f"Number of processed sentences: {len(sentences)}")
print(f"\nFirst 3 sentences:")
for s in sentences[:3]:
    print(f"  {s}")

# =============================================================================
# Task 2: Trigram Counts
# =============================================================================

# Train / Test split (80/20)
split_idx = int(len(sentences) * 0.8)
train_sentences_raw = sentences[:split_idx]
test_sentences_raw  = sentences[split_idx:]

# Build vocabulary with min frequency = 3; replace rare words with <UNK>
MIN_FREQ = 3

word_freq = Counter()
for sent in train_sentences_raw:
    word_freq.update(sent)

vocab = {w for w, c in word_freq.items() if c >= MIN_FREQ}
vocab.update(['<s>', '</s>', '<UNK>'])
vocab_size = len(vocab)


def replace_unk(sentence, vocabulary):
    """Replace words not in vocabulary with <UNK>."""
    return [w if w in vocabulary else '<UNK>' for w in sentence]


def pad_sentence(sentence):
    """Add <s> <s> at the beginning and </s> at the end."""
    return ['<s>', '<s>'] + sentence + ['</s>']


# Apply <UNK> replacement and padding
train_sentences = [pad_sentence(replace_unk(s, vocab)) for s in train_sentences_raw]
test_sentences  = [pad_sentence(replace_unk(s, vocab)) for s in test_sentences_raw]

# Build unigram, bigram, and trigram counts from training data
unigram_counts = Counter()
bigram_counts  = Counter()
trigram_counts = Counter()

for sent in train_sentences:
    for i in range(len(sent)):
        unigram_counts[(sent[i],)] += 1
        if i >= 1:
            bigram_counts[(sent[i-1], sent[i])] += 1
        if i >= 2:
            trigram_counts[(sent[i-2], sent[i-1], sent[i])] += 1

total_unigrams = sum(unigram_counts.values())

print("\n" + "=" * 60)
print("TASK 2: Trigram Counts")
print("=" * 60)
print(f"Train sentences: {len(train_sentences)}")
print(f"Test sentences:  {len(test_sentences)}")
print(f"Vocabulary size (freq >= {MIN_FREQ} + special tokens): {vocab_size}")
print(f"Unique unigrams: {len(unigram_counts)}")
print(f"Unique bigrams:  {len(bigram_counts)}")
print(f"Unique trigrams: {len(trigram_counts)}")
print(f"Total tokens:    {total_unigrams}")

print("\nTop 10 unigrams:")
for gram, count in unigram_counts.most_common(10):
    print(f"  {gram}: {count}")

print("\nTop 10 bigrams:")
for gram, count in bigram_counts.most_common(10):
    print(f"  {gram}: {count}")

print("\nTop 10 trigrams:")
for gram, count in trigram_counts.most_common(10):
    print(f"  {gram}: {count}")

# =============================================================================
# Task 3: Trigram Probability (with Linear Interpolation)
# =============================================================================


def trigram_prob(w1, w2, w3):
    """MLE trigram probability: P(w3 | w1, w2)."""
    denom = bigram_counts.get((w1, w2), 0)
    if denom == 0:
        return 0.0
    return trigram_counts.get((w1, w2, w3), 0) / denom


def bigram_prob(w2, w3):
    """MLE bigram probability: P(w3 | w2)."""
    denom = unigram_counts.get((w2,), 0)
    if denom == 0:
        return 0.0
    return bigram_counts.get((w2, w3), 0) / denom


def unigram_prob(w3):
    """MLE unigram probability: P(w3)."""
    return unigram_counts.get((w3,), 0) / total_unigrams


def interpolated_prob(w1, w2, w3, lambda1=0.6, lambda2=0.3, lambda3=0.1):
    """Linear interpolation of trigram, bigram, and unigram probabilities."""
    p3 = trigram_prob(w1, w2, w3)
    p2 = bigram_prob(w2, w3)
    p1 = unigram_prob(w3)
    return lambda1 * p3 + lambda2 * p2 + lambda3 * p1


print("\n" + "=" * 60)
print("TASK 3: Trigram Probability (Linear Interpolation)")
print("=" * 60)

w1, w2, w3 = 'firm', 'ripe', 'tannins'
p_tri = trigram_prob(w1, w2, w3)
p_bi  = bigram_prob(w2, w3)
p_uni = unigram_prob(w3)
p_int = interpolated_prob(w1, w2, w3)

print(f"P(tannins | firm, ripe):")
print(f"  Trigram  MLE:       P = {p_tri:.6f}")
print(f"  Bigram   MLE:       P = {p_bi:.6f}")
print(f"  Unigram  MLE:       P = {p_uni:.6f}")
print(f"  Interpolated:       P = {p_int:.6f}")
print(f"    = 0.6*{p_tri:.6f} + 0.3*{p_bi:.6f} + 0.1*{p_uni:.6f}")

# =============================================================================
# Task 4: Perplexity
# =============================================================================


def compute_perplexity(padded_test_sentences):
    """Compute perplexity of the interpolated trigram model on test sentences."""
    log_prob_sum = 0.0
    N = 0

    for sent in padded_test_sentences:
        for i in range(2, len(sent)):
            prob = interpolated_prob(sent[i-2], sent[i-1], sent[i])
            if prob == 0:
                prob = 1e-10  # floor to avoid log2(0)
            log_prob_sum += math.log2(prob)
            N += 1

    log2_pp = -log_prob_sum / N
    perplexity = 2 ** log2_pp
    return perplexity, N


pp, N = compute_perplexity(test_sentences)

print("\n" + "=" * 60)
print("TASK 4: Perplexity")
print("=" * 60)
print(f"Test set tokens (N): {N}")
print(f"Test-set perplexity (interpolated model): {pp:.2f}")

# =============================================================================
# Task 5: Word2Vec
# =============================================================================

# Remove sentence-boundary markers for Word2Vec training
w2v_train = [[w for w in sent if w not in ('<s>', '</s>')] for sent in train_sentences]

model = Word2Vec(
    sentences=w2v_train,
    vector_size=100,
    window=5,
    sg=1,           # Skip-gram
    min_count=2,
    epochs=50,
    seed=42,
    workers=1
)

print("\n" + "=" * 60)
print("TASK 5: Word2Vec Skip-gram")
print("=" * 60)
print(f"Word2Vec vocabulary size: {len(model.wv)}")
print(f"Embedding dimension:      {model.wv.vector_size}")

target_word = 'tannins'
if target_word in model.wv:
    neighbors = model.wv.most_similar(target_word, topn=5)
    print(f"\n5 most similar words to '{target_word}':")
    for word, score in neighbors:
        print(f"  {word:20s}  similarity = {score:.4f}")
else:
    print(f"\n'{target_word}' not in Word2Vec vocabulary")

# Additional exploration
extra_words = ['wine', 'red', 'white', 'dry', 'sweet']
print("\nAdditional nearest neighbors:")
for word in extra_words:
    if word in model.wv:
        similar = model.wv.most_similar(word, topn=5)
        sim_str = ', '.join([f"{w} ({s:.3f})" for w, s in similar])
        print(f"  '{word}': {sim_str}")

# =============================================================================
# Task 6: Short Analysis
# =============================================================================

print("\n" + "=" * 60)
print("TASK 6: Short Analysis")
print("=" * 60)

print("""
1. Why can a trigram model assign probability zero to a reasonable phrase
   that was not observed during training?

   A trigram model estimates P(w3|w1,w2) directly from counts: C(w1,w2,w3)/C(w1,w2).
   If the specific three-word sequence was never seen in training, the numerator is
   zero, making the probability exactly zero -- regardless of whether the phrase is
   linguistically reasonable. The space of possible trigrams grows as V^3, so even a
   large corpus leaves most valid trigrams unseen.

2. Why does taking log2(0) cause a problem when calculating perplexity?

   log2(0) is mathematically undefined (it tends to negative infinity). In the
   perplexity formula, we sum log2 P(wi|context) over all test tokens. A single
   zero-probability token sends the entire sum to negative infinity, making the
   perplexity 2^(+infinity) = infinity. One unseen trigram makes the model appear
   infinitely bad, which is why smoothing or a probability floor is essential.

3. How can Word2Vec identify words related to "tannins" even when a particular
   word sequence never appeared in the training corpus?

   Word2Vec learns dense vector representations by training on local context windows,
   not specific n-gram sequences. Words that frequently appear in similar contexts
   (e.g., surrounded by "firm", "dry", "astringent", "oak") are pushed close together
   in embedding space -- even if they never co-occur directly. This distributional
   similarity ("you shall know a word by the company it keeps" -- Firth, 1957) lets
   Word2Vec capture semantic relationships that count-based n-gram models miss.
""")

print("Lab complete.")
