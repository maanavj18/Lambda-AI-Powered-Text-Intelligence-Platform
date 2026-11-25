"""Word-sequence encoding and the review sentiment LSTM."""

from collections import Counter
import re

import torch
from torch import nn


PAD = 0
UNKNOWN = 1
MAX_TOKENS = 128
MAX_VOCABULARY = 15_000
EMBEDDING_SIZE = 64
HIDDEN_SIZE = 48
DROPOUT = 0.25
TOKEN_PATTERN = re.compile(r"(?u)\b\w+(?:['’]\w+)*\b")


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower().replace("’", "'"))


def build_vocabulary(texts):
    counts = Counter(token for text in texts for token in tokenize(text)[:MAX_TOKENS])
    return {
        word: index
        for index, (word, count) in enumerate(counts.most_common(MAX_VOCABULARY - 2), start=2)
        if count >= 2
    }


def encode_reviews(texts, vocabulary):
    rows = []
    for text in texts:
        ids = [vocabulary.get(token, UNKNOWN) for token in tokenize(text)[:MAX_TOKENS]]
        rows.append(ids + [PAD] * (MAX_TOKENS - len(ids)))
    return torch.tensor(rows, dtype=torch.long)


class SentimentLSTM(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, EMBEDDING_SIZE, padding_idx=PAD)
        self.recurrent = nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, token_ids):
        mask = token_ids.ne(PAD).unsqueeze(2)
        states, _ = self.recurrent(self.dropout(self.embedding(token_ids)))
        pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.classifier(self.dropout(pooled)).squeeze(1)
