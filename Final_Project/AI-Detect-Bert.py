import pandas as pd
import torchtext
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.nn import functional as F
from collections import Counter, OrderedDict
from d2l import torch as d2l
import torch.nn.utils.rnn as rnn_utils
import math
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import random
import collections
import time
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


data = "C:\\Users\\User\\NYU\\CSUY 4563\\data\\"

class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        """Defined in :numref:`sec_text-sequence`"""
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']


class TransformerEncoderBlock(nn.Module):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias=False):
        super().__init__()
        self.attention = nn.MultiheadAttention(num_hiddens, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(num_hiddens, ffn_num_hiddens),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_num_hiddens, num_hiddens)
        )
        # Layer normalization
        self.norm1 = nn.LayerNorm(num_hiddens)
        self.norm2 = nn.LayerNorm(num_hiddens)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, valid_lens=None):
        attn_output, _ = self.attention(X, X, X)
        Y = self.norm1(X + self.dropout(attn_output))
        return self.norm2(Y + self.dropout(self.ffn(Y)))

# BERT Encoder
class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_blks, max_len=1000, ffn_num_hiddens=256, num_heads=4, dropout=0.1):
        super(BERTEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for _ in range(num_blks):
            self.blks.add_module("block", TransformerEncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X)
        return X
    
def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences.

    Defined in :numref:`sec_utils`"""
    if len(line) > num_steps:
        return ['<cls>'] +line[:num_steps]+ ['<sep>'] # Truncate
    return ['<cls>'] + line + [padding_token] * (num_steps - len(line)) + ['<sep>']  # Pad



# Step 1: Tokenization and Vocabulary Creation


class BERTClassifier(nn.Module):
    def __init__(self, bert_encoder, num_hiddens, num_classes, device):
        super(BERTClassifier, self).__init__()
        self.encoder = bert_encoder
        self.classifier = nn.Sequential(
            nn.Linear(num_hiddens, 1024, device=device),
            nn.Linear(1024, num_hiddens, device=device), nn.ReLU(),
            nn.Linear(num_hiddens, 256, device=device), nn.ReLU(),
            nn.Linear(256, num_classes, device=device)
        )

    def forward(self, tokens, segments=None):
        if segments is None:
            segments = torch.zeros_like(tokens)  # Create dummy segment IDs if not provided
        encoded = self.encoder(tokens, segments)
        cls_token_output = encoded[:, 0, :]
        return self.classifier(cls_token_output)

def train_epoch(net, train_iter, loss_fn, optimizer, device):
    net.train()
    total_loss, total_accuracy, num_samples = 0, 0, 0
    for X, y in train_iter:
        X, y = X.to(device), y.to(device).long()
        optimizer.zero_grad()
        outputs = net(X,None)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_accuracy += (outputs.argmax(dim=1) == y).sum().item()
        num_samples += y.size(0)
    avg_loss = total_loss / len(train_iter)
    avg_accuracy = total_accuracy / num_samples
    return avg_loss, avg_accuracy

def train_bert(net, train_iter, loss_fn, optimizer, num_epochs, device):
    all_loss, all_accuracy = [], []
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_accuracy = train_epoch(net, train_iter, loss_fn, optimizer, device)
        all_loss.append(train_loss)
        all_accuracy.append(train_accuracy)
        elapsed_time = time.time() - start_time
        print(f'Epoch {epoch + 1}, Loss: {train_loss:.3f}, Accuracy: {train_accuracy:.3f}, Time: {elapsed_time:.2f} sec')
    plt.plot(range(num_epochs), all_loss, color = "red", label="loss", marker=None)
    plt.plot(range(num_epochs), all_accuracy, color = "blue", label="accuracy", marker=None)
    plt.legend()
    plt.show()
# Setting up the device for training

step_size = 30
batch_size = 1
num_hiddens = 512

off_train_essays = pd.read_csv(data + "train_essays.csv")   # ['id', 'prompt_id', 'text', 'generated']
off_train_prompts = pd.read_csv(data + "train_prompts.csv") # ['prompt_id', 'prompt_name', 'instructions', 'source_text', 'one_hot_prompts']
train_LLM1 = pd.read_csv(data + "LLM1.csv").drop("Unnamed: 0", axis=1)
train_LLM2 = pd.read_csv(data + "LLM2.csv").drop("Unnamed: 0", axis=1)
train_essays = pd.concat([off_train_essays, train_LLM1, train_LLM2]).reset_index(drop=True)

tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
tokenized_text = [tokenizer(essay) for essay in train_essays['text']]
vocab = Vocab(tokenized_text,min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])

# Step 2: Convert Text to Indices and Normalize Lengths
max_len = 512 
padded_tokens = [truncate_pad(line, 510, '<pad>') for line in tokenized_text]
padded_indices = [vocab[token] for token in padded_tokens]


features = torch.tensor(padded_indices)
labels = torch.tensor(train_essays['generated'].values)  # Assuming 'generated' is the target column
dataset = TensorDataset(features, labels)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

vocab_size = len(vocab)
num_blks = 8       
bert_encoder = BERTEncoder(vocab_size, num_hiddens, num_blks)

num_classes = 2  
device = torch.device("cuda:0")
bert_classifier = BERTClassifier(bert_encoder, num_hiddens, num_classes, device)

bert_classifier.to(device)

# Loss Function and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(bert_classifier.parameters(), lr=0.0001)

# Training the Model
num_epochs = 50
train_bert(bert_classifier, train_loader, loss_fn, optimizer, num_epochs, device)