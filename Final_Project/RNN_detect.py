import pandas as pd
import torchtext
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter, OrderedDict


data = data_repo
step_size = 7
batch_size = 1


train_essays = pd.read_csv(data + "train_essays.csv")
train_prompts = pd.read_csv(data + "train_prompts.csv")
test_essays = pd.read_csv(data + "test_essays.csv")

columns = train_essays.columns.tolist() # ['id', 'prompt_id', 'text', 'generated']

text_data = []
tokens = []
for id, row in train_essays.iterrows():
    text_data.append(row["text"])

text_data+= train_prompts["instructions"].tolist()

tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
for text in text_data:
    tokens += tokenizer(text)

counter = Counter(tokens)
sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples)
Vocab = torchtext.vocab.vocab(ordered_dict, specials=["<unk>"])

class ArticleDataSet(Dataset):
    def __init__(self, data, step_size, vocab, tokenizer):
        self.data = data
        self.vocab = vocab
        self.step_size = step_size
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index, 2]
        tokens = self.tokenizer(text)
        article_indices = [self.vocab[token] for token in tokens]
        num_elem = len(tokens) // self.step_size
        article_batches = [article_indices[i * self.step_size: (i + 1) * self.step_size] for i in range(num_elem)]
        

        return (torch.tensor(article_batches), self.data.iloc[index, 1]), self.data.iloc[index, 3]

dataSet = ArticleDataSet(train_essays, step_size, Vocab, tokenizer)
dataLoader = DataLoader(dataSet, batch_size, shuffle=True)

print(next(iter(dataLoader)))


