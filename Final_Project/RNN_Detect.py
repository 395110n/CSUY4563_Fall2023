import pandas as pd
import torchtext
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from collections import Counter, OrderedDict
from d2l import torch as d2l
import torch.nn.utils.rnn as rnn_utils
import math
import matplotlib.pyplot as plt

data = data_repo
step_size = 20
batch_size = 1
num_hiddens = 512


train_essays = pd.read_csv(data + "train_essays.csv")   # ['id', 'prompt_id', 'text', 'generated']
train_prompts = pd.read_csv(data + "train_prompts.csv") # ['prompt_id', 'prompt_name', 'instructions', 'source_text', 'one_hot_prompts']
test_essays = pd.read_csv(data + "test_essays.csv")
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
tokens = tokenizer(train_essays.iloc[0, 2])

one_hot_prompts = []
for num in range(2):
    prompt = train_prompts.iloc[num, 2]
    tokens = tokenizer(prompt)
    if (len(tokens) % step_size) != 0:
        tokens += ["<unk>"] * (step_size - len(tokens) % step_size)
    num_elem = len(tokens) // step_size
    prompt_indices = [Vocab[token] for token in tokens]
    prompt_elements = torch.tensor([prompt_indices[i * step_size: (i + 1) * step_size] for i in range(num_elem)])
    prompt_elements_one_hot = F.one_hot(prompt_elements, len(Vocab))
    one_hot_prompts.append(prompt_elements_one_hot)


class ArticleDataSet(Dataset):
    def __init__(self, data, step_size, vocab, tokenizer):
        self.data = data
        self.vocab = vocab
        self.step_size = step_size
        self.tokenizer = tokenizer
        self.encoder = [torch.tensor([1., 0.]), torch.tensor([0., 1.])]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index, 2]
        tokens = self.tokenizer(text)
        if (len(tokens) % self.step_size) != 0:
            tokens += ["<unk>"] * (self.step_size - len(tokens) % self.step_size)
        article_indices = [self.vocab[token] for token in tokens]
        num_elem = len(tokens) // self.step_size
        article_batches = torch.tensor([article_indices[i * self.step_size: (i + 1) * self.step_size] for i in range(num_elem)])
        article_batches_one_hot = F.one_hot(article_batches, len(Vocab))
        return (article_batches_one_hot, self.data.iloc[index, 1]), (self.encoder[self.data.iloc[index, 3]]).to(torch.device("cuda:0"))

def rnn(X, params, state):
    W_xh, W_hh, b_h = params
    state = torch.tanh(torch.mm(X.float(), W_xh) + torch.mm(state, W_hh) + b_h).to(torch.device("cuda:0"))
    return state

def get_params(vocab_size, num_hiddens, device): 
    num_inputs = vocab_size

    def xavier_normal(shape):
        return torch.nn.init.xavier_normal_(torch.empty(shape)).to(device)
    
    W_xh = xavier_normal((num_inputs, num_hiddens))
    W_hh = xavier_normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)

    params = [W_xh, W_hh, b_h]
    for param in params:
        param.requires_grad_(True)
    return params


class RNNModel:
    def __init__(self, vocab_size, step_size, num_hiddens, device, get_params, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.step_size = step_size
        self.device = device
        self.forward_fn = forward_fn
        self.fc = nn.Linear(num_hiddens, 2, device=device)
        


    def get_init_state_per_batch(self, prompt_id):
        prompt_elements_one_hot = one_hot_prompts[prompt_id].to(self.device)
        state = torch.zeros((step_size, num_hiddens)).to(torch.device("cuda:0"))
        for data in prompt_elements_one_hot:
            state = rnn(data.to(torch.device("cuda:0")), self.params, state)
        return state.to(torch.device("cuda:0"))

    def __call__(self, X):
        state = self.get_init_state_per_batch(X[1])
        text_data = X[0].squeeze(dim=0).to(torch.device("cuda:0"))
        for data in text_data:
            state = self.forward_fn(data, self.params, state).to(torch.device("cuda:0"))
        state_flatten = state.mean(dim=0, keepdim=True)
        return self.fc(state_flatten)


num_hiddens = 512
net = RNNModel(len(Vocab), step_size, num_hiddens, torch.device("cuda:0"), get_params, rnn)

def grad_clipping(net, theta):
    params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch(net, train_iter, loss, updater, device):
    timer = d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        y_hat = net(X).to(device)
        l = loss(y_hat, Y).to(device)
        l.backward()
        grad_clipping(net, 1)
        updater.step()
        metric.add(l * Y.numel(), Y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train(net, train_iter, lr, num_epochs, device):
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    updater = torch.optim.SGD(net.params, lr)
    for epoch in range(num_epochs):
        ppl, speed = train_epoch(
            net, train_iter, loss, updater, device)
        if (epoch + 1) % 10 == 0:

            animator.add(epoch + 1, [ppl])
        print("epoch: ", epoch, " ppl: ", ppl)
    print(f'perplexity {ppl:.1f}, {speed:.1f} token/second {str(device)}')

dataSet = ArticleDataSet(train_essays, step_size, Vocab, tokenizer)
train_iter = DataLoader(dataSet, batch_size=1, shuffle=True)
num_epochs, lr = 50, 1
train(net, train_iter, lr, num_epochs, torch.device("cuda:0"))
plt.show()



