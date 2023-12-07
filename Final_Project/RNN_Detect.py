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

data = data_repo
step_size = 25
batch_size = 1
num_hiddens = 1024


train_essays = pd.read_csv(data + "train_essays.csv")   # ['id', 'prompt_id', 'text', 'generated']
train_prompts = pd.read_csv(data + "train_prompts.csv") # ['prompt_id', 'prompt_name', 'instructions', 'source_text', 'one_hot_prompts']
train_LLM1 = pd.read_csv(data + "LLM1.csv").drop("Unnamed: 0", axis=1)
train_LLM2 = pd.read_csv(data + "LLM2.csv").drop("Unnamed: 0", axis=1)
total_train_data = pd.concat([train_essays, train_LLM1, train_LLM2]).reset_index(drop=True)

text_df = pd.concat([total_train_data['text'], train_prompts['instructions']])
tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
tokens = text_df.apply(tokenizer).tolist()
flat_tokens = [word for sublist in tokens for word in sublist]

counter = Counter(flat_tokens)
sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples)
Vocab = torchtext.vocab.vocab(ordered_dict, specials=["<unk>"])

def one_hot_encoding(df, column, step_size):
    df["one_hot"] = df[column].apply(lambda text: torch.tensor(
    [Vocab[token] for token in tokenizer(text) + ["<unk>"] * ((step_size - len(tokenizer(text)) % step_size) % step_size)]).view(-1, step_size).numpy())
    # df["one_hot"] = None
    # for id, row in df.iterrows():
    #     text = row[column]
    #     tokens = tokenizer(text)
    #     if (len(tokens) % step_size) != 0:
    #         tokens += ["<unk>"] * (step_size - len(tokens) % step_size)
    #     num_elem = len(tokens) // step_size
    #     indices = [Vocab[token] for token in tokens]
    #     elements = torch.tensor([indices[i * step_size: (i + 1) * step_size] for i in range(num_elem)])
    #     df.at[id, "one_hot"] = elements.numpy()

one_hot_encoding(train_prompts, "instructions", step_size)
one_hot_encoding(total_train_data, 'text', step_size)
train_prompts.at[0, "one_hot"] = F.one_hot(torch.tensor(train_prompts.iloc[0, 4]), len(Vocab)).numpy()
train_prompts.at[1, "one_hot"] = F.one_hot(torch.tensor(train_prompts.iloc[1, 4]), len(Vocab)).numpy()
print("end encoding!----------------------------------------------")

class ArticleDataSet(Dataset):
    def __init__(self, df, step_size, vocab, tokenizer):
        self.data = df.sample(frac=1).reset_index(drop=True)
        self.vocab = vocab
        self.step_size = step_size
        self.tokenizer = tokenizer
        self.encoder = [torch.tensor([1., 0.]), torch.tensor([0., 1.])]

    def __len__(self):
        return self.step_size

    def __getitem__(self, index):
        one_hot = F.one_hot(torch.tensor(self.data.iloc[index, 4]), len(Vocab)).to(torch.device("cuda:0"))
        return (one_hot, self.data.iloc[index, 1]), (self.encoder[self.data.iloc[index, 3]]).to(torch.device("cuda:0"))

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


class RNNModel(nn.Module):
    def __init__(self, vocab_size, step_size, num_hiddens, device, get_params, forward_fn):
        super(RNNModel, self).__init__()
        self.fc = nn.Linear(10, 1)
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.step_size = step_size
        self.device = device
        self.forward_fn = forward_fn
        self.fc = nn.Linear(num_hiddens, 2, device=device)
        
        
    def get_init_state_per_batch(self, prompt_id):
        prompt_elements_one_hot = torch.tensor(train_prompts.iloc[prompt_id, 4].item()).to(self.device)
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
        result = self.fc(state_flatten)
        result_soft_max = nn.functional.softmax(result, dim=1)
        return result_soft_max


net = RNNModel(len(Vocab), step_size, num_hiddens, torch.device("cuda:0"), get_params, rnn)


def train_epoch(net, train_iter, loss, updater, device):
    custom_params = [param for param in net.params]
    fully_connected_params = list(net.fc.parameters())
    all_params = custom_params + fully_connected_params
    loss_counter = 0
    for X, Y in train_iter:
        y_hat = net(X).to(device)
        l = loss(y_hat, Y).to(device)
        l.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=2)
        loss_counter += l.item()
        updater.step()
    return loss_counter / len(train_iter)

def train(net, train_iter, lr, decay, num_epochs, device):
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    custom_params = [param for param in net.params]  # Assuming model.params is a list of custom parameters
    fully_connected_params = list(net.fc.parameters())
    all_params = custom_params + fully_connected_params
    updater = torch.optim.SGD(all_params, lr, weight_decay=decay)
    for epoch in range(num_epochs):
        ppl= train_epoch(
            net, train_iter, loss, updater, device)
        if (epoch + 1) % 10 == 0:

            animator.add(epoch + 1, [ppl])
        print("epoch: ", epoch, " ppl: ", ppl)
    

dataSet = ArticleDataSet(total_train_data, step_size, Vocab, tokenizer)
random_sampler = RandomSampler(dataSet)
train_iter = DataLoader(dataSet, batch_size=1, sampler=random_sampler)
num_epochs, lr = 300, 0.0001
decay = 0.001
train(net, train_iter, lr, decay, num_epochs, torch.device("cuda:0"))

torch.save(net.state_dict(), data+"model")

plt.show()




