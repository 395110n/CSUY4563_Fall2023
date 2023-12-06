import torch.nn as nn
import pandas as pd
import torchtext
import torch
from torch.nn import functional as F
from collections import Counter, OrderedDict
from torch.utils.data import Dataset, DataLoader
import random

step_size = 20


data = data_repo
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

def ont_hot_encoding(df, column, step_size):
    df["one_hot"] = None
    for id, row in df.iterrows():
        text = row[column]
        tokens = tokenizer(text)
        if (len(tokens) % step_size) != 0:
            tokens += ["<unk>"] * (step_size - len(tokens) % step_size)
        num_elem = len(tokens) // step_size
        indices = [Vocab[token] for token in tokens]
        elements = torch.tensor([indices[i * step_size: (i + 1) * step_size] for i in range(num_elem)])
        df.at[id, "one_hot"] = elements.numpy()

ont_hot_encoding(train_prompts, "instructions", step_size)
ont_hot_encoding(train_essays, 'text', step_size)
train_prompts.at[0, "one_hot"] = F.one_hot(torch.tensor(train_prompts.iloc[0, 4]), len(Vocab)).numpy()
train_prompts.at[1, "one_hot"] = F.one_hot(torch.tensor(train_prompts.iloc[1, 4]), len(Vocab)).numpy()

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
        prompt_elements_one_hot = torch.tensor(train_prompts.iloc[prompt_id, 4]).to(self.device)
        state = torch.zeros((self.step_size, self.num_hiddens)).to(torch.device("cuda:0"))
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
        one_hot = F.one_hot(torch.tensor(self.data.iloc[index, 4]), len(Vocab)).to(torch.device("cuda:0"))
        return (one_hot, self.data.iloc[index, 1]), (self.encoder[self.data.iloc[index, 3]]).to(torch.device("cuda:0"))

########################################################################################################

loaded_model = RNNModel(len(Vocab), step_size=20, num_hiddens=512, device=torch.device("cuda:0"), get_params=get_params, forward_fn=rnn)
loaded_model.load_state_dict(torch.load(data+"model"))

def predict(loaded_model):
    index = random.randint(0, 1378)
    text_data = F.one_hot(torch.tensor(train_essays.iloc[index, 4]), len(Vocab))
    intput = (text_data, train_essays.iloc[index, 1])
    real = train_essays.iloc[index, 3]
    output = loaded_model(intput)
    print(output, " ", real)
    
predict(loaded_model)
