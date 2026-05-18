"""权重初始化方面，Linear 层的权重除以 fan_in**0.5，这是 Kaiming 初始化的思路，防止前向传播时激活值方差逐层放大或缩小。最后一层的 BatchNorm gamma 乘以 0.1，压小初始 logits，
让 softmax 输出接近均匀分布，避免训练初期 loss 过大。在第一部分的手动实现中，W1 还乘了 (5/3)/((n_embd * block_size)**0.5)，其中 5/3 是 tanh 的增益补偿，不过在有 BatchNorm 的深层网络中就不需要了。
Batch Normalization 是这份代码的核心优化。每个 Linear 层后面都跟了 BatchNorm，训练时用 batch 统计量归一化，推理时用 running 统计量。
它解决了几个问题：保持每层激活值在合理范围，防止 tanh 饱和；让梯度流在深层网络中保持稳定；降低了对权重初始化的敏感度。同时 Linear 层设置 bias=False，因为 BatchNorm 的 beta 参数已经起到偏置的作用。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear:
    """torch.nn.linear(fan_in, fan_out, bias=True)"""
    def __init__(self, fan_in, fan_out, bias = True):
        """beacuse of batchmorn, now the factor fan_in**0.5 is not necessary, but there is no harm to keep it"""
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5 #keep the output is normal distribution.
        """notice that if there is nonlinear layer after this layer,other normalization like maiking and xavier is needed"""
        self.bias = torch.zeros(fan_out) if bias else None #if next layer is batchnorm,there is no need to add bias
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameter(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    """torch.nn.BatchNorm1d"""

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum #help to change running_mean and running_var
        self.training = True
        #帮助模型获得一定学习灵活度
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameter(self):
        return [self.gamma, self.beta]
    
class tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameter(self):
        return []
words = open('names.txt', 'r').read().splitlines()
letters = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(letters)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}
vocab_size = len(itos)
block_size = 3
def build_dataset(words):  
  X, Y = [], []
  
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%
n_embd = 10
n_hidden = 100
g = torch.Generator().manual_seed(2147483647)

C = torch.randn((vocab_size, n_embd), generator=g)
layers = [Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), tanh(), 
          Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), tanh(), 
          Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), tanh(), 
          Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), tanh(), 
          Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), tanh(), 
          Linear(n_hidden, vocab_size, bias=False)]# output layer needn't to add batchnorm layer
with torch.no_grad():
    layers[-1].weight *= 0.1 # avoid overconfidence at beginning
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 1 # if no batchnorm, weight should times 5/3 as the nonlinear function is tanh
parameters = [C] + [p for layer in layers for p in layer.parameter()]
for p in parameters:
    p.requires_grad = True



max_steps = 200000
batch_size = 32
lossi = []
ud = []

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    emb = C[Xb]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 150000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    if i % 10000 == 0: 
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
    with torch.no_grad():
        ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])

def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
  }[split]
  emb = C[x]  # 一次性把整个数据集都传进去
  x = emb.view(emb.shape[0], -1)
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, y)

for layer in layers:
    layer.training = False
split_loss('train')
split_loss('test')

g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    name = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        x = emb.view(emb.shape[0], -1)
        for layer in layers:
            x = layer(x)
        logits = x
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        name.append(itos[ix])
        if ix == 0:
            break
    print(''.join(name))