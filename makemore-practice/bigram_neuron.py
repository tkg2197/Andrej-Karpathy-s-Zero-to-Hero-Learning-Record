import torch
import torch.nn.functional as F #神经网络中的函数操作许多都在这个库中

words = open('names.txt', 'r').read().splitlines()
letter = sorted(list(set(''.join(words))))
stoi = {s:i + 1 for i, s in enumerate(letter)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

xs = []
ys = []
"""注意这里是一次性处理所有单词，相当于是每次训练都对整个字符集训练"""
for word in words:
    chars = list('.' + word + '.')
    for ch1, ch2 in zip(chars, chars[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

xs = torch.tensor(xs)
x_onehot = F.one_hot(xs, num_classes=27).float() #矩阵乘法要求数据类型相同
ys = torch.tensor(ys)
W = torch.randn((27, 27), requires_grad=True) #为什么是27 * 27维？因为输入的数据是27维的（27种字符），输出的数据也是27维的（27个字符的概率）

for epoch in range(100):
    logits = x_onehot @ W #这里相当于矩阵的每一行是一个神经元，第i行对应第i个字符输入时输出各个字符的概率
    loss = F.cross_entropy(logits, ys) + 0.01 * (W ** 2).mean()
    W.grad = None
    loss.backward()
    with torch.no_grad():
        W -= 50*W.grad
    if epoch % 10:
        print(loss)

for i in range(20):
    idx = 0
    out = []
    while True:
        x = F.one_hot(torch.tensor([idx]), num_classes=27).float() #一定要把输入处理成one-hot向量形式
        logits = x @ W
        p = F.softmax(logits, dim=1)
        idx = torch.multinomial(p, num_samples=1).item()
        if idx == 0:
            break
        out.append(itos[idx])
    print(''.join(out))