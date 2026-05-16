import torch
words = open('names.txt', 'r').read().splitlines()
letters = sorted(list(set(''.join(words))))
stoi = {l:i + 1 for i, l in enumerate(letters)}
stoi['.'] = 0
itos = {i:l for l, i in stoi.items()}
N = torch.zeros((27, 27), dtype = torch.int32)
for word in words:
    s = '.' + word + '.'    
    for x, y in zip(s, s[1:]):
        i = stoi[x]
        j = stoi[y]
        N[i][j] += 1
P = N.float()
P = P / P.sum(1, keepdim=True)
for i in range(20):
    out = []
    idx = 0
    while True:
        p = P[idx]
        idx = torch.multinomial(p, num_samples=1, replacement=True).item()
        out.append(itos[idx])
        if idx == 0:
            break
    print(''.join(out))
