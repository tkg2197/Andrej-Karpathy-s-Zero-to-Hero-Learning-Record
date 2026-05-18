from pathlib import Path
import itertools
import random

import torch
import torch.nn.functional as F
import torch.nn.init as init


# 固定随机种子，保证实验结果可复现
torch.manual_seed(2147483647)
random.seed(42)

# 读取名字数据集（每行一个名字）
DATA_PATH = Path(__file__).with_name("names.txt")
words = DATA_PATH.read_text().splitlines()

# 构造字符表：把每个字符映射到一个整数 id
# "." 作为特殊的起始/结束符，固定占用 id=0
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

context_size = 3  # 上下文窗口大小：用前 3 个字符预测下一个字符
vocab_size = len(stoi)  # 词表大小（26 个字母 + 1 个特殊符号）


def build_dataset(words):
    # 把名字列表转换成训练样本：
    # X 是上下文（连续 context_size 个字符的 id），Y 是下一个字符的 id
    X, Y = [], []
    for word in words:
        # 用 0（即"."）填充作为初始上下文，相当于在每个名字前面补特殊起始符
        context = [0] * context_size
        # 名字末尾追加 "."，让模型学会在合适的位置预测"结束"
        for ch in word + ".":
            idx = stoi[ch]
            X.append(context)
            Y.append(idx)
            # 滑动窗口：丢掉最早的字符，把当前字符加到末尾
            context = context[1:] + [idx]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


# 把数据集按 8:1:1 划分为 训练集 / 验证集 / 测试集
# 训练集用于训练，验证集用于挑超参，测试集只在最后评估一次
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])


def make_model(emb_size, hidden_size):
    # 构建 MLP 模型的全部参数
    # C: 字符嵌入查找表，把每个字符 id 映射到 emb_size 维的向量
    C = torch.randn((vocab_size, emb_size))
    # W1, b1: 第一层（隐藏层）的权重和偏置
    # 输入维度是 context_size * emb_size（把上下文 3 个字符的嵌入拼接起来）
    # 乘以 0.2 是为了缩小初始权重，避免 tanh 进入饱和区导致梯度消失
    W1 = torch.randn((context_size * emb_size, hidden_size)) * 0.2
    b1 = torch.zeros(hidden_size)
    # W2, b2: 输出层，把隐藏层映射到 vocab_size 个 logits
    # 乘以 0.01 让初始 logits 接近 0，初始损失更接近理论值 log(vocab_size)
    W2 = torch.randn((hidden_size, vocab_size)) * 0.01
    b2 = torch.zeros(vocab_size)
    parameters = [C, W1, b1, W2, b2]

    # 开启梯度跟踪，后续 loss.backward() 才能算出各参数的梯度
    for p in parameters:
        p.requires_grad = True

    return {
        "C": C,
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "parameters": parameters,
        "emb_size": emb_size,
    }


def forward(model, X):
    # 前向传播：输入上下文 X，输出每个字符的 logits
    emb_size = model["emb_size"]
    # C[X] 的 shape 是 (batch, context_size, emb_size)
    # view(-1, ...) 把每行 context_size 个嵌入拼接成一个长向量
    emb = model["C"][X].view(-1, context_size * emb_size)
    # 隐藏层用 tanh 作为激活函数
    h = torch.tanh(emb @ model["W1"] + model["b1"])
    # 输出层直接给出 logits，不在这里做 softmax（交给 cross_entropy 一起算更稳定）
    logits = h @ model["W2"] + model["b2"]
    return logits


@torch.no_grad()  # 评估时不需要梯度，关掉可以节省内存、加速计算
def evaluate_loss(model, X, Y):
    logits = forward(model, X)
    loss = F.cross_entropy(logits, Y)
    return loss.item()


def get_lr(step, train_round, lr):
    # 学习率衰减：训练前半段用全学习率，中段降到 30%，后段降到 10%
    # 前期大步收敛、后期小步精修，避免在最优点附近震荡
    if step < train_round * 0.5:
        return lr
    if step < train_round * 0.8:
        return lr * 0.3
    return lr * 0.1


def train_model(config):
    # 用固定种子重置一次，让每组超参的初始化保持一致，方便公平对比
    torch.manual_seed(config["seed"])
    model = make_model(config["emb_size"], config["hidden_size"])

    for step in range(config["train_round"]):
        # 随机采样一个 mini-batch：每步只用一小批样本，加快迭代、引入随机性
        idx = torch.randint(0, Xtr.shape[0], (config["batch_size"],))
        logits = forward(model, Xtr[idx])
        loss = F.cross_entropy(logits, Ytr[idx])

        # 清空上一轮的梯度（设为 None 比 zero_() 更省一点）
        for p in model["parameters"]:
            p.grad = None

        # 反向传播：自动求出每个参数对 loss 的梯度
        loss.backward()

        # 取当前 step 应该用的学习率，做一次梯度下降更新
        # with torch.no_grad() 避免参数更新本身进入计算图
        lr = get_lr(step, config["train_round"], config["lr"])
        with torch.no_grad():
            for p in model["parameters"]:
                p -= lr * p.grad

    return model


def iter_configs():
    # 超参搜索空间：对所有组合做笛卡尔积，逐一训练评估
    search_space = {
        "emb_size": [3, 10],          # 字符嵌入维度
        "hidden_size": [100, 200],    # 隐藏层维度
        "lr": [0.1, 0.05],            # 初始学习率
        "batch_size": [32, 64],       # mini-batch 大小
    }

    keys = list(search_space)
    for values in itertools.product(*(search_space[key] for key in keys)):
        config = dict(zip(keys, values))
        config["train_round"] = 50000
        config["seed"] = 2147483647
        yield config


# 网格搜索：遍历所有超参组合，按 dev_loss 选出最优配置
best = None

for trial, config in enumerate(iter_configs(), start=1):
    print(f"\nrunning trial {trial:02d}: {config}")
    model = train_model(config)
    # 注意要在完整训练集和验证集上评估，而不是只看最后一个 batch 的 loss
    train_loss = evaluate_loss(model, Xtr, Ytr)
    dev_loss = evaluate_loss(model, Xdev, Ydev)

    print(
        f"trial {trial:02d}",
        config,
        f"train_loss={train_loss:.4f}",
        f"dev_loss={dev_loss:.4f}",
    )

    # 用验证集 loss 作为模型选择的依据（不能用测试集，否则就泄漏了）
    if best is None or dev_loss < best["dev_loss"]:
        best = {
            "config": config,
            "model": model,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
        }

# 最终在测试集上评估一次，得到对泛化能力的无偏估计
print("\nbest config:", best["config"])
print(f"best train loss: {best['train_loss']:.4f}")
print(f"best dev loss: {best['dev_loss']:.4f}")
print(f"test loss: {evaluate_loss(best['model'], Xte, Yte):.4f}")


# 用最佳模型采样 20 个名字，直观感受效果
for _ in range(20):
    # 从全是"."的初始上下文开始
    context = [0] * context_size
    name = []

    while True:
        X = torch.tensor([context])
        logits = forward(best["model"], X)
        # 把 logits 转成概率分布，然后按概率抽样一个字符
        # 用 multinomial 而不是 argmax，能产生多样性而不是每次都一样
        probs = F.softmax(logits, dim=1)
        idx = torch.multinomial(probs, num_samples=1).item()
        # 滑动上下文窗口，把刚采样的字符加进去
        context = context[1:] + [idx]

        # 采到结束符"."就停止
        if idx == 0:
            break

        name.append(itos[idx])

    print("".join(name))
