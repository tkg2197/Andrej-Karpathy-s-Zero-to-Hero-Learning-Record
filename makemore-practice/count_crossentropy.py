import math
from collections import defaultdict

names = open('names.txt', 'r').read().splitlines()

counts = defaultdict(lambda: defaultdict(int))

for name in names:
    seq = "<start>" + name + "<end>"
    for i in range(len(seq) - 1):
        context = seq[max(0, i - 2):i + 1]       # 当前上下文
        next_char = seq[i+1]      # 下一个字符
        counts[context][next_char] += 1

# 计算加权平均熵
total_steps = 0
total_entropy = 0

for context, char_counts in counts.items():
    n = sum(char_counts.values())  # 这个上下文出现了几次
    entropy = 0
    for count in char_counts.values():
        p = count / n
        entropy -= p * math.log(p)
    total_entropy += n * entropy   # 用出现次数加权
    total_steps += n

avg_entropy = total_entropy / total_steps
print(f"数据集条件熵: {avg_entropy:.3f}")