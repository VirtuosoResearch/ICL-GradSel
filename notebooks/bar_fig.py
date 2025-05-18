# %%
import matplotlib.pyplot as plt
import numpy as np

datasets = ['Poem-Sentiment', 'GLUE-SST2', 'GraphQA', 'CLRS-DFS', 'CR', 'Coin-Flip']
x = np.arange(len(datasets))
bar_width = 0.11
n_methods = 8

methods = [
    "Random-k", "BM25", "Top-k", "Top-k+ConE",
    "Uncertain_Ranker", "FastGradICL-ConE", "FastGradICL-RE", "FastGradICL-FS"
]

f1_scores = np.array([
    [0.2983, 0.7585, 0.7852, 0.2897, 0.3627, 0.3539],  # Random-k
    [0.2597, 0.7987, 0.8020, 0.4121, 0.5166, 0.3568],  # BM25
    [0.3556, 0.8062, 0.8883, 0.6178, 0.4800, 0.3711],  # Top-k
    [0.4150, 0.8193, 0.9177, 0.6019, 0.5308, 0.5015],  # Top-k + ConE
    [0.3674, 0.9048, 0.9341, 0.6250, 0.5836, 0.4417],  # Uncertain-Ranker
    [0.3181, 0.7599, 0.9058, 0.5544, 0.5130, 0.4766],  # FASTGRADICL-CONE
    [0.3130, 0.9356, 0.8593, 0.6575, 0.6007, 0.6082],  # FASTGRADICL-RE
    [0.7059, 0.9575, 0.9780, 0.6944, 0.5833, 0.6063],  # FASTGRADICL-FS
])



pastel_colors = [
    "#f1b15e",  # Random-k        - 浅橙
    "#e47d72",  # BM25            - 珊瑚红
    "#c6bce3",  # Top-k           - 淡紫
    "#7e75b5",  # Top-k + ConE    - 深紫
    "#99d3cd",  # Uncertain       - 青绿色
    "#d4e6a5",  # FASTGRADICL-CONE - 柔和黄绿（新增，浅色）
    "#5b998e",  # FASTGRADICL-RE  - 深青绿
    "#72a5c3",  # FASTGRADICL-FS  - 淡蓝
]

spacing = 0.117
offsets = np.linspace(-spacing*(n_methods-1)/2, spacing*(n_methods-1)/2, n_methods)

plt.figure(figsize=(25, 7))
for i in range(n_methods):
    plt.bar(x + offsets[i], f1_scores[i], width=bar_width,
            label=methods[i], color=pastel_colors[i])

plt.xticks(ticks=x, labels=datasets, fontsize=18)
plt.ylabel("F1 Score", fontsize=18)
plt.yticks(fontsize=16)
plt.ylim(0.1, 1.02)
# plt.title("Comparison of Selection Methods Across Datasets", fontsize=14)

plt.grid(False)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=8, fontsize=16)
plt.tight_layout()

plt.savefig("main_result.pdf")

# %%
import matplotlib.pyplot as plt
import numpy as np

font_size=29
# 数据集名称
datasets = ['Poem-Sentiment', 'GLUE-SST2', 'CR', 'GraphQA', 'CLRS-DFS', 'Coin-Flip']
x = np.arange(len(datasets)) * 1.3  # 每组之间扩大间距

# 方法名
methods = [
    "Random-k", "BM25", "Top-k", "Top-k+ConE",
    "Uncertain_Ranker", "FastGradICL-ConE", "FastGradICL-RE", "FastGradICL-FS"
]

# F1 分数
f1_scores = np.array([
    [0.2983, 0.7585, 0.7852, 0.2897, 0.3627, 0.3539],  # Random-k
    [0.3006, 0.7742, 0.7747, 0.3039, 0.3979, 0.3421],  # BM25
    [0.3556, 0.8062, 0.8883, 0.6178, 0.5229, 0.3711],  # Top-k
    [0.4150, 0.7729, 0.9076, 0.5845, 0.5308, 0.5015],  # Top-k + ConE
    [0.3674, 0.9048, 0.9341, 0.6250, 0.5836, 0.4417],  # Uncertain-Ranker
    [0.3549, 0.7599, 0.9058, 0.5544, 0.5130, 0.4766],  # FastGradICL-CONE
    [0.3130, 0.9168, 0.9706, 0.6571, 0.5928, 0.7202],  # FastGradICL-RE
    [0.7059, 0.9423, 0.9780, 0.6654, 0.5833, 0.6063],  # FastGradICL-FS
])

pastel_colors = [
    "#f1b15e",  # Random-k
    "#e47d72",  # BM25
    "#c6bce3",  # Top-k
    "#92B9DA",  # Top-k + ConE
    "#99d3cd",  # Uncertain-Ranker
    "#d4e6a5",  # FastGradICL-CONE
    "#5b998e",  # FastGradICL-RE
    "#72a5c3",  # FastGradICL-FS
]

# bar_width = 0.08
# spacing = 0.09
bar_width = 0.095
spacing = 0.105

offsets = np.linspace(-spacing * (len(methods)-1)/2, spacing * (len(methods)-1)/2, len(methods))

plt.figure(figsize=(25, 7))
for i in range(len(methods)):
    plt.bar(x + offsets[i], f1_scores[i], width=bar_width,
            label=methods[i], color=pastel_colors[i])

plt.xticks(ticks=x, labels=datasets, fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylabel("F1 Score", fontsize=font_size)
plt.ylim(0.1, 1.02)
plt.grid(False)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=4, fontsize=font_size)
plt.tight_layout()

plt.savefig("main_result.pdf")


# %%
import matplotlib.pyplot as plt
import numpy as np

font_size=29

# datasets = [r"$\#~Input~Sample=4$",r"$\#~Input~Sample=5$",r"$\#~Input~Sample=6$",r"$\#~Input~Sample=7$",r"$\#~Input~Sample=8$"]
datasets = [r"$\mathrm{k=4}$",r"$\mathrm{k=5}$",r"$\mathrm{k=6}$",r"$\mathrm{k=7}$"]
x = np.arange(len(datasets)) * 1 

methods = [
    r"$\mathrm{\#~Anchor=1}$",
    r"$\mathrm{\#~Anchor=2}$",
    r"$\mathrm{\#~Anchor=3}$",
    r"$\mathrm{\#~Anchor=4}$",
]

f1_scores = np.array([
    [0.5226, 0.5166, 0.5066, 0.6377],  # anchor=1
    [0.5226, 0.6427, 0.5066, 0.6377],  # anchor=2
    [0.4933, 0.5756, 0.5716, 0.5485],  # anchor=3
    [0.6847, 0.5756, 0.5322, 0.5484],  # anchor=4
])
# , 0.5398
# , 0.5495
# , 0.5169
# , 0.4949
pastel_colors = [
    "#f1b15e", 
    "#e47d72", 
    "#c6bce3", 
    "#92B9DA", 
]

# bar_width = 0.08
# spacing = 0.09
bar_width = 0.12
spacing = 0.128

offsets = np.linspace(-spacing * (len(methods)-1)/2, spacing * (len(methods)-1)/2, len(methods))

plt.figure(figsize=(12, 5))
for i in range(len(methods)):
    plt.bar(x + offsets[i], f1_scores[i], width=bar_width,
            label=methods[i], color=pastel_colors[i])

plt.xticks(ticks=x, labels=datasets, fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylabel(r"$\mathrm{F_1~Score}$", fontsize=font_size)
plt.ylim(0.1, 0.9)
plt.grid(False)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=2, fontsize=font_size-6)
plt.tight_layout()

plt.savefig("ranens_ablasion.pdf")