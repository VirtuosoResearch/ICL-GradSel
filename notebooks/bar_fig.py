# %%
import matplotlib.pyplot as plt
import numpy as np

datasets = ['Poem-Sentiment', 'GraphQA', 'GLUE-SST2', 'CLRS-DFS', 'CR']
x = np.arange(len(datasets))
bar_width = 0.11
n_methods = 6

methods = [
    "Random-k", "BM25", "Top-k", "Top-k+ConE", 
    "FASTGRADICL-RE", "FASTGRADICL-FS"
]

f1_scores = np.array([
    [0.2816, 0.3048, 0.7585, 0.3689, 0.7571],
    [0.2597, 0.4121, 0.7987, 0.5166, 0.8020],
    [0.3575, 0.4894, 0.8390, 0.4800, 0.9369],
    [0.4150, 0.4746, 0.8193, 0.5308, 0.9177],
    [0.7059, 0.5852, 0.9575, 0.5833, 0.9780],
    [0.7059, 0.5852, 0.9575, 0.5833, 0.9780],
])

pastel_colors = [
    "#f1b15e", "#e47d72", "#c6bce3",
    "#7e75b5", "#99d3cd", "#72a5c3"
]

spacing = 0.117
offsets = np.linspace(-spacing*(n_methods-1)/2, spacing*(n_methods-1)/2, n_methods)

plt.figure(figsize=(12, 5))
for i in range(n_methods):
    plt.bar(x + offsets[i], f1_scores[i], width=bar_width,
            label=methods[i], color=pastel_colors[i])

plt.xticks(ticks=x, labels=datasets, fontsize=12)
plt.ylabel("F1 Score", fontsize=12)
plt.ylim(0.2, 1.05)
# plt.title("Comparison of Selection Methods Across Datasets", fontsize=14)

plt.grid(False)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=10)
plt.tight_layout()

plt.savefig("main_result.pdf")
