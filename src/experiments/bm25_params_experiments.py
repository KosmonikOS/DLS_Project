from src.indexing.parse import fetch_and_parse
from rank_bm25 import BM25Okapi

import asyncio
import httpx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

queries = ["kernel methods support vector machines NLP", "machine translation history", 
    "What is BERT in NLP", "transformers in NLP", "typical errors associated with SMT output", 
    "kernel engineering natural language applications tutorial", "neural lattice search domain",
    "Neural machine translation evaluation methods", "generation of visual tables", "Context-Vector Analysis"
]


# Links to articles in benchmark
article_links = ["https://aclanthology.org/2025.wnut-1.15.pdf", "https://aclanthology.org/Y11-1013.pdf",
    "https://aclanthology.org/C10-5001.pdf", "https://aclanthology.org/1952.earlymt-1.24.pdf", 
    "https://aclanthology.org/2021.scil-1.26.pdf", "https://aclanthology.org/2020.lrec-1.259.pdf",
    "https://aclanthology.org/2023.ijcnlp-main.22.pdf", "https://aclanthology.org/W16-1707.pdf",
    "https://aclanthology.org/2012.amta-commercial.5.pdf", "https://aclanthology.org/D07-1068.pdf",
    "https://aclanthology.org/O03-1004.pdf", "https://aclanthology.org/2024.naacl-long.59.pdf",
    "https://aclanthology.org/I17-2003.pdf", "https://aclanthology.org/I17-2004.pdf", 
    "https://aclanthology.org/I17-2005.pdf", "https://aclanthology.org/2009.jeptalnrecital-demonstration.9.pdf",
    "https://aclanthology.org/2009.jeptalnrecital-demonstration.10.pdf", "https://aclanthology.org/2024.emnlp-main.391.pdf",
    "https://aclanthology.org/2024.emnlp-main.392.pdf", "https://aclanthology.org/1957.earlymt-1.23.pdf",
    "https://aclanthology.org/1957.earlymt-1.24.pdf", "https://aclanthology.org/2021.motra-1.10.pdf",
    "https://aclanthology.org/2021.motra-1.11.pdf"
]

# True articles to query mapping
query_article_dict = {"kernel methods support vector machines NLP":	"3, 8, 2, 10, 18", 
                      "machine translation history": "21, 4, 20, 13, 14", 
                      "What is BERT in NLP": "6, 3, 2, 15, 17",
                      "transformers in NLP": "6, 3, 2, 15, 17",
                      "typical errors associated with SMT output":	"9, 1, 19, 20, 13",
                      "kernel engineering natural language applications tutorial": "3, 13, 1, 20, 22",
                      "neural lattice search domain": "14, 19, 10, 15, 5",
                      "Neural machine translation evaluation methods":	"14, 22, 23, 2, 9",
                      "generation of visual tables": "18, 19, 20, 3, 10",
                      "Context-Vector Analysis": "10, 18, 3, 17, 1"
}

# Get texts from pdf
async def get_texts():
    async with httpx.AsyncClient() as client:
        texts = await fetch_and_parse(article_links, client=client)
    return texts

texts = asyncio.run(get_texts())

# Tokenize texts
tokenized_corpus = [text.split() if text else [] for text in texts]

# Lists with parameters value
k1_values = [0.5, 1.2, 2.0]
b_values = [0.25, 0.75, 1.0]


# Function for computing metrics
def compute_metrics(results, gold, k=5):
    mrrs, p1s, p5s, recalls, maps, ndcgs = [], [], [], [], [], []
    for found, true_set in zip(results, gold):
        # MRR
        rr = 0
        for rank, idx in enumerate(found, 1):
            if idx in true_set:
                rr = 1 / rank
                break
        mrrs.append(rr)

        # Precision@1
        p1s.append(1 if found[0] in true_set else 0)

        # Precision@5
        p5s.append(len([i for i in found[:k] if i in true_set]) / k)

        # Recall@5
        recall = len([i for i in found[:k] if i in true_set]) / len(true_set) if true_set else 0
        recalls.append(recall)

        # AP (Average Precision)
        hits = 0
        sum_precisions = 0
        for i, idx in enumerate(found[:k], 1):
            if idx in true_set:
                hits += 1
                sum_precisions += hits / i
        ap = sum_precisions / len(true_set) if true_set else 0
        maps.append(ap)

        # NDCG@5
        dcg = 0
        for i, idx in enumerate(found[:k]):
            rel = 1 if idx in true_set else 0
            dcg += rel / np.log2(i + 2)

        # Ideal DCG
        ideal_rels = [1] * min(len(true_set), k) + [0] * (k - min(len(true_set), k))
        idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels)])
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)

    return {
        "MRR": np.mean(mrrs),
        "P@1": np.mean(p1s),
        "P@5": np.mean(p5s),
        "Recall@5": np.mean(recalls),
        "MAP": np.mean(maps),
        "NDCG@5": np.mean(ndcgs),
    }

results_lines = []
results_table = []

for k1 in k1_values:
    for b in b_values:
        bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        all_results = []
        gold_sets = []
        for q in queries:
            scores = bm25.get_scores(q.split())
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            top_indices = np.argsort(scores)[::-1][:5]
            all_results.append(top_indices.tolist())
            gold = set(int(x)-1 for x in query_article_dict[q].split(","))
            gold_sets.append(gold)
        metrics = compute_metrics(all_results, gold_sets)
        line = f"BM25 k1={k1}, b={b}:\n" + \
               "\n".join([f"  {k}: {v:.4f}" for k, v in metrics.items()]) + "\n" + ("-"*30) + "\n"
        print(line)
        results_lines.append(line)

        row = {"k1": k1, "b": b}
        row.update(metrics)
        results_table.append(row)

with open("bm25_results.txt", "w", encoding="utf-8") as f:
    f.writelines(results_lines)

# Draw graphics
df = pd.DataFrame(results_table)

metrics_to_plot = ["MRR", "MAP", "Recall@5", "NDCG@5", "P@1", "P@5"]
n_metrics = len(metrics_to_plot)
ncols = 3
nrows = (n_metrics + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
axes = axes.flatten()

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx]
    for b in b_values:
        subset = df[df["b"] == b]
        ax.plot(subset["k1"], subset[metric], marker='o', label=f"b={b}")
    ax.set_title(f"{metric} vs k1")
    ax.set_xlabel("k1")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True)

for idx in range(n_metrics, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig("bm25_all_metrics.png")
plt.show()