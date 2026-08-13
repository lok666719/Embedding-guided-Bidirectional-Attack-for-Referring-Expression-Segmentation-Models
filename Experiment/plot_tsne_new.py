import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

save_analysis_dir = '/public/chenxingbai/chenxingbai/EVF-SAM-main/response'

with open(os.path.join(save_analysis_dir, "all_clean.pkl"), "rb") as f:
    all_clean = pickle.load(f)

with open(os.path.join(save_analysis_dir, "all_proxy.pkl"), "rb") as f:
    all_proxy = pickle.load(f)

print("all_clean:", all_clean.shape)
print("all_proxy:", all_proxy.shape)

X = np.concatenate([all_clean, all_proxy], axis=0)


# ============================================================
# Quantitative embedding-space analysis
# ============================================================

def l2_normalize(x, eps=1e-12):
    """
    L2-normalize each embedding.
    x: [N, D]
    """
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def paired_cosine_distance(clean, proxy):
    """
    Cosine distance between paired clean and proxy embeddings.

    clean[i] and proxy[i] must correspond to the same expression.

    Returns:
        distances: [N]
    """
    assert clean.shape == proxy.shape, \
        f"Shape mismatch: clean={clean.shape}, proxy={proxy.shape}"

    clean_norm = l2_normalize(clean)
    proxy_norm = l2_normalize(proxy)

    cosine_similarity = np.sum(
        clean_norm * proxy_norm,
        axis=1
    )

    cosine_distance = 1.0 - cosine_similarity

    return cosine_distance


def mean_pairwise_cosine_distance(x):
    """
    Mean pairwise cosine distance among all embeddings.

    This measures inter-expression dispersion.

    For N embeddings:
        D = 2 / (N*(N-1)) * sum_{i<j} [1 - cos(x_i, x_j)]
    """
    x_norm = l2_normalize(x)

    # cosine similarity matrix: [N, N]
    sim_matrix = x_norm @ x_norm.T

    # cosine distance matrix
    dist_matrix = 1.0 - sim_matrix

    n = x.shape[0]

    # only take upper triangle, excluding diagonal
    upper_indices = np.triu_indices(n, k=1)

    pairwise_distances = dist_matrix[upper_indices]

    return pairwise_distances


# ------------------------------------------------------------
# 1. Clean-to-proxy cosine displacement
# ------------------------------------------------------------

clean_proxy_distances = paired_cosine_distance(
    all_clean,
    all_proxy
)

clean_proxy_mean = np.mean(clean_proxy_distances)
clean_proxy_std = np.std(clean_proxy_distances)
clean_proxy_median = np.median(clean_proxy_distances)

print("\n" + "=" * 70)
print("Clean-to-Proxy Cosine Distance")
print("=" * 70)

print(f"Number of paired embeddings : {len(clean_proxy_distances)}")
print(f"Mean cosine distance        : {clean_proxy_mean:.6f}")
print(f"Std                         : {clean_proxy_std:.6f}")
print(f"Median                      : {clean_proxy_median:.6f}")
print(f"Min                         : {np.min(clean_proxy_distances):.6f}")
print(f"Max                         : {np.max(clean_proxy_distances):.6f}")


# ------------------------------------------------------------
# 2. Inter-expression dispersion
# ------------------------------------------------------------

clean_pairwise_distances = mean_pairwise_cosine_distance(
    all_clean
)

proxy_pairwise_distances = mean_pairwise_cosine_distance(
    all_proxy
)

clean_dispersion = np.mean(clean_pairwise_distances)
proxy_dispersion = np.mean(proxy_pairwise_distances)

dispersion_change = proxy_dispersion - clean_dispersion

# relative change, only for interpretation
dispersion_change_percent = (
    dispersion_change / clean_dispersion * 100.0
)

print("\n" + "=" * 70)
print("Inter-expression Embedding Dispersion")
print("=" * 70)

print(
    f"Number of pairwise comparisons : "
    f"{len(clean_pairwise_distances)}"
)

print(
    f"Clean embedding dispersion     : "
    f"{clean_dispersion:.6f}"
)

print(
    f"Proxy embedding dispersion     : "
    f"{proxy_dispersion:.6f}"
)

print(
    f"Absolute dispersion change     : "
    f"{dispersion_change:+.6f}"
)

print(
    f"Relative dispersion change     : "
    f"{dispersion_change_percent:+.2f}%"
)


# ------------------------------------------------------------
# Save quantitative results
# ------------------------------------------------------------

embedding_analysis = {
    "num_embedding_pairs": int(len(all_clean)),
    "embedding_dimension": int(all_clean.shape[1]),

    "clean_proxy_cosine_distance": {
        "mean": float(clean_proxy_mean),
        "std": float(clean_proxy_std),
        "median": float(clean_proxy_median),
        "min": float(np.min(clean_proxy_distances)),
        "max": float(np.max(clean_proxy_distances)),
    },

    "inter_expression_dispersion": {
        "clean_mean_pairwise_cosine_distance":
            float(clean_dispersion),

        "proxy_mean_pairwise_cosine_distance":
            float(proxy_dispersion),

        "absolute_change":
            float(dispersion_change),

        "relative_change_percent":
            float(dispersion_change_percent),
    },

    "tsne_settings": {
        "n_components": 2,
        "perplexity": 30,
        "learning_rate": 200,
        "init": "pca",
        "random_state": 42,
        "n_iter": 2000,
        "num_clean_embeddings": int(len(all_clean)),
        "num_proxy_embeddings": int(len(all_proxy)),
        "total_points": int(len(all_clean) + len(all_proxy)),
    }
}


import json

result_path = os.path.join(
    save_analysis_dir,
    "embedding_quantitative_analysis.json"
)

with open(result_path, "w", encoding="utf-8") as f:
    json.dump(
        embedding_analysis,
        f,
        indent=4
    )

print("\nSaved quantitative results to:")
print(result_path)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    init='pca',
    random_state=42,
    n_iter=2000
)

X_2d = tsne.fit_transform(X)

n = len(all_clean)
clean_2d = X_2d[:n]
proxy_2d = X_2d[n:]

fig, ax = plt.subplots(figsize=(7.0, 5.6))

# 先画 clean：空心圆，颜色深一点
ax.scatter(
    clean_2d[:, 0], clean_2d[:, 1],
    marker='o',
    s=24,
    c='#1f77b4',
    alpha=0.7,
    linewidths=0.0,
    label='Original Text Embeddings',
    zorder=2
)
# 再画 proxy：实心三角，另一种颜色
ax.scatter(
    proxy_2d[:, 0], proxy_2d[:, 1],
    marker='^',
    s=26,
    c='#d95f02',
    linewidths=0.4,
    alpha=0.65,
    label='Optimized Proxy Embeddings',
    zorder=3
)

ax.set_xlabel('t-SNE Dimension 1', fontsize=15)
ax.set_ylabel('t-SNE Dimension 2', fontsize=15)
ax.tick_params(axis='both', labelsize=12)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)

legend = ax.legend(
    fontsize=11.5,
    frameon=True,
    loc='upper right',
    borderpad=0.4,
    handletextpad=0.8
)
legend.get_frame().set_alpha(0.95)

fig.tight_layout()

pdf_path = os.path.join(save_analysis_dir, 'proxy_tsne_main_v4.pdf')
png_path = os.path.join(save_analysis_dir, 'proxy_tsne_main_v4.png')

fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
fig.savefig(png_path, dpi=500, bbox_inches='tight')
plt.show()

print("Saved PDF to:", pdf_path)
print("Saved PNG to:", png_path)