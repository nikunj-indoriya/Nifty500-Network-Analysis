"""
step3_threshold_network.py
----------------------------------------------------------
Constructs a thresholded correlation network (Moghadam 2019; Tse 2010)
and computes an extended set of network metrics.

Input:
  data/corr_matrix.csv
  data/returns.csv (optional)
  ind_nifty500list.csv (for industry homophily)

Output:
  data/adjacency_theta_<val>.csv
  data/metrics_theta_<val>.csv
  data/G_theta.csv
  data/plots/ (G(theta), degree hist, log-log power law, etc.)

Extended metrics added:
- Katz, PageRank, and HITS centralities
- K-core number, clique count, global transitivity
- Similarity measures (cosine, Jaccard)
- Small-world sigma (vs random graph)
- Homophily/assortativity by industry
- Power-law exponent fitting for degree distribution
----------------------------------------------------------
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, linregress

def cosine_similarity_manual(matrix):
    """Compute cosine similarity between rows of a matrix using NumPy only."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12  # avoid division by zero
    normalized = matrix / norms
    sim = np.dot(normalized, normalized.T)
    return sim

# ---------------- CONFIG ----------------
data_dir = Path("data")
plots_dir = data_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

corr_file = data_dir / "corr_matrix.csv"
returns_file = data_dir / "returns.csv"
sector_file = "ind_nifty500list.csv"  # for homophily analysis

theta_grid = np.linspace(0.0, 0.9, 91)
save_prefix = "theta"

# ---------------- LOAD --------------------
print("Loading correlation matrix:", corr_file)
C = pd.read_csv(corr_file, index_col=0)
tickers = C.index.tolist()
n = len(tickers)
print(f"Loaded correlation matrix for {n} tickers. Shape: {C.shape}")

C = C.reindex(index=tickers, columns=tickers)
np.fill_diagonal(C.values, 1.0)

# ---------------- Flatten upper triangle ----------------
def flatten_upper(matrix_df):
    vals, pairs = [], []
    idx = matrix_df.index.tolist()
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            pairs.append((idx[i], idx[j]))
            vals.append(abs(matrix_df.iloc[i, j]))
    return np.array(vals), pairs

absC_vec, pairs = flatten_upper(C)

# ---------------- Consistency function G(theta) ----------------
def compute_G_theta(absC_vec, theta):
    A = (absC_vec >= theta).astype(float)
    if A.std(ddof=0) == 0 or absC_vec.std(ddof=0) == 0:
        return np.nan
    r, _ = pearsonr(absC_vec, A)
    return r

G_vals = [compute_G_theta(absC_vec, th) for th in theta_grid]
G_df = pd.DataFrame({"theta": theta_grid, "G": G_vals})
G_df.to_csv(data_dir / "G_theta.csv", index=False)
print("Saved G(theta) values to data/G_theta.csv")

plt.figure(figsize=(7, 4))
plt.plot(G_df["theta"], G_df["G"], marker='o', markersize=3, linewidth=1)
plt.xlabel("Theta (threshold)")
plt.ylabel("G(theta) (consistency)")
plt.title("Consistency function G(theta)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / "G_theta.png", dpi=150)
plt.close()

valid = G_df["G"].replace([np.inf, -np.inf], np.nan).dropna()
if valid.empty:
    raise RuntimeError("No valid G(theta) values found.")
best_row = G_df.loc[G_df["G"].idxmax()]
theta_best = float(best_row["theta"])
print(f"Selected theta_hat = {theta_best:.3f} with G = {best_row['G']:.4f}")

# ---------------- Build adjacency ----------------
A = np.zeros((n, n), dtype=float)
for idx, (i, j) in enumerate(pairs):
    val = absC_vec[idx]
    if val >= theta_best:
        ii, jj = tickers.index(i), tickers.index(j)
        A[ii, jj] = val
        A[jj, ii] = val

A_df = pd.DataFrame(A, index=tickers, columns=tickers)
adj_file = data_dir / f"adjacency_{save_prefix}_{theta_best:.3f}.csv"
A_df.to_csv(adj_file)
print("Saved adjacency matrix (weighted) to", adj_file)

# ---------------- Build graph ----------------
G = nx.from_numpy_array(A)
mapping = {i: tickers[i] for i in range(n)}
G = nx.relabel_nodes(G, mapping)
print(f"Constructed graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# ---------------- Core metrics ----------------
metrics = pd.DataFrame(index=tickers)
metrics["degree"] = dict(G.degree())
metrics["strength"] = dict(G.degree(weight="weight"))

# Weighted betweenness & closeness (use 1/weight as distance)
G_dist = nx.Graph()
for u, v, data in G.edges(data=True):
    w = data["weight"]
    if w > 0:
        G_dist.add_edge(u, v, weight=1.0 / w)

bet = nx.betweenness_centrality(G_dist, weight="weight", normalized=True)
clo = nx.closeness_centrality(G_dist, distance="weight")
metrics["betweenness"] = pd.Series(bet)
metrics["closeness"] = pd.Series(clo)

# Eigenvector, Katz, PageRank, HITS (on largest connected component)
largest_cc = max(nx.connected_components(G), key=len)
G_main = G.subgraph(largest_cc).copy()

metrics["eigenvector"] = pd.Series(nx.eigenvector_centrality_numpy(G_main, weight="weight"))
metrics["katz"] = pd.Series(nx.katz_centrality_numpy(G_main, alpha=0.005, beta=1.0))
metrics["pagerank"] = pd.Series(nx.pagerank(G_main, alpha=0.85, weight="weight"))

try:
    hubs, auth = nx.hits(G_main, max_iter=500, normalized=True)
    metrics["hub"] = pd.Series(hubs)
    metrics["authority"] = pd.Series(auth)
except Exception as e:
    print("HITS computation skipped:", e)

# Clustering
metrics["clustering"] = pd.Series(nx.clustering(G, weight="weight"))

# K-core
core_num = nx.core_number(G)
metrics["kcore"] = pd.Series(core_num)

# Graph-level metrics
assort = nx.degree_pearson_correlation_coefficient(G)
transitivity = nx.transitivity(G)
print(f"Degree assortativity: {assort:.3f}, Transitivity: {transitivity:.3f}")

# Clique statistics
largest_clique = max(nx.find_cliques(G), key=len)
print(f"Largest clique size: {len(largest_clique)}")

# ---------------- Similarity metrics ----------------
# Cosine similarity of adjacency rows
cos_sim = cosine_similarity_manual(A)
mean_cosine = np.mean(cos_sim[np.triu_indices_from(cos_sim, 1)])
print(f"Mean cosine similarity (node adjacency): {mean_cosine:.3f}")

# Jaccard coefficient
jac = list(nx.jaccard_coefficient(G))
jac_mean = np.mean([p for (_, _, p) in jac]) if jac else 0.0
print(f"Mean Jaccard coefficient: {jac_mean:.3f}")

# ---------------- Small-world test ----------------
if nx.is_connected(G):
    L = nx.average_shortest_path_length(G_dist, weight="weight")
else:
    L = np.nan

Cw = nx.average_clustering(G, weight="weight")
Gr = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges())
Lr = nx.average_shortest_path_length(Gr)
Cr = nx.average_clustering(Gr)

if np.isfinite(L) and np.isfinite(Lr) and Lr > 0 and Cr > 0:
    sigma = (Cw / Cr) / (L / Lr)
else:
    sigma = np.nan

print(f"Small-world sigma = {sigma:.3f}")

# ---------------- Power-law fit (fixed) ----------------
deg_vals = np.array(list(dict(G.degree()).values()))
deg_vals = deg_vals[deg_vals > 0]

# Compute empirical degree distribution
unique_k, counts = np.unique(deg_vals, return_counts=True)
pk = counts / counts.sum()

mask = pk > 0
log_k = np.log10(unique_k[mask])
log_pk = np.log10(pk[mask])

if len(log_k) >= 5:
    slope, intercept, r, p, _ = linregress(log_k, log_pk)
    alpha = -slope
    print(f"Power-law exponent (α) ≈ {alpha:.2f}, r = {r:.2f}")
else:
    alpha, r = np.nan, np.nan
    print("Power-law fit skipped (insufficient data points).")

# Plot power-law (log–log)
plt.figure(figsize=(6,4))
plt.scatter(log_k, log_pk, s=20)
if np.isfinite(alpha):
    plt.plot(log_k, intercept + slope * log_k, color='red', label=f'fit α={alpha:.2f}')
plt.xlabel("log10(k)")
plt.ylabel("log10(P(k))")
plt.title("Degree Distribution (Log-Log)")
plt.legend()
plt.tight_layout()
plt.savefig(plots_dir / f"degree_powerlaw_{save_prefix}_{theta_best:.3f}.png", dpi=150)
plt.close()

# ---------------- Homophily (if sector file exists) ----------------
try:
    sectors = pd.read_csv(sector_file)
    sector_map = sectors.set_index("Symbol")["Industry"].to_dict()
    nx.set_node_attributes(
        G,
        {t: sector_map.get(t.replace(".NS", ""), None) for t in G.nodes()},
        "industry"
    )
    assort_sector = nx.attribute_assortativity_coefficient(G, "industry")
    print(f"Sectoral assortativity (homophily): {assort_sector:.3f}")
except Exception as e:
    print("Sector homophily skipped:", e)

# ---------------- Save all metrics ----------------
metrics_file = data_dir / f"metrics_{save_prefix}_{theta_best:.3f}.csv"
metrics.to_csv(metrics_file)
print("Saved extended metrics to", metrics_file)

# Degree histogram
plt.figure(figsize=(6,4))
plt.hist(deg_vals, bins=30, color='steelblue')
plt.xlabel("Degree"); plt.ylabel("Frequency")
plt.title(f"Degree Distribution (theta={theta_best:.3f})")
plt.tight_layout()
plt.savefig(plots_dir / f"degree_hist_{save_prefix}_{theta_best:.3f}.png", dpi=150)
plt.close()

print("\nTop-10 by degree:\n", metrics.sort_values("degree", ascending=False).head(10)[
    ["degree","strength","betweenness","closeness","eigenvector","pagerank","kcore"]
])

print("\nDone. All extended network outputs saved in data/ and data/plots/.")
