"""
threshold_parameter_study_extended.py
----------------------------------------------------------
Performs a full parameter study for correlation-thresholded
financial networks, consistent with the main analysis.

It replicates all key node- and network-level metrics from
`step3_threshold_network.py` (degree, strength, betweenness,
closeness, eigenvector, Katz, PageRank, HITS, clustering, k-core,
assortativity, transitivity, power-law exponent, and homophily),
computed over a range of thresholds.

Input:
  - data/corr_matrix.csv
  - ind_nifty500list.csv (for sector homophily)

Output:
  - data/metrics_by_theta_extended.csv
  - data/plots/parameter_study_extended/*.png

Configurable flags allow skipping heavy metrics (e.g., betweenness, HITS)
for faster runtime.
----------------------------------------------------------
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress
import time

# ---------------- CONFIG ----------------
DATA_DIR = Path("data")
PLOTS_DIR = DATA_DIR / "plots" / "parameter_study_extended"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CORR_FILE = DATA_DIR / "corr_matrix.csv"
SECTOR_FILE = "ind_nifty500list.csv"

THETA_GRID = np.arange(0.05, 0.51, 0.02)   # Adjust as needed
USE_ABS = True
SAVE_NODE_METRICS = False

# Toggle heavy computations
COMPUTE_BETWEENNESS = True
COMPUTE_KATZ = True
COMPUTE_HITS = True

# ---------------- LOAD ----------------
print("Loading correlation matrix:", CORR_FILE)
C = pd.read_csv(CORR_FILE, index_col=0)
tickers = C.index.tolist()
n = len(tickers)
np.fill_diagonal(C.values, 1.0)
print(f"Loaded correlation matrix for {n} tickers. Shape: {C.shape}")

# Load sector info for homophily
try:
    sectors = pd.read_csv(SECTOR_FILE)
    sector_map = sectors.set_index("Symbol")["Industry"].to_dict()
except Exception:
    sector_map = {}

# ---------------- HELPERS ----------------
def build_weighted_adj(C, theta):
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            val = abs(C.iat[i, j]) if USE_ABS else C.iat[i, j]
            if val >= theta:
                W[i, j] = val
                W[j, i] = val
    return W

def cosine_similarity_manual(matrix):
    """Compute cosine similarity safely, ignoring zero rows."""
    matrix = np.where(np.all(matrix == 0, axis=1, keepdims=True), 1e-12, matrix)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normalized = matrix / norms
    sim = np.dot(normalized, normalized.T)
    return sim

# ---------------- MAIN ----------------
summary_records = []

for theta in THETA_GRID:
    t0 = time.time()
    W = build_weighted_adj(C, theta)
    G = nx.from_numpy_array(W)
    mapping = {i: tickers[i] for i in range(n)}
    G = nx.relabel_nodes(G, mapping)
    print(f"\n--- Theta = {theta:.3f} ---")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    if G.number_of_edges() == 0:
        continue

    # degree & strength
    degree = dict(G.degree())
    strength = dict(G.degree(weight="weight"))

    # distance graph (for path-based metrics)
    G_dist = nx.Graph()
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 0)
        if w > 0:
            G_dist.add_edge(u, v, weight=1.0 / w)

    # betweenness
    if COMPUTE_BETWEENNESS:
        try:
            bet = nx.betweenness_centrality(G_dist, weight="weight", normalized=True)
        except Exception:
            bet = {node: 0.0 for node in G.nodes()}
    else:
        bet = {node: 0.0 for node in G.nodes()}

    # closeness
    try:
        clos = nx.closeness_centrality(G_dist, distance="weight")
    except Exception:
        clos = {node: 0.0 for node in G.nodes()}

    # eigenvector, Katz, PageRank, HITS
    largest_cc = max(nx.connected_components(G), key=len)
    G_main = G.subgraph(largest_cc).copy()
    eig = nx.eigenvector_centrality_numpy(G_main, weight="weight")
    katz = nx.katz_centrality_numpy(G_main, alpha=0.005, beta=1.0) if COMPUTE_KATZ else {n: 0.0 for n in G_main}
    pagerank = nx.pagerank(G_main, alpha=0.85, weight="weight")
    if COMPUTE_HITS:
        try:
            hubs, auth = nx.hits(G_main, max_iter=500)
        except Exception:
            hubs, auth = {}, {}
    else:
        hubs, auth = {}, {}

    clustering = nx.clustering(G, weight="weight")
    kcore = nx.core_number(G)
    assort = nx.degree_pearson_correlation_coefficient(G)
    trans = nx.transitivity(G)

    # cosine similarity (overall adjacency structure)
    cos_sim = cosine_similarity_manual(W)
    mean_cosine = np.mean(cos_sim[np.triu_indices_from(cos_sim, 1)])

    # power-law fit (safe log10)
    deg_vals = np.array(list(degree.values()))
    deg_vals = deg_vals[deg_vals > 0]
    if len(deg_vals) > 1:
        pk = np.bincount(deg_vals)[1:] / len(deg_vals)
        pk = pk[pk > 0]  # avoid log(0)
        log_k = np.log10(np.arange(1, len(pk) + 1))
        log_pk = np.log10(pk)
        m = min(len(log_k), len(log_pk))
        slope, intercept, r, _, _ = linregress(log_k[:m], log_pk[:m])
        alpha = -slope
    else:
        alpha, r = np.nan, np.nan

    # sectoral homophily
    nx.set_node_attributes(G, {t: sector_map.get(t.replace(".NS", ""), None) for t in G.nodes()}, "industry")
    try:
        assort_sector = nx.attribute_assortativity_coefficient(G, "industry")
    except Exception:
        assort_sector = np.nan

    # summary record
    summary_records.append({
        "theta": theta,
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": np.mean(list(degree.values())),
        "avg_strength": np.mean(list(strength.values())),
        "assortativity": assort,
        "transitivity": trans,
        "mean_cosine": mean_cosine,
        "powerlaw_alpha": alpha,
        "powerlaw_r": r,
        "sector_assortativity": assort_sector
    })

    if SAVE_NODE_METRICS:
        node_df = pd.DataFrame({
            "degree": pd.Series(degree),
            "strength": pd.Series(strength),
            "betweenness": pd.Series(bet),
            "closeness": pd.Series(clos),
            "eigenvector": pd.Series(eig),
            "katz": pd.Series(katz),
            "pagerank": pd.Series(pagerank),
            "hub": pd.Series(hubs),
            "authority": pd.Series(auth),
            "clustering": pd.Series(clustering),
            "kcore": pd.Series(kcore)
        })
        node_df.to_csv(DATA_DIR / f"node_metrics_theta_{theta:.3f}.csv")

    t1 = time.time()
    print(f"Completed theta={theta:.3f} in {t1 - t0:.2f}s")

# ---------------- SAVE ----------------
summary_df = pd.DataFrame(summary_records)
summary_file = DATA_DIR / "metrics_by_theta_extended.csv"
summary_df.to_csv(summary_file, index=False)
print(f"\nSaved extended parameter study summary to {summary_file}")

# ---------------- PLOTS ----------------
def plot_metric(x, y, label, title, fname):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker='o', linewidth=1)
    plt.xlabel("θ"); plt.ylabel(label)
    plt.title(title)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150)
    plt.close()

plot_metric(summary_df["theta"], summary_df["edges"], "Edges", "Edge count vs θ", "edges_vs_theta.png")
plot_metric(summary_df["theta"], summary_df["density"], "Density", "Density vs θ", "density_vs_theta.png")
plot_metric(summary_df["theta"], summary_df["assortativity"], "Assortativity", "Degree assortativity vs θ", "assortativity_vs_theta.png")
plot_metric(summary_df["theta"], summary_df["transitivity"], "Transitivity", "Transitivity vs θ", "transitivity_vs_theta.png")
plot_metric(summary_df["theta"], summary_df["sector_assortativity"], "Sector homophily", "Sectoral assortativity vs θ", "sector_assortativity_vs_theta.png")
plot_metric(summary_df["theta"], summary_df["powerlaw_alpha"], "α (power-law exponent)", "Power-law exponent vs θ", "powerlaw_vs_theta.png")

print("\nSaved all extended parameter study plots to", PLOTS_DIR)
print("Done.")
