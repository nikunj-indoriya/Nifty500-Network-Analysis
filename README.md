# Network Science Analysis of the Indian Stock Market (NIFTY 500)

This repository contains the full pipeline used in the course project for  
**ECS414: Network Science — Theory and Applications**, IISER Bhopal.

The project constructs and analyses a complex network of Indian stocks using
Pearson correlation of daily log returns.  
It implements:

- Price fetching (Yahoo Finance)
- Cleaning and preprocessing
- Log returns and correlation matrix computation
- Thresholded correlation network construction
- Optimal threshold selection using the consistency function \(G(\theta)\)
- Full network metrics at optimal threshold
- Parameter study across many values of \(\theta\)
- Plots and report written in LaTeX

---

## Repository Structure

```

Nifty500-Network-Analysis/
│
├── data/
│   ├── prices_clean.csv
│   ├── returns.csv
│   ├── corr_matrix.csv
│   ├── metrics_theta_0.230.csv
│   ├── metrics_by_theta_extended.csv
│   └── plots/      
│
├── fetch_prices_from_csv.py
├── clean_returns_corr.py
├── threshold_network.py
├── threshold_parameter_study.py
├── README.md
└── .gitignore

````

**Note:**  
`data/prices_raw.csv` is intentionally not uploaded because of size.  
You can regenerate it using the fetch script.

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nikunj-indoriya/Nifty500-Network-Analysis.git
cd Nifty500-Network-Analysis
````

2. Create a virtual environment (example):

```bash
python -m venv netsci
source netsci/bin/activate   # Linux / Mac
netsci\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```
---

## Usage Instructions

### 1. Fetch price data

This downloads adjusted close prices for NIFTY 500 stocks from Yahoo Finance
and saves them into `data/prices_raw.csv`.

```bash
python fetch_prices_from_csv.py
```

Output:

* `data/prices_raw.csv`
* `data/failed_tickers.txt` (if any ticker fails)

---

### 2. Clean prices, compute returns, and correlation matrix

```bash
python clean_returns_corr.py
```

Outputs:

* `data/prices_clean.csv`
* `data/returns.csv`
* `data/corr_matrix.csv`
* correlation distribution plot
* sample heatmap plot

---

### 3. Construct network at optimal threshold ( \hat{\theta} )

```bash
python threshold_network.py
```

This computes:

* Consistency function (G(\theta))
* Optimal threshold (≈ 0.23)
* Full network metrics
* Degree distribution plot
* Power-law log–log plot

Outputs saved in:

* `data/metrics_theta_0.230.csv`
* `data/plots/`

---

### 4. Parameter study across thresholds

```bash
python threshold_parameter_study.py
```

This recomputes network metrics across θ ∈ [0.05, 0.50] and saves:

* `data/metrics_by_theta_extended.csv`
* density vs θ plot
* assortativity vs θ plot
* power-law exponent vs θ plot
* sector homophily vs θ plot

Useful for checking robustness of results.

---

## Notes

* Raw price data is not uploaded. Regenerate using the fetch script.
* Virtual environments are excluded using `.gitignore`.
* All results in the PDF can be reproduced using the scripts in this repository.

---

## Contact

**Author:** Nikunj Indoriya
**Institute:** IISER Bhopal
For questions: open an issue or contact directly.
