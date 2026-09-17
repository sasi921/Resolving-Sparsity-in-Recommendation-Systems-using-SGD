# Resolving Sparsity in Recommendation Systems using SGD

A recommendation-system experiment that applies **matrix factorization optimized with stochastic gradient descent (SGD)** to sparse user–item rating data. The repository includes the executable Jupyter notebook, project documentation, presentation material, and datasets used for the experiment.

## Why this project matters

Real recommendation data is sparse: most users interact with only a small fraction of available items. Matrix factorization learns compact latent representations for users and items so that missing preferences can be estimated from observed ratings.

This project demonstrates:

- building a user–item ratings representation from sparse data;
- learning latent user and item factors with SGD;
- reconstructing/predicting missing ratings from those factors;
- evaluating recommendation error experimentally;
- communicating an ML experiment through code, documentation, and results.

## Repository contents

| Path | Purpose |
| --- | --- |
| `Matrix_SGD.ipynb` | Main executable experiment and analysis |
| `Datasets/` | Data used by the notebook |
| `Batch_161 Final Document.docx` | Detailed project report |
| `sparsity_Final_ppt[1].pptx` | Project presentation |
| `Installation instructions.docx` | Original setup notes |

> The earlier README referenced `src/`, `data/`, `docs/`, and `notebooks/` directories that are not present in the repository. This guide intentionally documents the files that actually exist so a reviewer can navigate the project without guessing.

## Quick start

1. Clone the repository:

```bash
git clone https://github.com/sasi921/Resolving-Sparsity-in-Recommendation-Systems-using-SGD.git
cd Resolving-Sparsity-in-Recommendation-Systems-using-SGD
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

3. Install the libraries used by the notebook:

```bash
python -m pip install --upgrade pip
pip install jupyter numpy pandas scikit-learn matplotlib
```

4. Start Jupyter and open `Matrix_SGD.ipynb`:

```bash
jupyter notebook
```

Run the notebook cells in order. Dataset paths are relative to the checked-in `Datasets/` directory; if you substitute another dataset, preserve the rating fields expected by the notebook or update the loading cells accordingly.

## Method at a glance

Given a sparse ratings matrix `R`, matrix factorization approximates it with lower-dimensional user and item factors:

```text
R ≈ P × Qᵀ
```

SGD iteratively updates `P` and `Q` to reduce prediction error on observed ratings. This avoids treating every missing user–item pair as a known rating and gives the model a compact representation of preference patterns.

## Tech stack

- Python
- Jupyter Notebook
- NumPy
- Pandas
- scikit-learn
- Matplotlib

## Reproducibility notes

The repository preserves the original experiment artifacts. Exact metrics can depend on dataset selection, preprocessing, random initialization, hyperparameters, and train/test methodology. For that reason, results should be interpreted together with the notebook configuration rather than as a universal benchmark.

For a production-grade extension, the next useful steps would be to extract the factorization logic into testable Python modules, pin dependencies, set deterministic random seeds, and report a clearly defined held-out evaluation protocol.

## Project scope

This is an educational/research implementation intended to demonstrate recommendation-system sparsity and SGD-based matrix factorization. It is not presented as a production recommender service.