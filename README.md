# Resolving Sparsity in Recommendation Systems using SGD

A recommendation-system experiment that applies matrix factorization and stochastic gradient descent (SGD) to sparse Netflix-style user–movie rating data.

## What this repository contains

- `Matrix_SGD.ipynb` — the main experiment notebook, including data preparation, matrix construction, SGD-based matrix factorization, plots, and RMSE evaluation.
- `Datasets/` — prepared rating subsets at multiple sizes plus `movie_titles.csv`.
- `Batch_161 Final Document.docx` — project report.
- `sparsity_Final_ppt[1].pptx` — presentation deck.
- `Installation instructions.docx` — original setup notes.
- `requirements.txt` — Python dependencies required by the notebook.

## Approach

The notebook builds a sparse user–item matrix from rating data, with missing ratings represented as zero values for the factorization stage. A custom matrix-factorization implementation learns latent user and item representations with SGD, user/item biases, and regularization. The notebook then compares reconstructed ratings against held-out values using RMSE.

The experiment also imports Surprise utilities for recommendation-system analysis and uses Pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, and CVXPY during the workflow.

## Dataset options

The repository includes prepared subsets of the Netflix-style ratings data under `Datasets/`, including:

- 1,024 rows
- 5,000 rows
- 10,000 rows
- 25,000 rows
- 50,000 rows
- 75,000 rows
- 100,000 rows
- 150,000 rows
- 175,000 rows
- 200,000 rows

The interactive notebook currently offers 1,024, 10,000, 25,000, 75,000, 100,000, and 200,000-row choices.

## Local setup

```bash
git clone https://github.com/sasi921/Resolving-Sparsity-in-Recommendation-Systems-using-SGD.git
cd Resolving-Sparsity-in-Recommendation-Systems-using-SGD
python -m venv .venv
```

Activate the virtual environment, then install the notebook dependencies:

```bash
pip install -r requirements.txt
jupyter notebook Matrix_SGD.ipynb
```

### Data-path note

The notebook was originally authored in a Colab-style environment and its data-loading cells use bare filenames such as `feasible_data_25000.txt` and `movie_titles (2).csv`, while this repository stores the files under `Datasets/` and names the title file `movie_titles.csv`.

When running locally, update those paths in the notebook to point to the repository files, for example:

```python
pd.read_csv("Datasets/feasible_data_25000.txt", ...)
pd.read_csv("Datasets/movie_titles.csv", ...)
```

This keeps the original experiment intact while making the repository layout explicit to new contributors.

## Main dependencies

- NumPy
- Pandas
- SciPy
- scikit-learn
- Matplotlib
- Seaborn
- scikit-surprise
- CVXPY
- Jupyter

## Evaluation

The notebook performs a random holdout by masking a portion of the rating matrix, trains the matrix-factorization model, reconstructs predictions, and reports RMSE for the held-out comparison. Because the split is random and no fixed seed is set in the original notebook, exact RMSE values can vary between executions.

## Why this project matters

Sparse interaction data is a core challenge in recommendation systems. This project demonstrates the mechanics of transforming sparse ratings into a latent-factor representation, optimizing it with SGD, and measuring reconstruction quality on held-out data.
