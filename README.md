# explainable-software-defect-prediction
“Explainable software defect prediction on PROMISE/NASA datasets using SHAP and LIME.”
# Explainable Software Defect Prediction (SHAP and LIME)

Minor project reproducing and testing ideas from *"Explainable deep learning for software
defect prediction: a residual-shuffle network approach with SHAP and LIME"*
(Talatapeh et al., International Journal of Data Science and Analytics, 2026, 22:24).

We use classical models (Random Forest, XGBoost) on the PROMISE/NASA defect datasets and
compare two post-hoc explanation methods, SHAP and LIME. The residual-shuffle network (RSN)
from the paper is **not** implemented here.

## Datasets

| Dataset | Source | Modules | Modules after removing duplicates |
|---|---|---|---|
| JM1 | OpenML (ID 1053) | 10,883 | 8,910 |
| KC1 | PROMISE/NASA (GitHub mirror) | 2,107 | 1,210 |
| PC1 | PROMISE/NASA (GitHub mirror) | 1,107 | 952 |
| CM1 | PROMISE/NASA (GitHub mirror) | 496 | 440 |

Each dataset has 21 static code metrics (McCabe, Halstead, LOC) and a binary defect label.
Two header-artifact rows present in the raw files are removed by the loader.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/01_data_exploration.ipynb` and run all cells. Outputs are saved to `results/`.

## Method

- Preprocessing: median imputation, min-max scaling, SMOTE (training folds only).
- Baselines: Random Forest and XGBoost, stratified 5-fold CV, reported both with and
  without duplicate rows.
- Explanations (JM1 and KC1): 80/20 stratified split, Random Forest, SHAP (TreeExplainer)
  and LIME on 30 predicted-defective test modules.

## Results

### Baselines (F1 of the defective class, %, with duplicates / without duplicates)

| Dataset | Random Forest | XGBoost |
|---|---|---|
| JM1 | 42.2 / 39.0 | 37.1 / 37.1 |
| KC1 | 46.7 / 42.9 | 46.2 / 42.0 |
| PC1 | 44.7 / 35.4 | 46.2 / 36.7 |
| CM1 | 21.5 / 16.8 | 31.4 / 23.8 |

Accuracy is misleading on these imbalanced datasets, so F1 and recall are reported.

### SHAP vs LIME agreement (30 modules per dataset)

| Dataset | LIME setting | Surrogate R² | Top-5 overlap | Sign agreement |
|---|---|---|---|---|
| JM1 | discretized | 0.13 | 35% | 38% |
| JM1 | continuous | 0.12 | 47% | 81% |
| KC1 | discretized | 0.12 | 27% | 95% |
| KC1 | continuous | 0.25 | 35% | 71% |

Top SHAP features: JM1: iv(g), loc, lOBlank, branchCount, v(g). KC1: total_Opnd,
uniq_Opnd, lOBlank, l, d. Plots are in `results/`.

## Limitations

- F1 scores are far below the 98% reported in the paper. We did not find the reason.
- No outlier removal or correlation filter was applied (both are in the paper's pipeline).
- LIME surrogate R² is low (0.12 to 0.25) and sign agreement varied between runs, so
  LIME weights should be treated as noisy.
- SHAP/LIME analysis uses one train/test split and 30 modules per dataset.
- Important features differ between JM1 and KC1, so explanations do not transfer
  across datasets.

## Reference

Talatapeh, S.K., Ebrahimi Dishabi, M.R., Pezeshki, H., Madadi, H.: Explainable deep learning
for software defect prediction: a residual-shuffle network approach with SHAP and LIME.
Int. J. Data Sci. Anal. 22, 24 (2026). https://doi.org/10.1007/s41060-025-00987-6
