# explainable-software-defect-prediction
“Explainable software defect prediction on PROMISE/NASA datasets using SHAP and LIME.”
# Explainable Software Defect Prediction (SHAP and LIME)

Minor project built around the paper *"Explainable deep learning for software defect prediction:
a residual-shuffle network approach with SHAP and LIME"* (Talatapeh et al., International Journal
of Data Science and Analytics, 2026, 22:24).

We evaluate classical baselines (Random Forest, XGBoost) and our own reconstruction of the paper's
residual-shuffle network (RSN) on PROMISE/NASA defect datasets, and we compare two post-hoc
explanation methods, SHAP and LIME. The RSN is a reconstruction from the paper's description, not
the authors' code, which is not publicly available.

## Datasets

| Dataset | Source | Modules | Modules after removing duplicates |
|---|---|---|---|
| JM1 | OpenML (ID 1053) | 10,883 | 8,910 |
| KC1 | PROMISE/NASA (GitHub mirror) | 2,107 | 1,210 |
| PC1 | PROMISE/NASA (GitHub mirror) | 1,107 | 952 |
| CM1 | PROMISE/NASA (GitHub mirror) | 496 | 440 |

Each dataset has 21 static code metrics (McCabe, Halstead, LOC) and a binary defect label.
Two header-artifact rows present in the raw files are removed by the loader. All model
comparisons below use data with duplicate rows removed, unless stated otherwise.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/01_data_exploration.ipynb` and run the cells in order. Outputs are saved to
`results/`. All random seeds are fixed to 42.

## Method

- Preprocessing: median imputation, min-max scaling and SMOTE (training data only, never on
  test data).
- Baselines: Random Forest and XGBoost with stratified 5-fold cross-validation, reported with
  and without duplicate rows.
- RSN (our reconstruction): the 21 metrics are zero-padded to a 5x5 grid (1 channel), followed by
  a 3x3 convolution stem and 3 residual-shuffle units implementing equations (1) to (6) of the
  paper (channel split with alpha = 0.5, GELU, depthwise 3x3 convolution, channel shuffle with
  g = 2, residual 1x1 convolution), global average pooling and a small classifier head with
  dropout 0.5. The model has 8,882 parameters. Training follows the paper's description: AdamW
  with learning rate 3e-4 and cosine decay, weight decay 0.01, batch size 64, at most 100 epochs,
  early stopping on validation loss (patience 10) using 10% of the training fold, evaluated with
  stratified 5-fold cross-validation.
- Explanations: SHAP and LIME on 30 predicted-defective test modules per dataset. For Random
  Forest (JM1, KC1) we use TreeSHAP; for RSN (KC1) we use KernelSHAP (model-agnostic). LIME is
  run with discretized and with continuous features.

## Results

### Baselines (F1 of the defective class, %, with duplicates / without duplicates)

| Dataset | Random Forest | XGBoost |
|---|---|---|
| JM1 | 42.2 / 39.0 | 37.1 / 37.1 |
| KC1 | 46.7 / 42.9 | 46.2 / 42.0 |
| PC1 | 44.7 / 35.4 | 46.2 / 36.7 |
| CM1 | 21.5 / 16.8 | 31.4 / 23.8 |

Accuracy is misleading on these imbalanced datasets, so F1 and recall are reported.

### RSN vs baselines (duplicates removed, 5-fold CV, defective class, %)

| Dataset | Model | Precision | Recall | F1 |
|---|---|---|---|---|
| KC1 | Random Forest | 46.2 ± 3.9 | 40.1 ± 3.0 | 42.9 ± 3.0 |
| KC1 | XGBoost | 48.6 ± 5.3 | 36.9 ± 4.6 | 42.0 ± 4.9 |
| KC1 | RSN | 40.1 ± 2.2 | 50.7 ± 7.0 | 44.6 ± 3.8 |
| JM1 | Random Forest | 44.8 ± 1.2 | 34.5 ± 0.7 | 39.0 ± 0.6 |
| JM1 | XGBoost | 48.3 ± 2.8 | 30.2 ± 1.6 | 37.1 ± 1.5 |
| JM1 | RSN | 39.8 ± 2.3 | 43.3 ± 8.8 | 40.7 ± 4.0 |

The RSN F1 is comparable to the baselines (the differences are within one standard deviation).
The RSN reaches higher recall and lower precision than the baselines on both datasets. RSN was
not run on PC1 and CM1.

### SHAP vs LIME agreement, Random Forest (30 modules per dataset)

| Dataset | LIME setting | Surrogate R² | Top-5 overlap | Sign agreement |
|---|---|---|---|---|
| JM1 | discretized | 0.13 | 35% | 38% |
| JM1 | continuous | 0.12 | 47% | 81% |
| KC1 | discretized | 0.12 | 27% | 95% |
| KC1 | continuous | 0.25 | 35% | 71% |

Top SHAP features (Random Forest): JM1: iv(g), loc, lOBlank, branchCount, v(g). KC1:
total_Opnd, uniq_Opnd, lOBlank, l, d. Plots for JM1 are in `results/`.

### SHAP vs LIME agreement, RSN (KC1, 30 modules, one split)

| LIME setting | Surrogate R² | Top-5 overlap | Sign agreement |
|---|---|---|---|
| discretized | 0.42 | 70% | 100% |
| continuous | 0.65 | 65% | 95% |

Top KernelSHAP features for the RSN on KC1: d, uniq_Op, i, uniq_Opnd, lOCode. RSN test F1 on
this single 80/20 split is 0.493 (the 5-fold mean is 44.6 ± 3.8). SHAP and LIME agree more on
the RSN than on the Random Forest, but the SHAP method also differs (KernelSHAP vs TreeSHAP), so
the two agreement tables are not directly comparable and we cannot attribute the difference to
the model alone.

## Limitations

- F1 scores are far below the 98% reported in the paper. We did not find the reason. Our RSN is
  a reconstruction: the paper does not specify how tabular metrics are mapped to the CNN input,
  and its reported model size (1.42 M parameters) is much larger than ours.
- No outlier removal or correlation filter was applied (both are in the paper's pipeline).
- RSN results cover KC1 and JM1 only, and its explanations only KC1.
- Explanation analyses use one train/test split and 30 modules per dataset. For Random Forest,
  LIME surrogate R² is low (0.12 to 0.25) and sign agreement varied between runs, so LIME
  weights should be treated as noisy.
- Important features differ between datasets and between models, so explanations do not
  transfer across them.

## Reference

Talatapeh, S.K., Ebrahimi Dishabi, M.R., Pezeshki, H., Madadi, H.: Explainable deep learning
for software defect prediction: a residual-shuffle network approach with SHAP and LIME.
Int. J. Data Sci. Anal. 22, 24 (2026). https://doi.org/10.1007/s41060-025-00987-6
