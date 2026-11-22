from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from imblearn.over_sampling import SMOTE
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

BASE = Path(__file__).resolve().parent
FILES = {"KC1": "kc1.csv", "PC1": "pc1.csv", "CM1": "cm1.csv", "JM1": "jm1_clean.csv"}
LABELS = {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}

st.set_page_config(page_title="Explainable Defect Prediction", layout="wide")
st.title("Explainable Software Defect Prediction")
st.caption("Minor project: Random Forest with SHAP and LIME explanations on PROMISE/NASA datasets")


@st.cache_data
def load(name):
    df = pd.read_csv(BASE / "data" / FILES[name])
    df.columns = [c.strip() for c in df.columns]
    if float(df.iloc[0, 0]) == 1.1:                       # remove the two junk rows
        df = df.iloc[2:].reset_index(drop=True)
    label = "defects" if "defects" in df.columns else "problems"
    y = df[label].astype(str).str.strip().str.lower().map(LABELS)
    X = df.drop(columns=[label]).apply(pd.to_numeric, errors="coerce")
    keep = ~X.assign(y=y).duplicated()                    # remove duplicate rows
    return X[keep].reset_index(drop=True), y[keep].astype(int).reset_index(drop=True), len(df)


@st.cache_resource
def train(name):
    X, y, n_raw = load(name)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    imp = SimpleImputer(strategy="median").fit(X_tr)
    sc = MinMaxScaler().fit(imp.transform(X_tr))
    Xtr_s = sc.transform(imp.transform(X_tr))
    Xs, ys = SMOTE(random_state=42).fit_resample(Xtr_s, y_tr)          # SMOTE on training data only
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(Xs, ys)
    lime_exp = LimeTabularExplainer(Xtr_s, feature_names=list(X.columns),
                                    class_names=["not defective", "defective"], mode="classification",
                                    discretize_continuous=False, random_state=42)
    return {"rf": rf, "imp": imp, "sc": sc, "shap": shap.TreeExplainer(rf), "lime": lime_exp,
            "X_tr": X_tr, "X_te": X_te.reset_index(drop=True), "y_te": y_te.reset_index(drop=True),
            "n_raw": n_raw, "n": len(X), "n_def": int(y.sum())}


def shap_top(m, scaled_row, cols, k=8):
    sv = m["shap"].shap_values(scaled_row, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    s = pd.Series(sv[0], index=cols)
    return s.reindex(s.abs().sort_values(ascending=False).index)[:k]


def lime_top(m, scaled_row, k=8):
    e = m["lime"].explain_instance(scaled_row[0], m["rf"].predict_proba, num_features=k, num_samples=1000)
    s = pd.Series({name: w for name, w in e.as_list()})
    return s.reindex(s.abs().sort_values(ascending=False).index)[:k]


def bar(s, title):
    s = s[::-1]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.barh(s.index, s.values, color=["#d9534f" if v > 0 else "#5b8def" for v in s.values])
    ax.set_title(title)
    ax.set_xlabel("red: towards defective | blue: towards not defective")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_result(m, row_df, true_label=None):
    scaled = m["sc"].transform(m["imp"].transform(row_df))
    p = m["rf"].predict_proba(scaled)[0, 1]
    st.subheader(f"Predicted defect probability: {p:.2f}  ->  "
                 f"{'DEFECTIVE' if p >= 0.5 else 'not defective'}")
    if true_label is not None:
        st.write(f"True label in the dataset: **{'defective' if true_label == 1 else 'not defective'}**")
    c1, c2 = st.columns(2)
    with c1:
        bar(shap_top(m, scaled, row_df.columns), "SHAP: why this prediction")
    with c2:
        bar(lime_top(m, scaled), "LIME: why this prediction")
    st.caption("The probability is the model's confidence, not the real chance of a bug. "
               "SHAP and LIME can disagree, which is one of the findings of this project.")


name = st.sidebar.selectbox("Dataset", list(FILES))
m = train(name)

st.sidebar.markdown(f"**{name}**: {m['n_raw']} modules in the file, {m['n']} after removing duplicates, "
                    f"{m['n_def']} defective")

pred = m["rf"].predict(m["sc"].transform(m["imp"].transform(m["X_te"])))
c1, c2, c3 = st.columns(3)
c1.metric("Test precision", f"{precision_score(m['y_te'], pred, zero_division=0):.2f}")
c2.metric("Test recall", f"{recall_score(m['y_te'], pred, zero_division=0):.2f}")
c3.metric("Test F1 (defective)", f"{f1_score(m['y_te'], pred, zero_division=0):.2f}")
st.info("The model is far from perfect: it misses some defective modules and raises some false alarms.")

tab1, tab2, tab3 = st.tabs(["Pick a test module", "Enter your own metrics", "Project results"])

with tab1:
    idx = st.slider("Test module number", 0, len(m["X_te"]) - 1, 0)
    row = m["X_te"].iloc[[idx]]
    st.dataframe(row.T.rename(columns={idx: "value"}), height=250)
    show_result(m, row, int(m["y_te"].iloc[idx]))

with tab2:
    st.write("Change the metrics of a module and see what the model predicts.")
    cols = st.columns(4)
    vals = {}
    for k, f in enumerate(m["X_tr"].columns):
        vals[f] = cols[k % 4].number_input(f, value=float(m["X_tr"][f].median()), format="%.2f")
    if st.button("Check this module"):
        show_result(m, pd.DataFrame([vals]))

with tab3:
    st.write("Results saved by the notebook (`results/` folder).")
    for title, fname in [("Baselines: Random Forest and XGBoost", "baseline_results.csv"),
                         ("RSN (our reconstruction), 5-fold CV", "rsn_cv.csv"),
                         ("SHAP vs LIME agreement, Random Forest", "shap_lime_agreement.csv"),
                         ("SHAP vs LIME agreement, RSN (KC1)", "shap_lime_agreement_rsn.csv")]:
        path = BASE / "results" / fname
        if path.exists():
            st.markdown(f"**{title}**")
            st.dataframe(pd.read_csv(path), hide_index=True)
    for img in ["shap_summary_jm1.png", "shap_local_jm1.png", "lime_local_jm1.png"]:
        if (BASE / "results" / img).exists():
            st.image(str(BASE / "results" / img), caption=img, width=520)