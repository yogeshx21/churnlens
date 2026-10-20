"""ChurnLens engine: prediction, SHAP, LIME, trust score, what-if."""
import numpy as np
import pandas as pd
import joblib

from common import encode, MULTI

BUNDLE_PATH = "models/bundle.joblib"


def load_bundle():
    import os
    if not os.path.exists(BUNDLE_PATH):          # first run on a fresh machine / cloud: train quickly
        import subprocess, sys
        subprocess.run([sys.executable, "train.py", "--fast"], check=True)
    return joblib.load(BUNDLE_PATH)


def risk_level(p):
    return "low" if p < 0.30 else ("medium" if p < 0.60 else "high")


def predict(bundle, raw_df):
    X = encode(raw_df, bundle["columns"])
    return float(bundle["model"].predict_proba(X)[:, 1][0])


class Explainers:
    """SHAP (primary) + LIME (independent check) + nearest neighbours."""

    def __init__(self, bundle):
        import shap
        from lime.lime_tabular import LimeTabularExplainer
        from sklearn.neighbors import NearestNeighbors

        self.shap = shap
        self.b = bundle
        self.cols = bundle["columns"]
        self.n = len(self.cols)
        self.model = bundle["model"]
        Xtr = bundle["X_train"]
        bg = Xtr.sample(min(100, len(Xtr)), random_state=0)
        masker = shap.maskers.Independent(bg.values, max_samples=50)
        self.explainer = shap.Explainer(self._f, masker, algorithm="permutation")
        self.lime = LimeTabularExplainer(
            Xtr.values, feature_names=self.cols, class_names=["stay", "churn"],
            mode="classification", discretize_continuous=True, random_state=0)
        self.mu = Xtr.mean().values
        self.sd = Xtr.std().replace(0, 1).values
        self.nn = NearestNeighbors(n_neighbors=5).fit((Xtr.values - self.mu) / self.sd)

    def _f(self, d):
        return self.model.predict_proba(pd.DataFrame(np.asarray(d), columns=self.cols))[:, 1]

    def shap_values(self, X):
        sv = self.explainer(X.values, max_evals=10 * self.n)
        self._base = float(np.ravel(np.asarray(sv.base_values))[0])  # average-customer risk
        return np.asarray(sv.values)

    def lime_top(self, x, k=5):
        exp = self.lime.explain_instance(
            x, lambda a: self.model.predict_proba(pd.DataFrame(a, columns=self.cols)),
            num_features=k, num_samples=1500)
        return [i for i, _ in exp.as_map()[1]]

    def explain(self, bundle_raw_X):
        """bundle_raw_X: 1-row encoded DataFrame. Returns dict with shap, trust etc."""
        x = bundle_raw_X.values[0]
        _, idx = self.nn.kneighbors(((x - self.mu) / self.sd).reshape(1, -1))
        neigh = self.b["X_train"].iloc[idx[0]]
        allX = pd.concat([bundle_raw_X, neigh], ignore_index=True)
        sv = self.shap_values(allX)
        s0, sn = sv[0], sv[1:]
        shap_top = list(np.argsort(-np.abs(s0))[:5])
        lime_top = self.lime_top(x, 5)
        overlap = len(set(shap_top) & set(lime_top)) / 5.0
        cos = []
        for v in sn:
            den = np.linalg.norm(s0) * np.linalg.norm(v)
            cos.append(float(np.dot(s0, v) / den) if den > 0 else 0.0)
        neigh_sim = max(0.0, float(np.mean(cos)))
        trust = round(100 * (0.5 * overlap + 0.5 * neigh_sim))
        return {
            "shap": s0, "base": self._base, "overlap": overlap, "neigh_sim": neigh_sim, "trust": trust,
            "cols": self.cols, "values": x,
        }

    def global_importance(self):
        sv = self.shap_values(self.b["X_test_sample"])
        s = pd.Series(np.abs(sv).mean(axis=0), index=self.cols).sort_values(ascending=False)
        return s


def top_factors(res, k=3):
    s, cols, vals = res["shap"], res["cols"], res["values"]
    def present(i):  # skip one-hot columns the customer does NOT have
        return not (cols[i].split("_", 1)[0] in MULTI and "_" in cols[i] and vals[i] == 0)
    order = np.argsort(-s)
    up = [(cols[i], float(s[i])) for i in order if s[i] > 0 and present(i)][:k]
    order = np.argsort(s)
    down = [(cols[i], float(s[i])) for i in order if s[i] < 0 and present(i)][:k]
    return up, down


# ---------- What-if (actionable, business-controlled features only) ----------
def whatif(bundle, raw):
    """raw: dict of raw feature values. Returns list of scenarios ranked by risk reduction."""
    base = predict(bundle, pd.DataFrame([raw]))
    scen = []
    internet = raw["InternetService"] != "No"

    def add(key, changes):
        changes = {k: v for k, v in changes.items() if raw[k] != v}
        if not changes:
            return
        new = {**raw, **changes}
        p = predict(bundle, pd.DataFrame([new]))
        scen.append({"key": key, "changes": changes, "new_p": p, "delta": base - p})

    if raw["Contract"] == "Month-to-month":
        add("contract_1y", {"Contract": "One year"})
        add("contract_2y", {"Contract": "Two year"})
    elif raw["Contract"] == "One year":
        add("contract_2y", {"Contract": "Two year"})
    if internet:
        add("techsupport", {"TechSupport": "Yes"})
        add("security", {"OnlineSecurity": "Yes"})
    if raw["PaymentMethod"] in ("Electronic check", "Mailed check"):
        add("autopay", {"PaymentMethod": "Credit card (automatic)"})
    combo = {}
    if raw["Contract"] == "Month-to-month":
        combo["Contract"] = "One year"
    if internet:
        combo.update({"TechSupport": "Yes", "OnlineSecurity": "Yes"})
    if raw["PaymentMethod"] in ("Electronic check", "Mailed check"):
        combo["PaymentMethod"] = "Credit card (automatic)"
    if len([k for k in combo if raw[k] != combo[k]]) >= 2:
        add("combo", combo)
    scen.sort(key=lambda d: -d["delta"])
    return base, scen
