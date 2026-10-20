"""ChurnLens decision layer: revenue at risk, offer ROI, trust-gated action, batch scoring,
customer-level fairness. Pure Python (no Streamlit) so it is easy to test."""
import os
import numpy as np
import pandas as pd

from common import DATA_PATH, MULTI, RAW_FEATURES, encode

DEFAULT_CFG = {
    "horizon": 12,        # months of revenue we look ahead
    "disc1": 10.0,        # % monthly discount for a One-year contract offer
    "disc2": 15.0,        # % monthly discount for a Two-year contract offer
    "tech_cost": 15.0,    # one-time cost of giving Tech support free for a few months
    "sec_cost": 15.0,     # same for Online security
    "autopay_credit": 5.0,  # one-time credit for switching to automatic payment
    "gate_hi": 70,        # trust >= this  -> act
    "gate_lo": 40,        # trust <  this  -> do not trust the explanation
    "low_risk": 0.30,     # risk below this -> just monitor
}

# every offer is built from these small "parts"
ACTIONS = {
    "contract_1y": ["contract1"],
    "contract_2y": ["contract2"],
    "techsupport": ["tech"],
    "security": ["sec"],
    "autopay": ["autopay"],
    "combo": ["contract1", "tech", "sec", "autopay"],
}
PART_APPLY = {
    "contract1": ("Contract", "One year"),
    "contract2": ("Contract", "Two year"),
    "tech": ("TechSupport", "Yes"),
    "sec": ("OnlineSecurity", "Yes"),
    "autopay": ("PaymentMethod", "Credit card (automatic)"),
}
CHECKS = ("Electronic check", "Mailed check")


def predict_batch(bundle, df):
    X = encode(df, bundle["columns"])
    return bundle["model"].predict_proba(X)[:, 1]


def revenue_at_risk(p, monthly, horizon):
    return np.asarray(p) * np.asarray(monthly) * horizon


def part_masks(df):
    internet = df["InternetService"] != "No"
    return {
        "contract1": df["Contract"] == "Month-to-month",
        "contract2": df["Contract"].isin(["Month-to-month", "One year"]),
        "tech": internet & (df["TechSupport"] != "Yes"),
        "sec": internet & (df["OnlineSecurity"] != "Yes"),
        "autopay": df["PaymentMethod"].isin(CHECKS),
    }


def part_cost(part, df, cfg):
    M = df["MonthlyCharges"].values.astype(float)
    H = cfg["horizon"]
    if part == "contract1":
        return M * H * cfg["disc1"] / 100.0
    if part == "contract2":
        return M * H * cfg["disc2"] / 100.0
    if part == "tech":
        return np.full(len(df), float(cfg["tech_cost"]))
    if part == "sec":
        return np.full(len(df), float(cfg["sec_cost"]))
    return np.full(len(df), float(cfg["autopay_credit"]))


def plan_actions(bundle, df, cfg):
    """Vectorised what-if for every offer. df = raw customer rows.
    Returns (base_probability_array, {action: dict of arrays})."""
    df = df.reset_index(drop=True)
    base = predict_batch(bundle, df)
    masks = part_masks(df)
    rev = df["MonthlyCharges"].values.astype(float) * cfg["horizon"]
    out = {}
    for key, parts in ACTIONS.items():
        n_applied = sum(masks[p].values.astype(int) for p in parts)
        ok = (n_applied >= 2) if key == "combo" else (n_applied >= 1)
        new = df.copy()
        cost = np.zeros(len(df))
        for p in parts:
            col, val = PART_APPLY[p]
            m = masks[p].values
            if m.any():
                new.loc[m, col] = val
            cost += np.where(m, part_cost(p, df, cfg), 0.0)
        newp = predict_batch(bundle, new)
        delta = base - newp
        out[key] = {"ok": np.asarray(ok), "new_p": newp, "delta": delta, "cost": cost,
                    "saved": delta * rev, "net": delta * rev - cost}
    return base, out


def single_rows(plan, i=0):
    """Applicable offers of customer i, best net gain first."""
    rows = []
    for k, d in plan.items():
        if bool(d["ok"][i]):
            rows.append({"key": k, "new_p": float(d["new_p"][i]), "delta": float(d["delta"][i]),
                         "cost": float(d["cost"][i]), "saved": float(d["saved"][i]),
                         "net": float(d["net"][i])})
    rows.sort(key=lambda r: -r["net"])
    return rows


def best_of(rows):
    return next((r for r in rows if r["net"] > 0 and r["delta"] > 0.001), None)


def decide(p, trust, best_net, cfg):
    """Trust-gated action: monitor | act | review | distrust | no_roi"""
    if p < cfg["low_risk"]:
        return "monitor"
    if trust < cfg["gate_lo"]:
        return "distrust"
    if trust < cfg["gate_hi"]:
        return "review"
    if best_net is None or best_net <= 0:
        return "no_roi"
    return "act"


def scenario_cost(orig, new, cfg):
    """Offer cost of an arbitrary change made with the live simulator."""
    M, H = float(orig["MonthlyCharges"]), cfg["horizon"]
    c = 0.0
    if new["Contract"] != orig["Contract"]:
        if new["Contract"] == "One year" and orig["Contract"] == "Month-to-month":
            c += M * H * cfg["disc1"] / 100.0
        elif new["Contract"] == "Two year":
            c += M * H * cfg["disc2"] / 100.0
    if orig["TechSupport"] != "Yes" and new["TechSupport"] == "Yes":
        c += cfg["tech_cost"]
    if orig["OnlineSecurity"] != "Yes" and new["OnlineSecurity"] == "Yes":
        c += cfg["sec_cost"]
    if orig["PaymentMethod"] in CHECKS and new["PaymentMethod"] not in CHECKS:
        c += cfg["autopay_credit"]
    return c


# ---------------------------------------------------------------- explanation helpers
def grouped_shap(cols, shap_vals):
    """Merge one-hot columns back to the original feature -> [(feature, shap_sum)] sorted by |impact|."""
    g = {}
    for c, s in zip(cols, shap_vals):
        head = c.split("_", 1)[0]
        base = head if (head in MULTI and "_" in c) else c
        g[base] = g.get(base, 0.0) + float(s)
    return sorted(g.items(), key=lambda kv: -abs(kv[1]))


def senior_counterfactual(bundle, raw):
    """Flip the senior-citizen flag only. Gender is not a model input, so its effect is 0 by design."""
    a = pd.DataFrame([raw])
    b = a.copy()
    b["SeniorCitizen"] = 1 - int(raw["SeniorCitizen"])
    p0 = float(predict_batch(bundle, a)[0])
    p1 = float(predict_batch(bundle, b)[0])
    return p0, p1, p1 - p0


# ---------------------------------------------------------------- fast SHAP for batches
def fast_shap(bundle, X):
    """Tree SHAP when possible; otherwise a simple 'replace by training mean' occlusion."""
    try:
        import shap
        sv = shap.TreeExplainer(bundle["model"]).shap_values(X)
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        if sv.shape == X.shape:
            return sv
    except Exception:  # noqa
        pass
    m = bundle["model"]
    mu = bundle["X_train"].mean().values
    X = X.astype(float)
    p0 = m.predict_proba(X)[:, 1]
    S = np.zeros(X.shape)
    for j in range(X.shape[1]):
        Xo = X.copy()
        Xo.iloc[:, j] = mu[j]
        S[:, j] = p0 - m.predict_proba(Xo)[:, 1]
    return S


class BatchExplainer:
    """Quick trust (neighbour consistency) + top reasons for many customers at once."""

    def __init__(self, bundle, n=3000):
        from sklearn.neighbors import NearestNeighbors
        self.b = bundle
        self.cols = list(bundle["columns"])
        Xtr = bundle["X_train"]
        self.sub = Xtr.sample(min(n, len(Xtr)), random_state=0).reset_index(drop=True)
        self.mu = Xtr.mean().values
        self.sd = Xtr.std().replace(0, 1).values
        self.nn = NearestNeighbors(n_neighbors=5).fit((self.sub.values - self.mu) / self.sd)
        self.shap_sub = fast_shap(bundle, self.sub)
        self.dummy = np.array([(c.split("_", 1)[0] in MULTI and "_" in c) for c in self.cols])

    def run(self, X):
        S = fast_shap(self.b, X)
        _, idx = self.nn.kneighbors((X.values - self.mu) / self.sd)
        sn = self.shap_sub[idx]                                   # N x 5 x F
        num = (S[:, None, :] * sn).sum(axis=2)
        den = np.linalg.norm(S, axis=1)[:, None] * np.linalg.norm(sn, axis=2)
        cos = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
        trust = np.round(100 * np.clip(cos.mean(axis=1), 0, None)).astype(int)
        present = ~(self.dummy[None, :] & (X.values == 0))
        Sm = np.where(present, S, -np.inf)
        order = np.argsort(-Sm, axis=1)[:, :2]
        reasons = []
        for k in range(2):
            reasons.append([self.cols[j] if Sm[i, j] > 0 else None for i, j in enumerate(order[:, k])])
        return trust, reasons[0], reasons[1]


# ---------------------------------------------------------------- batch CSV
def prepare_batch(df_raw, max_rows=20000):
    """Clean an uploaded Telco-style CSV. Returns (df, missing_columns)."""
    missing = [c for c in RAW_FEATURES if c not in df_raw.columns]
    if missing:
        return None, missing
    df = df_raw.copy().head(max_rows)
    ids = df["customerID"].astype(str) if "customerID" in df.columns else pd.Series(
        ["C%04d" % (i + 1) for i in range(len(df))], index=df.index)
    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"]).fillna(0.0)
    sc = df["SeniorCitizen"]
    if sc.dtype == object or str(sc.dtype).startswith("str"):
        sc = sc.astype(str).str.strip().str.lower().map({"yes": 1, "1": 1, "true": 1}).fillna(0)
    df["SeniorCitizen"] = sc.astype(int)
    for c in RAW_FEATURES:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            df[c] = df[c].astype(str).str.strip()
    keep = df["tenure"].notna() & df["MonthlyCharges"].notna()
    df, ids = df[keep], ids[keep]
    df = df[RAW_FEATURES].reset_index(drop=True)
    df.insert(0, "_id", ids.reset_index(drop=True))
    return df, []


def sample_csv_bytes(n=300):
    if not os.path.exists(DATA_PATH):
        return None
    d = pd.read_csv(DATA_PATH)
    d = d.sample(min(n, len(d)), random_state=1)
    d = d.drop(columns=[c for c in ["Churn"] if c in d.columns])
    return d.to_csv(index=False).encode("utf-8")


def risk_word(p, cfg):
    return "low" if p < cfg["low_risk"] else ("medium" if p < 0.60 else "high")


def run_batch(bundle, be, df, cfg):
    """Score every customer: risk, revenue at risk, best offer, quick trust, decision."""
    n = len(df)
    p, plan = plan_actions(bundle, df.drop(columns=["_id"]), cfg)
    X = encode(df, bundle["columns"])
    trust, r1, r2 = be.run(X)
    keys = list(plan)
    nets = np.stack([np.where(plan[k]["ok"], plan[k]["net"], -np.inf) for k in keys], axis=1)
    j = nets.argmax(axis=1)
    ar = np.arange(n)
    bestnet = nets[ar, j]
    has = bestnet > 0
    pick = lambda field: np.array([plan[keys[jj]][field][i] for i, jj in enumerate(j)])  # noqa: E731
    M = df["MonthlyCharges"].values.astype(float)
    out = pd.DataFrame({
        "id": df["_id"].values,
        "p": p,
        "monthly": M,
        "tenure": df["tenure"].values,
        "rar": revenue_at_risk(p, M, cfg["horizon"]),
        "reason1": r1, "reason2": r2,
        "best": [keys[jj] if h else None for jj, h in zip(j, has)],
        "new_p": np.where(has, pick("new_p"), np.nan),
        "cost": np.where(has, pick("cost"), 0.0),
        "saved": np.where(has, pick("saved"), 0.0),
        "net": np.where(has, bestnet, 0.0),
        "trust": trust,
        "trust_type": "quick",
    })
    return apply_decisions(out, cfg)


def apply_decisions(out, cfg):
    out = out.copy()
    out["decision"] = [decide(p, t, (n if isinstance(b, str) else None), cfg)
                       for p, t, n, b in zip(out["p"], out["trust"], out["net"], out["best"])]
    return out
