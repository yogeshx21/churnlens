# ChurnLens: Explainable, Actionable and Trust-Scored Customer Churn Prediction

**AI Innovation Hackathon 2026 | Track 3: Explainable AI (XAI) | Challenge 08: Explain My Customer Churn Prediction**

ChurnLens is a customer churn prediction tool in which every prediction is **explained**, **checked for reliability** and **turned into a decision**: what to do, how much money it saves, and how far to trust the explanation.

## Links

| What | Link |
|---|---|
| Source code | https://github.com/yogeshx21/churnlens |
| Pitch deck (PDF) | [docs/Visionaries_ChurnLens_Pitch_Deck.pdf](docs/Visionaries_ChurnLens_Pitch_Deck.pdf) |
| Live demo (Streamlit app) | https://churnlens-visionaries.streamlit.app/ |

## Team Visionaries (Dream · Innovate · Build)

| Name | Role |
|---|---|
| **Yogesh Mondhe** | Team Leader |
| **Khushi Wanjari** | Team Member |

## Problem statement

Businesses use AI to predict which customers will leave, but a churn score alone (for example "0.78") does not tell retention teams **why** a customer is at risk, **whether the explanation can be trusted**, or **what action** could prevent the loss. Opaque scores lead to badly targeted offers, wasted budget and low confidence in AI-driven decisions.

## Our solution

ChurnLens takes a customer's details and returns a churn probability together with the reasons behind it, a measured trust score for those reasons, and a ranked list of retention offers with their cost and net gain.

1. **Predict:** the user enters customer details (or uploads a whole CSV) and gets a churn probability.
2. **Explain:** SHAP shows which factors push the risk up or down, drawn as a step-by-step waterfall from the average customer to this customer.
3. **Trust:** a trust score (0 to 100) shows how consistent the explanation is (SHAP vs LIME agreement and similarity with comparable customers).
4. **Decide:** the trust score drives the action: **Act**, **Human review** or **Do not trust this explanation**.
5. **Act:** a what-if engine changes only business-controlled features (contract, tech support, online security, payment method) and shows the new risk, the offer cost and the net gain.

**Key novelty:** most XAI projects stop at a SHAP plot. ChurnLens adds a *measured* explanation-reliability score and connects it to a decision and to money, so the explanation becomes a tool for real decisions.

## Key features

- Churn probability with a risk gauge and a plain-language summary for non-technical managers
- **Trust-gated action:** Act / Human review / Do not trust / Offer not worth it / Monitor
- **Revenue at risk** (probability x monthly charges x look-ahead months) and **net gain of every offer** after the offer's own cost
- Editable business assumptions (look-ahead months, discounts, offer costs, trust thresholds), results update instantly
- **Live what-if simulator:** toggle contract, tech support, security or payment method and watch the gauge move
- **SHAP waterfall story** from the average customer to this customer, with icons
- **Similar-customers map:** this customer shown among past customers (churned vs stayed) with its 10 nearest neighbours
- **Batch mode:** upload a customer CSV, get a ranked list with top reasons, revenue at risk, best offer and a decision for every customer; download CSV or Excel
- **Retention message and voice:** ready SMS and call script in English, Hinglish and Hindi, plus a listen button that uses the browser's built-in text-to-speech
- **Customer-level fairness check:** flip the senior-citizen flag and see how much this customer's prediction moves
- One-page customer **PDF report** for the account manager
- App interface in English, Hinglish and Hindi
- Global model performance page with metrics, SHAP feature importance and a group-wise fairness table

## Who can use it and why

| User | Use |
|---|---|
| Retention / customer-success managers | See who is at risk, why, and which offer is worth its cost |
| Business analysts | Score a whole customer list and download a ranked report |
| Team leads / reviewers | Decide when to trust the AI and when a human should check |
| Students / researchers | Reusable pipeline: training, explainability, trust score and fairness audit |

## Tech stack

### Languages

Python 3.14

### Machine learning and explainability

| Tool | Version | Used for |
|---|---|---|
| scikit-learn | 1.9.1 | Logistic Regression, Random Forest, preprocessing, cross-validation, nearest neighbours |
| XGBoost | 3.4.1 | Gradient boosting model (selected model in our run) |
| SHAP | 0.54.1 | Local and global explanations |
| LIME | 0.2.0.1 | Second, independent explanation method for the trust score |
| pandas / NumPy / SciPy | 3.0.6 / 2.5.4 / 1.18.1 | Data handling |
| joblib | 1.6.0 | Saving and loading the trained model bundle |

### App and reports

| Tool | Used for |
|---|---|
| Streamlit 1.66.0 | Web interface |
| Plotly | Gauge, SHAP waterfall and similar-customers map |
| ReportLab, matplotlib | One-page PDF report |
| openpyxl | Excel report for batch mode |

### Platforms and tools

| Platform / tool | Used for |
|---|---|
| GitHub | Source control |
| Git | Version control |

No API keys, paid services or external AI services are needed.

## System architecture

```
 IBM Telco CSV
      |
      v
 +----------------+    +--------------------+    +-------------------+
 |  Preprocessing | -> |  Model training    | -> |  Model selection  |
 |  (common.py)   |    |  LR / RF / XGBoost |    |  F1 + threshold   |
 +----------------+    |  (train.py)        |    +-------------------+
                       +--------------------+              |
                                                           v
 +--------------------------------------------------------------------+
 |  engine.py : SHAP (local + global), LIME, trust score              |
 |  decision.py : revenue at risk, offer ROI, trust gate, batch,      |
 |                customer-level fairness                             |
 +--------------------------------------------------------------------+
                                  |
                                  v
 +--------------------------------------------------------------------+
 |  app.py (Streamlit): Home | Analyze | Batch (CSV) | Model & Fairness |
 |  Decision card, waterfall, live what-if, similar customers map,    |
 |  message + voice, fairness, PDF / Excel / CSV downloads            |
 +--------------------------------------------------------------------+
```

## Dataset

IBM Telco Customer Churn (public dataset): 7,043 customers with 19 attributes (demographics, services, contract and billing; the customer ID is not used) and the churn label. `train.py` downloads it automatically into `data/telco.csv` if the file is missing. **Gender is not used as a model input**; it is kept only for the fairness audit.

## Machine learning pipeline

1. **Cleaning:** `TotalCharges` converted to numeric, missing values handled; binary columns encoded 0/1, multi-class columns one-hot encoded.
2. **Split:** stratified 80/20 train/test split to preserve the churn ratio.
3. **Models compared:** Logistic Regression (interpretable baseline), Random Forest and XGBoost, with class weights to handle imbalance.
4. **Tuning:** randomized hyperparameter search with stratified cross-validation, scored on F1.
5. **Threshold tuning:** the decision threshold is chosen on cross-validated training predictions only (never on the test set).
6. **Explainability:** SHAP for local and global explanations, LIME as a second method.
7. **Trust score:** `0.5 x (SHAP vs LIME top-5 agreement) + 0.5 x (explanation similarity with the 5 nearest customers)`, scaled to 0 to 100.
8. **What-if engine:** re-predicts after changing only actionable features and ranks offers by net gain (risk reduction x monthly charges x months, minus offer cost).
9. **Fairness audit:** predicted vs actual churn rate by gender and senior-citizen group, plus a customer-level flip test.

### Results (quick run: `python train.py --fast`, held-out test set)

| Model | CV F1 | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.627 | 0.506 | 0.783 | 0.615 | 0.841 | 0.633 |
| Random Forest | 0.631 | 0.514 | 0.799 | 0.626 | 0.842 | 0.647 |
| **XGBoost (selected)** | 0.635 | 0.520 | 0.789 | 0.627 | 0.842 | 0.658 |

Final model at the tuned threshold (0.525): Precision 0.530, Recall 0.773, F1 0.629, ROC-AUC 0.842, PR-AUC 0.658. Running `python train.py` without `--fast` does a larger search. Exact numbers can change slightly with the search.

### Fairness audit (whole test data)

| Group | Predicted churn rate | Actual churn rate |
|---|---|---|
| Female | 0.392 | 0.281 |
| Male | 0.382 | 0.251 |
| Non-senior | 0.335 | 0.233 |
| Senior | 0.662 | 0.441 |

The model predicts a higher churn rate than the actual rate for every group, and the gap is largest for senior citizens. We report this openly; it is why the customer-level fairness check and the human-review step exist.

## Trust-gated decision logic

| Situation | Decision |
|---|---|
| Risk below 30% | **Monitor** (no offer needed) |
| Trust below the lower gate (default 40) | **Do not trust this explanation** |
| Trust between the gates (40 to 69) | **Human review** |
| Trust at or above the upper gate (70+) and a profitable offer exists | **Act now** |
| Trust at or above the upper gate but no offer is worth its cost | **Offer not worth it** (use a free touchpoint) |

Gates, offer costs and look-ahead months are editable in the app under "Business assumptions".

## Project structure

```
churnlens/
|-- app.py              Streamlit app: Home, Analyze, Batch (CSV), Model & Fairness, About
|-- engine.py           SHAP, LIME, trust score, global importance
|-- decision.py         Revenue at risk, offer ROI, trust gate, batch scoring, fairness check
|-- common.py           Data loading, cleaning and encoding
|-- train.py            Trains and compares models, tunes threshold, saves models/bundle.joblib
|-- report.py           One-page PDF report
|-- i18n.py             Interface texts (English, Hinglish, Hindi)
|-- i18n_extra.py       Texts for the decision-tool features and message scripts
|-- requirements.txt    Python dependencies
|-- logo.jpg            Team logo
|-- .streamlit/         Theme and app configuration
|-- data/               telco.csv (downloaded automatically if missing)
|-- models/             bundle.joblib (created by train.py; not stored in the repository)
`-- docs/               Pitch deck (PDF)
```

## Prerequisites

| Tool | Version used | How to check |
|---|---|---|
| Git | any recent version | `git --version` |
| Python | 3.14 (tested) | `python --version` or `py --version` |
| Browser | a recent Chrome or Edge (voice feature uses the browser's text-to-speech) | |

No API keys, accounts or paid services are needed. An internet connection is needed once to install packages and to download the dataset.

## Setup

### 1. Get the code

```
git clone https://github.com/yogeshx21/churnlens.git
cd churnlens
```

### 2. Create a virtual environment and install packages

Windows:

```
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Train the model (about 1 to 2 minutes)

```
python train.py --fast
```

This downloads the dataset, trains the three models, tunes the threshold and saves `models/bundle.joblib`. Use `python train.py` (without `--fast`) for a larger hyperparameter search.

## Run the project

From the repository root (with the virtual environment active):

```
streamlit run app.py
```

Open http://localhost:8501. Quick tour:

1. On **Analyze Customer**, click "Load high-risk example", then **Analyze**.
2. Read the decision card, then open the tabs: Decision, Why (story), Live what-if, Similar customers, Message + voice, Fairness.
3. On **Batch (CSV)**, download the sample CSV, upload it and click "Score all customers".

## Troubleshooting

| Problem | Fix |
|---|---|
| "Python was not found" (Windows) | Use `py` instead of `python` for creating the environment: `py -m venv venv` |
| "streamlit is not recognized" | Activate the virtual environment first: `venv\Scripts\activate` |
| "No such file: models/bundle.joblib" | Run `python train.py --fast` once (the app also trains automatically on first start) |
| Dataset download fails | Check the internet connection, or place the IBM Telco CSV at `data/telco.csv` |
| Port 8501 is already in use | Run `streamlit run app.py --server.port 8502` |
| No voice for the listen button | Use Chrome or Edge; Hindi needs a Hindi voice installed on the device |
| PDF or Excel download shows an error | Run `pip install -r requirements.txt` again |

## Limitations and safety

- Predictions are **risk alerts, not verdicts**. Explanations show model reasoning, not guaranteed causes of customer behaviour.
- The what-if engine shows how the model's prediction changes when features change. It does **not prove** that an offer will cause the same change in real customers: pilot an offer on a small group first.
- Money figures use editable assumptions (discounts, offer costs, look-ahead months), not real company data.
- In batch mode the trust score is a quick neighbour-consistency score; the full SHAP + LIME trust can be run on the top N risky customers.
- The model overestimates churn for senior citizens compared with the actual rate (see the fairness table), so those predictions should get a human review.
- The IBM Telco dataset is public and one-company; the model is not validated on other businesses.
- Final decisions stay with a human decision-maker.

## Acknowledgements

- IBM Telco Customer Churn dataset (public)
- Open-source libraries: scikit-learn, XGBoost, SHAP, LIME, Streamlit, Plotly, ReportLab

---

Built by **Team Visionaries** for the **AI Innovation Hackathon 2026** (Track 3: Explainable AI, Challenge 08: Explain My Customer Churn Prediction).
