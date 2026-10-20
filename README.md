# ChurnLens - Team Visionaries
AI Innovation Hackathon 2026 | Track 3: Explainable AI | Challenge 08

## Run (5 steps)
1. `pip install -r requirements.txt`
2. Dataset: `train.py` downloads IBM Telco CSV automatically. Agar download fail ho to Kaggle/IBM se CSV leke `data/telco.csv` naam se rakho.
3. `python train.py`  (quick test ke liye `python train.py --fast`)
4. `streamlit run app.py`
5. Browser me kholo, upar se English / Hinglish / हिन्दी chuno.

## New: decision-tool features
- Trust-gated action (Act / Human review / Don't trust) + revenue at risk and offer ROI
- Live what-if simulator with risk gauge, SHAP waterfall story, similar-customers map
- Batch CSV mode (ranked list, Excel/CSV report), retention message + browser voice (EN/Hinglish/Hindi)
- Customer-level fairness check, one-page PDF report
- After updating, run `python train.py --fast` once so the similar-customers map can show real churn labels.

## Files
- `app.py` - Streamlit UI (home page, language switch, analyze, model & fairness, about)
- `i18n.py` - English, Hindi, Hinglish text
- `engine.py` - SHAP, LIME, trust score, what-if
- `decision.py` - revenue at risk, offer ROI, trust gate, batch scoring, fairness check
- `i18n_extra.py` - texts for the new features (English, Hinglish, Hindi)
- `report.py` - one-page PDF
- `train.py` - training, model selection, threshold tuning, fairness audit
- `common.py` - data cleaning and encoding
- `logo.jpg` - Visionaries logo
