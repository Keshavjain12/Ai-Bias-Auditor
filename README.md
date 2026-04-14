AI Bias Auditor
Automated Detection, Tracing, and Mitigation of Bias in Machine Learning Models
---
Overview
The AI Bias Auditor is an end-to-end system that automatically audits machine learning models for algorithmic bias. It detects bias across multiple fairness metrics, traces bias to its root causes using explainability techniques (SHAP), and evaluates mitigation strategies — presenting results as a clear accuracy-vs-fairness trade-off report.
The system is built around three core research questions:
RQ1 — How can we automate comprehensive bias quantification across multiple fairness definitions?
RQ2 — Can XAI techniques trace bias back to its root causes (proxy features)?
RQ3 — How can a system evaluate and compare mitigation strategies as a clear trade-off?
---
Features
Pre-Audit Report — Dataset shape, missing values, class balance, group representation, intersectional analysis
4 Fairness Metrics — Demographic Parity Difference (DPD), Disparate Impact (DI), Equalized Odds Difference (EOD), Predictive Parity Difference (PPD)
3 Baseline Models — Logistic Regression, Random Forest, MLP (64-32 architecture)
SHAP Root-Cause Analysis — Feature importance with proxy feature detection (features correlated with sensitive attributes)
3 Mitigation Strategies
Pre-processing: Sample Reweighting
In-processing: Exponentiated Gradient (Demographic Parity)
Post-processing: Threshold Optimizer (Equalized Odds)
Reporting Dashboard — Pareto frontier plot, multi-metric comparison, bias scorecard with GOOD / MODERATE / HIGH BIAS ratings
Interactive Web UI — Upload any binary classification CSV and get a full audit instantly
---
Project Structure
```
ai-bias-auditor/
│
├── app.py                  # Streamlit web application (live demo)
├── AI_Bias_Auditor.ipynb   # Google Colab notebook (full implementation)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```
---
Dataset
The notebook is demonstrated on the Adult Income dataset (UCI) — a standard fairness benchmark with known gender and race bias.
~48,000 samples
Task: Predict if income exceeds $50K/year
Sensitive attributes: `sex`, `race`
The web app accepts any binary classification CSV — not just Adult Income.
---
Installation & Usage
Running the Notebook
Open `AI_Bias_Auditor.ipynb` in Google Colab and run all cells top to bottom. No local setup required.
Running the Web App Locally
Step 1 — Clone the repository
```bash
git clone https://github.com/Keshavjain12/ai-bias-auditor.git
cd Ai-Bias-Auditor
```
Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```
Step 3 — Run the app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501` in your browser.
Using the Web App
Upload a CSV file using the sidebar
Select the target column (binary label)
Select the sensitive attribute column (e.g. sex, race)
Select a model (Logistic Regression, Random Forest, or MLP)
Click Run Audit
View results across all 5 sections
---
Requirements
```
streamlit
pandas
numpy
scikit-learn
fairlearn
shap
matplotlib
seaborn
```
Install all with:
```bash
pip install -r requirements.txt
```
---
Pipeline Architecture
```
Input (Model + Dataset + Sensitive Attributes)
        │
        ▼
1. Data Ingest & Pre-Audit
   └── Schema validation, group representation, class balance
        │
        ▼
2. Metric Computation Engine
   └── DPD, DI, EOD, PPD for all sensitive attributes
        │
        ▼
3. Root-Cause Analysis (RCA) Engine
   └── SHAP feature importance + proxy feature detection
        │
        ▼
4. Mitigation Engine
   └── Reweighting → Exponentiated Gradient → ThresholdOptimizer
        │
        ▼
5. Reporting Dashboard
   └── Pareto frontier + trade-off table + bias scorecard
```
---
Fairness Metrics
Metric	Formula	Fair Value
Demographic Parity Difference (DPD)	|P(ŷ=1|unprivileged) − P(ŷ=1|privileged)|	0
Disparate Impact (DI)	P(ŷ=1|unprivileged) / P(ŷ=1|privileged)	1.0 (< 0.8 flagged)
Equalized Odds Difference (EOD)	max(|FPR_a − FPR_b|, |TPR_a − TPR_b|)	0
Predictive Parity Difference (PPD)	|PPV_unprivileged − PPV_privileged|	0
---
Mitigation Strategies
Strategy	Type	Method	Expected Effect
Sample Reweighting	Pre-processing	Assigns higher weights to underrepresented group-class combinations	↑ Fairness, slight ↓ Accuracy
Exponentiated Gradient	In-processing	Constrained optimization with Demographic Parity constraint	↑↑ Fairness, moderate ↓ Accuracy
Threshold Optimizer	Post-processing	Per-group classification thresholds for Equalized Odds	↑ Fairness, minimal ↓ Accuracy
---
Live Demo
The app is deployed on Streamlit Cloud:
🔗 https://ai-bias-auditor-jzn99tpf2hdjvy8kbmhfpu.streamlit.app/ 
---
Tech Stack
Python 3.x
scikit-learn — Model training and evaluation
Fairlearn — Fairness metrics and mitigation algorithms
SHAP — Explainability and root-cause analysis
Streamlit — Web application framework
Matplotlib / Seaborn — Visualizations
Pandas / NumPy — Data processing
---
Limitations
Currently supports binary classification tasks only
SHAP analysis uses a sample of 300 rows for performance — results are approximate on large datasets
Influence Functions (mentioned in research) are not implemented due to computational cost — SHAP is used as the practical alternative
Fairness metrics are mathematical proxies — they do not capture all dimensions of real-world fairness
---
Future Work
Support for multi-class classification
Integration with MLOps pipelines for continuous fairness monitoring
Extension to NLP and computer vision bias auditing
Individual fairness and intersectional fairness metrics
Influence Functions implementation for deeper root-cause analysis
---
Author
Keshav Raj Jain
Department of Computer Science and Engineering  
Manipal University Jaipur
---
Acknowledgements
Fairlearn — Microsoft's fairness toolkit
SHAP — SHapley Additive exPlanations
Adult Income Dataset — UCI Machine Learning Repository
Barocas, Hardt & Narayanan — Fairness and Machine Learning (fairmlbook.org)
