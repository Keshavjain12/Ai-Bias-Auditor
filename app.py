import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer

pd.options.mode.chained_assignment = None

st.set_page_config(page_title="AI Bias Auditor", layout="wide")

# ================= CUSTOM CSS =================
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2d3250;
    }
    .stButton > button {
        background-color: #4A90D9;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #357ABD;
    }
    div[data-testid="stSidebar"] {
        background-color: #161b2e;
    }
</style>
""", unsafe_allow_html=True)

# ================= HERO SECTION =================
st.title("AI Bias Auditor")
st.markdown("#### Automated Detection, Tracing & Mitigation of Bias in ML Models")
hero_c1, hero_c2, hero_c3 = st.columns(3)
with hero_c1:
    st.info("**4 Fairness Metrics**\n\nDPD, DI, EOD, PPD")
with hero_c2:
    st.info("**3 Mitigation Strategies**\n\nPre, In, and Post-processing")
with hero_c3:
    st.info("**SHAP Root-Cause Analysis**\n\nProxy feature detection")

st.divider()

# ================= SIDEBAR =================
st.sidebar.title("AI Bias Auditor")
st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader("📁 Upload CSV", type=['csv'])

@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        return e

def compute_metrics(y_true, y_pred, sensitive_features):
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
    
    df_temp = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'sensitive': sensitive_features})
    
    pos_rates = df_temp.groupby('sensitive')['y_pred'].mean()
    if pos_rates.max() == 0:
        di = 1.0
    else:
        di = pos_rates.min() / pos_rates.max()
        
    def get_precision(group_df):
        pred_pos = group_df[group_df['y_pred'] == 1]
        if len(pred_pos) == 0:
            return 0
        return pred_pos['y_true'].mean()
    
    precisions = df_temp.groupby('sensitive').apply(get_precision)
    ppd = precisions.max() - precisions.min()
    
    return {
        'DPD': round(dp_diff, 4),
        'DI': round(di, 4),
        'EOD': round(eo_diff, 4),
        'PPD': round(ppd, 4)
    }

# ================= MAIN AREA (BEFORE UPLOAD) =================
if uploaded_file is None:
    st.info("👈 **Upload a CSV file in the sidebar to get started**")
else:
    data_load_result = load_data(uploaded_file)
    if isinstance(data_load_result, Exception):
        st.error(f"Error reading CSV: {data_load_result}")
        st.stop()
        
    df = data_load_result.copy()
    
    st.sidebar.subheader("Dataset Preview")
    st.sidebar.dataframe(df.head())
    st.sidebar.divider()
    
    target_col = st.sidebar.selectbox("🎯 Select Target Column", df.columns.tolist(), index=len(df.columns)-1)
    available_cols = [c for c in df.columns if c != target_col]
    sensitive_col = st.sidebar.selectbox("👤 Select Sensitive Attribute", available_cols)
    st.sidebar.divider()
    
    model_choice = st.sidebar.selectbox("🤖 Select Model", ["Logistic Regression", "Random Forest", "MLP"])
    st.sidebar.divider()
    
    run_audit = st.sidebar.button("🚀 Run Audit", use_container_width=True, type="primary")

    if run_audit:
        progress_bar = st.progress(0, text="Initializing Audit...")
        
        # 10% — Loading data
        progress_bar.progress(10, text="10% - Loading Data & Preprocessing...")
        
        df = df.dropna(subset=[target_col, sensitive_col])
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        unique_y = y.dropna().unique()
        if len(unique_y) != 2:
            st.error(f"Target column must be binary. Found classes: {unique_y}")
            st.stop()
            
        y_mapping = {unique_y[0]: 0, unique_y[1]: 1}
        y = y.map(y_mapping)
        sensitive_features = df[sensitive_col].astype(str)

        # ================= SECTION 1 =================
        st.markdown("<h2 style='color:#4A90D9'>📋 Section 1: Pre-Audit Report</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Info")
            st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
            st.write(f"**Missing Values:** {df.isna().sum().sum()}")
            st.write("**Class Balance (Target):**")
            st.dataframe(y.value_counts().rename(index={0: unique_y[0], 1: unique_y[1]}))
            
        with col2:
            st.subheader(f"Group Representation ({sensitive_col})")
            st.dataframe(df[sensitive_col].value_counts())
        st.divider()

        # 30% — Training model
        progress_bar.progress(30, text="30% - Training Baseline Model...")
        
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        if model_choice == "Logistic Regression":
            clf = LogisticRegression(random_state=42, max_iter=1000)
        elif model_choice == "Random Forest":
            clf = RandomForestClassifier(random_state=42)
        else:
            clf = MLPClassifier(random_state=42, max_iter=500)

        X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
            X, y, sensitive_features, test_size=0.2, random_state=42, stratify=y
        )

        baseline_model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
        baseline_model.fit(X_train, y_train)

        y_pred_base = baseline_model.predict(X_test)
        try:
            y_prob_base = baseline_model.predict_proba(X_test)[:, 1]
            auc_base = roc_auc_score(y_test, y_prob_base)
        except:
            auc_base = np.nan
            
        acc_base = accuracy_score(y_test, y_pred_base)
        f1_base = f1_score(y_test, y_pred_base)

        # ================= SECTION 2 =================
        st.markdown(f"<h2 style='color:#4A90D9'>📈 Section 2: Baseline Model ({model_choice})</h2>", unsafe_allow_html=True)
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Accuracy", f"{acc_base:.4f}")
        m_col2.metric("AUC", f"{auc_base:.4f}" if not np.isnan(auc_base) else "N/A")
        m_col3.metric("F1 Score", f"{f1_base:.4f}")
        st.divider()

        # 50% — Computing fairness metrics
        progress_bar.progress(50, text="50% - Computing Fairness Metrics...")
        base_metrics = compute_metrics(y_test, y_pred_base, sens_test)

        # ================= SECTION 3 =================
        st.markdown("<h2 style='color:#4A90D9'>⚖️ Section 3: Fairness Metrics</h2>", unsafe_allow_html=True)
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        
        f_col1.metric("DPD (Ideal: 0)", f"{base_metrics['DPD']:.4f}", delta=f"{base_metrics['DPD']:.4f} gap", delta_color="inverse")
        di_delta = f"{base_metrics['DI'] - 1.0:+.4f} off ideal"
        f_col2.metric("DI (Ideal: 1)", f"{base_metrics['DI']:.4f}", delta=di_delta, delta_color="off")
        f_col3.metric("EOD (Ideal: 0)", f"{base_metrics['EOD']:.4f}", delta=f"{base_metrics['EOD']:.4f} gap", delta_color="inverse")
        f_col4.metric("PPD (Ideal: 0)", f"{base_metrics['PPD']:.4f}", delta=f"{base_metrics['PPD']:.4f} gap", delta_color="inverse")
        st.divider()

        # 70% — Running SHAP analysis
        progress_bar.progress(70, text="70% - Running SHAP Analysis...")

        # ================= SECTION 4 =================
        st.markdown("<h2 style='color:#4A90D9'>🔍 Section 4: Root Cause Analysis (SHAP)</h2>", unsafe_allow_html=True)
        
        X_train_proc = pd.DataFrame(preprocessor.fit_transform(X_train), columns=preprocessor.get_feature_names_out())
        X_test_proc = pd.DataFrame(preprocessor.transform(X_test), columns=preprocessor.get_feature_names_out())

        sens_train_num = pd.factorize(sens_train)[0]
        correlations = {}
        for col in X_train_proc.columns:
            corr = np.corrcoef(X_train_proc[col], sens_train_num)[0, 1]
            correlations[col] = 0 if np.isnan(corr) else abs(corr)

        shap_sample = X_test_proc.sample(n=min(300, len(X_test_proc)), random_state=42)

        if model_choice == "Logistic Regression":
            explainer = shap.LinearExplainer(baseline_model.named_steps['classifier'], X_train_proc)
            shap_values_obj = explainer(shap_sample)
            shap_values = shap_values_obj.values
        elif model_choice == "Random Forest":
            explainer = shap.TreeExplainer(baseline_model.named_steps['classifier'])
            shap_values_raw = explainer.shap_values(shap_sample)
            shap_values = shap_values_raw[1] if isinstance(shap_values_raw, list) else shap_values_raw
        else:
            summary = shap.kmeans(X_train_proc, 10)
            explainer = shap.KernelExplainer(baseline_model.named_steps['classifier'].predict_proba, summary)
            shap_values_raw = explainer.shap_values(shap_sample)
            shap_values = shap_values_raw[1] if isinstance(shap_values_raw, list) else shap_values_raw
            if len(np.array(shap_values).shape) == 3:
                 shap_values = np.array(shap_values)[:, :, -1]

        shap_importance = np.abs(shap_values).mean(axis=0)
        imp_df = pd.DataFrame({'Feature': X_train_proc.columns, 'Importance': shap_importance})
        imp_df = imp_df.sort_values(by='Importance', ascending=True).tail(10)

        fig1, ax1 = plt.subplots(figsize=(8, 6))
        # Add background color to matplotlib figure to match the dark theme nicely
        fig1.patch.set_facecolor('#0e1117')
        ax1.set_facecolor('#0e1117')
        ax1.tick_params(colors='white')
        ax1.yaxis.label.set_color('white')
        ax1.xaxis.label.set_color('white')
        ax1.title.set_color('white')
        
        bar_colors = ['#ff4b4b' if correlations.get(f, 0) > 0.1 else '#4A90D9' for f in imp_df['Feature']]
        ax1.barh(imp_df['Feature'], imp_df['Importance'], color=bar_colors)
        ax1.set_title("Top 10 Features by SHAP Importance (Red = Proxy Feature > 0.1)")
        ax1.set_xlabel("Mean |SHAP value|")
        st.pyplot(fig1)

        proxy_features = [f for f in imp_df['Feature'] if correlations.get(f, 0) > 0.1]
        if proxy_features:
            st.error(f"**Flagged Proxy Features** (High Correlation > 0.1 with Sensitive Attribute):\n {', '.join(proxy_features)}")
        else:
            st.success("**No strong proxy features** detected among the top 10 important features.")
        st.divider()

        # 90% — Running mitigations
        progress_bar.progress(90, text="90% - Running Mitigation Strategies...")

        # ================= SECTION 5 =================
        st.markdown("<h2 style='color:#4A90D9'>🛡️ Section 5: Mitigation Engine</h2>", unsafe_allow_html=True)
        mitigation_results = [{
            "Method": "Baseline", "Accuracy": acc_base,
            "DPD": base_metrics['DPD'], "EOD": base_metrics['EOD'],
            "DI": base_metrics['DI'], "PPD": base_metrics['PPD']
        }]

        # 1. Sample Reweighting
        joint_groups = sens_train.astype(str) + "_" + y_train.astype(str)
        sample_weights = compute_sample_weight(class_weight='balanced', y=joint_groups)

        if model_choice == "MLP":
            indices = np.random.choice(len(X_train), size=len(X_train), p=sample_weights/sample_weights.sum())
            X_train_resampled = X_train.iloc[indices]
            y_train_resampled = y_train.iloc[indices]
            model_rw = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', MLPClassifier(random_state=42, max_iter=500))])
            model_rw.fit(X_train_resampled, y_train_resampled)
        else:
            model_rw = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf.__class__(random_state=42))])
            model_rw.fit(X_train, y_train, classifier__sample_weight=sample_weights)
        
        y_pred_rw = model_rw.predict(X_test)
        met_rw = compute_metrics(y_test, y_pred_rw, sens_test)
        mitigation_results.append({
            "Method": "Sample Reweighting", "Accuracy": accuracy_score(y_test, y_pred_rw),
            "DPD": met_rw['DPD'], "EOD": met_rw['EOD'], "DI": met_rw['DI'], "PPD": met_rw['PPD']
        })

        # 2. ExponentiatedGradient
        if model_choice == "MLP":
            eg_estimator = LogisticRegression(random_state=42, max_iter=1000)
        else:
            eg_estimator = clf.__class__(random_state=42)
            
        try:
            eg = ExponentiatedGradient(estimator=eg_estimator, constraints=DemographicParity())
            eg.fit(X_train_proc, y_train, sensitive_features=sens_train)
            y_pred_eg = eg.predict(X_test_proc)
            met_eg = compute_metrics(y_test, y_pred_eg, sens_test)
            mitigation_results.append({
                "Method": "ExponentiatedGradient", "Accuracy": accuracy_score(y_test, y_pred_eg),
                "DPD": met_eg['DPD'], "EOD": met_eg['EOD'], "DI": met_eg['DI'], "PPD": met_eg['PPD']
            })
        except Exception as e:
            st.warning(f"⚠️ **ExponentiatedGradient Skipped:** {str(e)}")

        # 3. ThresholdOptimizer
        try:
            to = ThresholdOptimizer(estimator=baseline_model, constraints="equalized_odds", prefit=True, predict_method="predict")
            to.fit(X_train, y_train, sensitive_features=sens_train)
            y_pred_to = to.predict(X_test, sensitive_features=sens_test)
            met_to = compute_metrics(y_test, y_pred_to, sens_test)
            mitigation_results.append({
                "Method": "ThresholdOptimizer", "Accuracy": accuracy_score(y_test, y_pred_to),
                "DPD": met_to['DPD'], "EOD": met_to['EOD'], "DI": met_to['DI'], "PPD": met_to['PPD']
            })
        except Exception as e:
            st.warning(f"⚠️ **ThresholdOptimizer Skipped:** {str(e)} \n\n*(This is usually caused by 'degenerate labels' — meaning some sensitive groups in your dataset only contain outcomes of a single class).*")

        res_df = pd.DataFrame(mitigation_results)
        
        display_df = res_df.copy()
        for col in ["Accuracy", "DPD", "EOD", "DI", "PPD"]:
            base_val = res_df.loc[res_df['Method'] == 'Baseline', col].values[0]
            display_df[col] = res_df[col].apply(lambda x: f"{x:.4f}") + res_df[col].apply(lambda x: f" ({(x - base_val):+.4f})" if (x - base_val) != 0 else "")

        st.dataframe(display_df, use_container_width=True)

        cols_plots = st.columns(2)
        with cols_plots[0]:
            fig2, ax2 = plt.subplots()
            fig2.patch.set_facecolor('#0e1117')
            ax2.set_facecolor('#0e1117')
            ax2.tick_params(colors='white')
            ax2.yaxis.label.set_color('white')
            ax2.xaxis.label.set_color('white')
            
            colors = ['#ffffff', '#4A90D9', '#ff9f36', '#32cd32']
            for i, row in res_df.iterrows():
                ax2.scatter(row['DPD'], row['Accuracy'], label=row['Method'], color=colors[i], s=120)
                ax2.annotate(row['Method'], (row['DPD'], row['Accuracy']), xytext=(5, 5), textcoords='offset points', color='white')
            ax2.set_xlabel("DPD (Lower is Fairer)")
            ax2.set_ylabel("Accuracy (Higher is Better)")
            ax2.set_title("Pareto Frontier", color="white")
            st.pyplot(fig2)

        with cols_plots[1]:
            fig3, ax3 = plt.subplots(figsize=(8,6))
            fig3.patch.set_facecolor('#0e1117')
            ax3.set_facecolor('#0e1117')
            ax3.tick_params(colors='white')
            ax3.yaxis.label.set_color('white')
            ax3.title.set_color('white')

            x = np.arange(len(res_df['Method']))
            width = 0.25
            multiplier = 0
            bar_cols = ['#ff4b4b', '#ff9f36', '#4A90D9']
            for metric in ["DPD", "EOD", "PPD"]:
                values = res_df[metric].values
                offset = width * multiplier
                ax3.bar(x + offset, values, width, label=metric, color=bar_cols[multiplier])
                multiplier += 1

            ax3.set_ylabel('Scores')
            ax3.set_title("Fairness Measures Comparison")
            ax3.set_xticks(x + width)
            ax3.set_xticklabels(res_df['Method'], rotation=30, ha="right")
            legend = ax3.legend()
            for text in legend.get_texts(): text.set_color("black")
            plt.tight_layout()
            st.pyplot(fig3)

        st.divider()

        # ================= BIAS SCORECARD =================
        st.markdown("<h2 style='color:#4A90D9'>✅ Final Bias Scorecard</h2>", unsafe_allow_html=True)
        def render_score(name, val):
            if "Accuracy" in name:
                if val >= 0.75: st.success(f"**{name}: {val:.3f}**\n\n✓ GOOD")
                elif val >= 0.65: st.warning(f"**{name}: {val:.3f}**\n\n~ MODERATE")
                else: st.error(f"**{name}: {val:.3f}**\n\n✗ POOR")
            elif "DI" in name:
                if 0.8 <= val <= 1.25: st.success(f"**{name}: {val:.3f}**\n\n✓ FAIR")
                elif 0.6 <= val <= 1.4: st.warning(f"**{name}: {val:.3f}**\n\n~ MODERATE")
                else: st.error(f"**{name}: {val:.3f}**\n\n✗ HIGH BIAS")
            else: 
                val = abs(val)
                if val <= 0.1: st.success(f"**{name}: {val:.3f}**\n\n✓ FAIR")
                elif val <= 0.2: st.warning(f"**{name}: {val:.3f}**\n\n~ MODERATE")
                else: st.error(f"**{name}: {val:.3f}**\n\n✗ HIGH BIAS")

        sc_a, sc_b, sc_c, sc_d, sc_e = st.columns(5)
        with sc_a: render_score("Accuracy", acc_base)
        with sc_b: render_score("DPD", base_metrics['DPD'])
        with sc_c: render_score("DI", base_metrics['DI'])
        with sc_d: render_score("EOD", base_metrics['EOD'])
        with sc_e: render_score("PPD", base_metrics['PPD'])

        st.divider()

        # ================= RECOMMENDATION =================
        st.markdown("<h2 style='color:#4A90D9'>💡 Recommendation</h2>", unsafe_allow_html=True)
        valid_methods = res_df[res_df["Method"] != "Baseline"]
        valid_methods = valid_methods[valid_methods["Accuracy"] >= acc_base - 0.05]
        
        if len(valid_methods) > 0:
            best_row = valid_methods.loc[valid_methods["DPD"].abs().idxmin()]
        else:
            best_row = res_df.loc[res_df["Method"] != "Baseline"].loc[res_df["DPD"].abs().idxmin()]
            
        best_method = best_row["Method"]
        best_dpd = best_row["DPD"]
        acc_drop = (acc_base - best_row["Accuracy"]) * 100

        st.info(f"""
        **Recommended Strategy:** `{best_method}`
        
        **Reasoning:** This method achieved a Demographic Parity Difference (DPD) of **{best_dpd:.4f}** (which means highly fair decisions across groups), 
        while maintaining an accuracy drop of only **{acc_drop:.1f}%** from the untreated baseline.
        """)
        
        progress_bar.progress(100, text="100% - Audit Complete!")
        time.sleep(1)
        progress_bar.empty()
        st.success("✅ Audit Complete!")
