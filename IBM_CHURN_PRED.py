import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight | Telecom Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 60%, #16213e 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 14px;
    padding: 22px 24px;
    text-align: center;
}
.metric-label { font-size:0.75rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#64748b; margin-bottom:6px; }
.metric-value { font-family:'Space Mono',monospace; font-size:2rem; font-weight:700; color:#63b3ed; }
.metric-sub { font-size:0.78rem; color:#475569; margin-top:4px; }
.section-title {
    font-family:'Space Mono',monospace; font-size:1.3rem; font-weight:700;
    color:#e2e8f0; border-left:3px solid #63b3ed;
    padding-left:14px; margin:24px 0 16px;
}
.badge { display:inline-block; background:rgba(99,179,237,0.15); color:#63b3ed;
         border:1px solid rgba(99,179,237,0.3); border-radius:20px;
         padding:3px 12px; font-size:0.75rem; font-weight:600; font-family:'Space Mono',monospace; }
.badge-green { background:rgba(72,199,142,0.15); color:#48c78e; border-color:rgba(72,199,142,0.3); }
.badge-red   { background:rgba(248,113,113,0.15); color:#f87171; border-color:rgba(248,113,113,0.3); }
.hero-title {
    font-family:'Space Mono',monospace; font-size:2.4rem; font-weight:700;
    background:linear-gradient(135deg,#63b3ed,#a78bfa,#f472b6);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1.2;
}
.hero-sub { color:#94a3b8; font-size:1.05rem; margin-top:8px; }
.step-box {
    background:#0f172a; border:1px solid rgba(99,179,237,0.15);
    border-radius:10px; padding:14px 18px; margin:8px 0;
    font-size:0.88rem; color:#94a3b8;
}
.step-num { color:#63b3ed; font-family:'Space Mono',monospace; font-weight:700; }
div[data-testid="stButton"] button {
    background:linear-gradient(135deg,#2563eb,#7c3aed); color:white;
    border:none; border-radius:10px; font-family:'DM Sans',sans-serif;
    font-weight:600; padding:10px 22px; transition:opacity 0.2s;
}
div[data-testid="stButton"] button:hover { opacity:0.88; }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────
for key, val in [("page","Home"),("trained",False),("df_raw",None)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar navigation ───────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 24px;'>
        <div style='font-family:Space Mono,monospace;font-size:1.3rem;font-weight:700;
                    background:linear-gradient(135deg,#63b3ed,#a78bfa);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            📡 ChurnSight
        </div>
        <div style='font-size:0.75rem;color:#475569;margin-top:4px;letter-spacing:0.05em;'>
            TELECOM CHURN ANALYTICS
        </div>
    </div>
    """, unsafe_allow_html=True)

    for icon, label in [("🏠","Home"),("📂","Data Upload"),("🔍","Explore Data"),
                         ("⚙️","Train Models"),("📊","Results"),("🔮","Predict")]:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
            st.rerun()

    st.markdown("---")
    if st.session_state.trained:
        st.markdown('<div class="badge badge-green">✓ Models Trained</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge badge-red">○ Not Trained</div>', unsafe_allow_html=True)

    if st.session_state.df_raw is not None:
        r, c = st.session_state.df_raw.shape
        st.markdown(f"<div style='margin-top:12px;font-size:0.78rem;color:#475569;'>"
                    f"Dataset: <b style='color:#63b3ed'>{r:,}</b> rows · "
                    f"<b style='color:#63b3ed'>{c}</b> cols</div>", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────
def page_nav(prev_label=None, next_label=None):
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns([1, 6, 1])
    if prev_label:
        with cols[0]:
            if st.button(f"← {prev_label}"):
                st.session_state.page = prev_label; st.rerun()
    if next_label:
        with cols[2]:
            if st.button(f"{next_label} →"):
                st.session_state.page = next_label; st.rerun()

def dark_fig(figsize=(9,4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    ax.tick_params(colors='#94a3b8')
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.title.set_color('#e2e8f0')
    for spine in ax.spines.values(): spine.set_edgecolor('#1e293b')
    return fig, ax

def clean_df(df):
    """Drop useless cols, normalise column names, fix dtypes."""
    df = df.drop(columns=['CustomerID','Lat Long','Zip Code','City','State',
                           'Country','Churn Reason','Count'], errors='ignore')
    df.columns = df.columns.str.strip().str.replace(' ', '_')

    # Fix Total_Charges
    df['Total_Charges'] = pd.to_numeric(df['Total_Charges'], errors='coerce')
    df['Total_Charges'] = df['Total_Charges'].fillna(df['Total_Charges'].median())

    # Build numeric Churn_Value from whatever column exists
    if 'Churn_Value' in df.columns:
        df['Churn_Value'] = pd.to_numeric(
            df['Churn_Value'].astype(str).str.strip()
                             .map({'Yes':1,'No':0,'1':1,'0':0,'1.0':1,'0.0':0}),
            errors='coerce').fillna(
            pd.to_numeric(df['Churn_Value'], errors='coerce')
        ).fillna(0).astype(int)
    elif 'Churn_Label' in df.columns:
        df['Churn_Value'] = (df['Churn_Label'].astype(str).str.strip()
                             .map({'Yes':1,'No':0}).fillna(0).astype(int))

    # Convert Yes/No binary cols
    for col in df.select_dtypes(include='object').columns:
        uniq = set(df[col].dropna().astype(str).str.strip().str.lower().unique())
        if uniq <= {'yes','no'}:
            df[col] = df[col].astype(str).str.strip().str.lower().map({'yes':1,'no':0})

    return df

# ════════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════════
if st.session_state.page == "Home":
    st.markdown("""
    <div style='padding:40px 0 20px;'>
        <div class='hero-title'>Predict Customer Churn<br>Before It Happens.</div>
        <div class='hero-sub'>Upload your Telco dataset, explore it, train ML models, and get predictions — all in one place.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, label, val, sub in [
        (c1,"Models Available","4","RF · XGBoost · LR · AdaBoost"),
        (c2,"Balancing Method","SMOTE","Handles class imbalance"),
        (c3,"Explainability","SHAP","Feature importance"),
    ]:
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{val}</div>
                <div class='metric-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>How It Works</div>", unsafe_allow_html=True)
    for i,(step,desc) in enumerate([
        ("Upload Data","Upload your Telco churn Excel/CSV file."),
        ("Explore","View distributions, missing values, and correlations."),
        ("Train Models","Select models, configure settings, and train with SMOTE."),
        ("Review Results","Compare accuracy, AUC, confusion matrix, SHAP plots."),
        ("Predict","Enter customer attributes to get a real-time churn score."),
    ],1):
        st.markdown(f"""<div class='step-box'>
            <span class='step-num'>0{i}.</span>&nbsp;
            <b style='color:#e2e8f0'>{step}</b>&nbsp;— {desc}
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀  Get Started — Upload Data"):
        st.session_state.page = "Data Upload"; st.rerun()

# ════════════════════════════════════════════════════════════════
# PAGE: DATA UPLOAD
# ════════════════════════════════════════════════════════════════
elif st.session_state.page == "Data Upload":
    st.markdown("<div class='hero-title' style='font-size:1.8rem'>📂 Data Upload</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Upload the Telco Customer Churn dataset (Excel or CSV).</div><br>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop your file here", type=["xlsx","xls","csv"])

    if uploaded:
        with st.spinner("Reading and cleaning file…"):
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)

                df = clean_df(df)
                st.session_state.df_raw = df
                st.session_state.trained = False

                churn_pct = pd.to_numeric(df.get('Churn_Value', pd.Series([])), errors='coerce').mean()
                churn_pct_str = f"{churn_pct*100:.1f}%" if not pd.isna(churn_pct) else "N/A"
                st.success(f"✅ Loaded **{df.shape[0]:,}** rows × **{df.shape[1]}** cols · Churn rate: **{churn_pct_str}**")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    if st.session_state.df_raw is not None:
        df = st.session_state.df_raw
        st.markdown("<div class='section-title'>Preview</div>", unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing", int(df.isnull().sum().sum()))
        if 'Churn_Value' in df.columns:
            pct = pd.to_numeric(df['Churn_Value'], errors='coerce').mean()
            c4.metric("Churn %", f"{pct*100:.1f}%" if not pd.isna(pct) else "N/A")
        else:
            c4.metric("Churn %", "N/A")

    page_nav(prev_label="Home", next_label="Explore Data")

# ════════════════════════════════════════════════════════════════
# PAGE: EXPLORE DATA
# ════════════════════════════════════════════════════════════════
elif st.session_state.page == "Explore Data":
    st.markdown("<div class='hero-title' style='font-size:1.8rem'>🔍 Explore Data</div>", unsafe_allow_html=True)

    if st.session_state.df_raw is None:
        st.warning("Please upload data first.")
        if st.button("Go to Data Upload"):
            st.session_state.page = "Data Upload"; st.rerun()
    else:
        df = st.session_state.df_raw
        tab1,tab2,tab3,tab4 = st.tabs(["📈 Distributions","🔗 Correlations","📋 Stats","❓ Missing"])

        with tab1:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            chosen = st.selectbox("Select column", num_cols)
            hue_on = st.checkbox("Color by Churn", value=True)
            fig,ax = dark_fig((9,4))
            if hue_on and 'Churn_Value' in df.columns:
                for val,grp in df.groupby('Churn_Value'):
                    ax.hist(grp[chosen].dropna(), bins=40, alpha=0.7,
                            label=f"Churn={val}",
                            color='#f87171' if val==1 else '#63b3ed')
                ax.legend(facecolor='#0f172a', labelcolor='#94a3b8')
            else:
                ax.hist(df[chosen].dropna(), bins=40, color='#63b3ed', alpha=0.8)
            ax.set_xlabel(chosen); ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {chosen}")
            st.pyplot(fig); plt.close()

            if 'Churn_Value' in df.columns:
                fig2,ax2 = dark_fig((5,4))
                cv = pd.to_numeric(df['Churn_Value'], errors='coerce').fillna(0).astype(int)
                ax2.bar(['No Churn','Churn'],[(cv==0).sum(),(cv==1).sum()],
                        color=['#63b3ed','#f87171'], alpha=0.85, width=0.4)
                ax2.set_title("Churn Distribution")
                st.pyplot(fig2); plt.close()

        with tab2:
            num_df = df.select_dtypes(include=np.number)
            fig,ax = dark_fig((10,8))
            sns.heatmap(num_df.corr(), ax=ax, cmap='coolwarm', center=0,
                        annot=False, linewidths=0.3, cbar_kws={'shrink':0.8})
            ax.set_title("Correlation Matrix")
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with tab3:
            st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

        with tab4:
            miss = df.isnull().sum().reset_index()
            miss.columns = ['Column','Missing']
            miss['%'] = (miss['Missing']/len(df)*100).round(2)
            miss = miss[miss['Missing']>0]
            if miss.empty:
                st.success("No missing values found! 🎉")
            else:
                st.dataframe(miss, use_container_width=True)

    page_nav(prev_label="Data Upload", next_label="Train Models")

# ════════════════════════════════════════════════════════════════
# PAGE: TRAIN MODELS
# ════════════════════════════════════════════════════════════════
elif st.session_state.page == "Train Models":
    st.markdown("<div class='hero-title' style='font-size:1.8rem'>⚙️ Train Models</div>", unsafe_allow_html=True)

    if st.session_state.df_raw is None:
        st.warning("Please upload data first.")
        if st.button("Go to Data Upload"):
            st.session_state.page = "Data Upload"; st.rerun()
    else:
        df = st.session_state.df_raw
        st.markdown("<div class='section-title'>Configuration</div>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
            use_smote = st.checkbox("Apply SMOTE (handle imbalance)", value=True)
        with c2:
            model_choices = st.multiselect(
                "Select models to train",
                ["Random Forest","XGBoost","Logistic Regression","AdaBoost"],
                default=["Random Forest","XGBoost","Logistic Regression","AdaBoost"]
            )

        with st.expander("Advanced Hyperparameters"):
            n_est_rf  = st.slider("RF: n_estimators", 50, 500, 200, 50)
            n_est_xgb = st.slider("XGB: n_estimators", 50, 500, 200, 50)
            lr_xgb    = st.slider("XGB: learning_rate", 0.01, 0.3, 0.05, 0.01)
            md_xgb    = st.slider("XGB: max_depth", 2, 10, 5)
            n_est_ada = st.slider("AdaBoost: n_estimators", 50, 300, 100, 25)

        if st.button("🚀 Train Selected Models"):
            if not model_choices:
                st.error("Select at least one model.")
            elif 'Churn_Value' not in df.columns:
                st.error("No 'Churn_Value' column found. Please re-upload your data.")
            else:
                y = pd.to_numeric(df['Churn_Value'], errors='coerce').fillna(0).astype(int)
                X = df.drop(['Churn_Label','Churn_Value','Churn_Score'], axis=1, errors='ignore')
                X = pd.get_dummies(X, drop_first=False)
                # Drop any remaining non-numeric columns
                X = X.select_dtypes(include=np.number)
                feature_names = X.columns.tolist()

                X_train,X_test,y_train,y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y)

                scaler = StandardScaler()
                X_tr_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
                X_te_sc = pd.DataFrame(scaler.transform(X_test),  columns=feature_names)

                if use_smote:
                    smote = SMOTE(random_state=42)
                    X_tr, y_tr = smote.fit_resample(X_tr_sc, y_train)
                    st.info(f"SMOTE: {pd.Series(y_tr).value_counts().to_dict()}")
                else:
                    X_tr, y_tr = X_tr_sc, y_train

                model_defs = {
                    "Random Forest":      RandomForestClassifier(n_estimators=n_est_rf, random_state=42, n_jobs=-1),
                    "XGBoost":            XGBClassifier(n_estimators=n_est_xgb, learning_rate=lr_xgb,
                                                        max_depth=md_xgb, eval_metric='logloss',
                                                        random_state=42, n_jobs=-1),
                    "Logistic Regression":LogisticRegression(max_iter=1000, random_state=42),
                    "AdaBoost":           AdaBoostClassifier(n_estimators=n_est_ada, random_state=42),
                }

                results = {}
                prog = st.progress(0)
                status = st.empty()
                for i,name in enumerate(model_choices):
                    status.text(f"Training {name}…")
                    m = model_defs[name]
                    m.fit(X_tr, y_tr)
                    yp  = m.predict(X_te_sc)
                    ypr = m.predict_proba(X_te_sc)[:,1]
                    results[name] = {
                        "model":m, "acc":accuracy_score(y_test,yp),
                        "auc":roc_auc_score(y_test,ypr),
                        "y_pred":yp, "y_prob":ypr,
                        "report":classification_report(y_test,yp,output_dict=True),
                        "cm":confusion_matrix(y_test,yp)
                    }
                    prog.progress((i+1)/len(model_choices))
                status.text("✅ Done!")

                best = max(results, key=lambda k: results[k]["auc"])
                st.session_state.update({
                    "results":results, "feature_names":feature_names,
                    "X_test_sc":X_te_sc, "y_test":y_test, "scaler":scaler,
                    "feature_names_all":feature_names,
                    "xgb_model":results.get("XGBoost",{}).get("model"),
                    "rf_model":results.get("Random Forest",{}).get("model"),
                    "trained":True, "best_model_name":best
                })
                st.success(f"🏆 Best: **{best}** (AUC = {results[best]['auc']:.4f})")

    page_nav(prev_label="Explore Data", next_label="Results")

# ════════════════════════════════════════════════════════════════
# PAGE: RESULTS
# ════════════════════════════════════════════════════════════════
elif st.session_state.page == "Results":
    st.markdown("<div class='hero-title' style='font-size:1.8rem'>📊 Results</div>", unsafe_allow_html=True)

    if not st.session_state.trained:
        st.warning("No results yet. Please train models first.")
        if st.button("Go to Train Models"):
            st.session_state.page = "Train Models"; st.rerun()
    else:
        results      = st.session_state.results
        X_test_sc    = st.session_state.X_test_sc
        y_test       = st.session_state.y_test
        feature_names= st.session_state.feature_names

        st.markdown("<div class='section-title'>Model Comparison</div>", unsafe_allow_html=True)
        cols = st.columns(len(results))
        for i,(name,res) in enumerate(results.items()):
            badge = "🏆 " if name==st.session_state.best_model_name else ""
            with cols[i]:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-label'>{badge}{name}</div>
                    <div class='metric-value'>{res['auc']:.3f}</div>
                    <div class='metric-sub'>AUC · Acc {res['acc']*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        tab1,tab2,tab3,tab4 = st.tabs(["📉 ROC Curves","🗺 Confusion Matrix","📋 Reports","🌟 Feature Importance"])

        with tab1:
            fig,ax = dark_fig((9,5))
            for i,(name,res) in enumerate(results.items()):
                fpr,tpr,_ = roc_curve(y_test,res['y_prob'])
                ax.plot(fpr,tpr,lw=2,
                        color=['#63b3ed','#f472b6','#48c78e','#fbbf24'][i%4],
                        label=f"{name} (AUC={res['auc']:.3f})")
            ax.plot([0,1],[0,1],'--',color='#334155',lw=1)
            ax.set(xlabel="FPR",ylabel="TPR",title="ROC Curve Comparison")
            ax.legend(facecolor='#0f172a',labelcolor='#94a3b8',fontsize=9)
            ax.grid(True,alpha=0.1); fig.tight_layout()
            st.pyplot(fig); plt.close()

        with tab2:
            sel = st.selectbox("Select model", list(results.keys()))
            fig,ax = dark_fig((5,4))
            sns.heatmap(results[sel]['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No Churn','Churn'],
                        yticklabels=['No Churn','Churn'],
                        cbar_kws={'shrink':0.8})
            ax.set(title=f"Confusion Matrix — {sel}", xlabel="Predicted", ylabel="Actual")
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with tab3:
            sel2 = st.selectbox("Select model ", list(results.keys()))
            st.dataframe(pd.DataFrame(results[sel2]['report']).T.style.format("{:.3f}"),
                         use_container_width=True)

        with tab4:
            fi1,fi2 = st.tabs(["🌲 Random Forest","⚡ XGBoost SHAP"])
            with fi1:
                rf = st.session_state.rf_model
                if rf:
                    fi = pd.Series(rf.feature_importances_,index=feature_names).sort_values(ascending=False)
                    top_n = st.slider("Top N features",5,20,10,key="rf_top")
                    fig,ax = dark_fig((9,5))
                    fi_top = fi.head(top_n)
                    ax.barh(fi_top.index[::-1],fi_top.values[::-1],color='#63b3ed',alpha=0.85)
                    ax.set(title=f"Top {top_n} Features — Random Forest",xlabel="Importance")
                    ax.grid(True,alpha=0.1,axis='x'); fig.tight_layout()
                    st.pyplot(fig); plt.close()
                else:
                    st.info("Random Forest was not trained.")

            with fi2:
                xgb_m = st.session_state.xgb_model
                if xgb_m:
                    with st.spinner("Computing SHAP values…"):
                        X_reset = X_test_sc.reset_index(drop=True)
                        idx = np.random.RandomState(42).choice(len(X_reset),min(200,len(X_reset)),replace=False)
                        X_shap = X_reset.iloc[idx]
                        shap_vals = shap.TreeExplainer(xgb_m).shap_values(X_shap)
                        fig,_ = plt.subplots(figsize=(9,5))
                        fig.patch.set_facecolor('#0f172a')
                        shap.summary_plot(shap_vals,X_shap,plot_type="bar",show=False,max_display=12)
                        plt.gcf().patch.set_facecolor('#0f172a')
                        st.pyplot(plt.gcf()); plt.close()
                else:
                    st.info("XGBoost was not trained.")

    page_nav(prev_label="Train Models", next_label="Predict")

# ════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ════════════════════════════════════════════════════════════════
elif st.session_state.page == "Predict":
    st.markdown("<div class='hero-title' style='font-size:1.8rem'>🔮 Predict Churn</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Enter customer attributes to get a real-time churn probability.</div><br>", unsafe_allow_html=True)

    if not st.session_state.trained:
        st.warning("Please train models first.")
        if st.button("Go to Train Models"):
            st.session_state.page = "Train Models"; st.rerun()
    else:
        best_name    = st.session_state.best_model_name
        best_model   = st.session_state.results[best_name]["model"]
        scaler       = st.session_state.scaler
        feature_names= st.session_state.feature_names_all

        st.markdown(f"Using: <span class='badge'>{best_name}</span>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("predict_form"):
            st.markdown("#### 📋 Customer Profile")
            c1,c2,c3 = st.columns(3)
            with c1:
                tenure          = st.number_input("Tenure (months)", 0, 120, 12)
                monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, 5.0)
                total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0, 800.0, 50.0)
            with c2:
                contract = st.selectbox("Contract Type",["Month-to-month","One year","Two year"])
                payment  = st.selectbox("Payment Method",["Electronic check","Mailed check",
                                        "Bank transfer (automatic)","Credit card (automatic)"])
                internet = st.selectbox("Internet Service",["DSL","Fiber optic","No"])
            with c3:
                senior     = st.selectbox("Senior Citizen",["No","Yes"])
                partner    = st.selectbox("Partner",["No","Yes"])
                dependents = st.selectbox("Dependents",["No","Yes"])
                paperless  = st.selectbox("Paperless Billing",["Yes","No"])
            submitted = st.form_submit_button("🔮 Predict Churn Risk")

        if submitted:
            inp = {f:0 for f in feature_names}
            for col,val in [
                ('Tenure_Months',tenure),('Monthly_Charges',monthly_charges),
                ('Total_Charges',total_charges),
                ('Senior_Citizen',1 if senior=="Yes" else 0),
                ('Partner',1 if partner=="Yes" else 0),
                ('Dependents',1 if dependents=="Yes" else 0),
                ('Paperless_Billing',1 if paperless=="Yes" else 0),
            ]:
                if col in inp: inp[col] = val

            for prefix,chosen in [("Contract_",contract),
                                   ("Payment_Method_",payment),
                                   ("Internet_Service_",internet)]:
                key = f"{prefix}{chosen}"
                if key in inp: inp[key] = 1

            input_df     = pd.DataFrame([inp])
            input_scaled = scaler.transform(input_df)
            prob = best_model.predict_proba(input_scaled)[0][1]

            color = "#f87171" if prob>0.5 else "#48c78e"
            label = "HIGH CHURN RISK" if prob>0.5 else "LOW CHURN RISK"
            emoji = "⚠️" if prob>0.5 else "✅"

            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#0f172a,#1e293b);
                        border:1px solid {color}40;border-radius:16px;
                        padding:30px;text-align:center;margin-top:20px;'>
                <div style='font-size:2.5rem;margin-bottom:8px;'>{emoji}</div>
                <div style='font-family:Space Mono,monospace;font-size:2.6rem;
                            font-weight:700;color:{color};'>{prob*100:.1f}%</div>
                <div style='font-size:0.85rem;font-weight:700;letter-spacing:0.15em;
                            color:{color};margin-top:6px;'>{label}</div>
                <div style='color:#475569;font-size:0.8rem;margin-top:12px;'>
                    Computed by {best_name}
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fig,ax = dark_fig((7,1.2))
            ax.barh(['Risk'],[prob],color=color,height=0.4,alpha=0.85)
            ax.barh(['Risk'],[1-prob],left=[prob],color='#1e293b',height=0.4)
            ax.set_xlim(0,1)
            ax.set_xticks([0,0.25,0.5,0.75,1.0])
            ax.set_xticklabels(['0%','25%','50%','75%','100%'])
            ax.set_title("Churn Probability Gauge")
            fig.tight_layout(); st.pyplot(fig); plt.close()

    page_nav(prev_label="Results")