# ================================================================
#  Customer Churn Predictor Pro
#  - Auto data cleaning on upload (NaN, types, outliers, encoding)
#  - 4 ML models with full evaluation
#  - Smart predict page
# ================================================================

# ── auto-install missing packages ────────────────────────────────
import subprocess, sys

def _ensure(pkg, import_as=None):
    try:
        __import__(import_as or pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure("openpyxl")
_ensure("xgboost")
_ensure("imbalanced-learn", "imblearn")
# ─────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             precision_score, recall_score, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ================================================================
# CONSTANTS
# ================================================================
# Columns that are never useful for prediction
JUNK_COLS = [
    'customerid','customer_id','lat_long','zip_code','city','state',
    'country','churn_reason','count','latitude','longitude',
    'phone_number','phone','email','name','customer_name'
]
# All possible target/leakage columns (normalised)
TARGET_COLS = ['churn_label','churn_value','churn_score','churn_category']

P = {   # colour palette
    "primary" : "#4F8EF7",
    "success" : "#27AE60",
    "danger"  : "#E74C3C",
    "warning" : "#F39C12",
    "purple"  : "#A78BFA",
    "bg"      : "#0E1117",
    "card"    : "#1A1F2E",
    "border"  : "#2A3050",
    "text"    : "#F0F2F6",
    "muted"   : "#7C8598",
}

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(page_title="Churn Predictor Pro",
                   layout="wide", page_icon="📊",
                   initial_sidebar_state="expanded")

# ================================================================
# CSS
# ================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400&family=JetBrains+Mono:wght@400;600&display=swap');

html,body,[class*="css"]{{font-family:'Plus Jakarta Sans',sans-serif;}}
.block-container{{padding:1.8rem 2.8rem 3rem;max-width:1400px;}}

/* hero */
.hero{{background:linear-gradient(135deg,#141929 0%,#0E1117 65%);
       border:1px solid {P['border']};border-radius:18px;
       padding:2rem 2.6rem;margin-bottom:1.8rem;
       position:relative;overflow:hidden;}}
.hero::after{{content:'';position:absolute;top:-80px;right:-80px;
              width:300px;height:300px;border-radius:50%;
              background:radial-gradient(circle,rgba(79,142,247,.15) 0%,transparent 68%);}}
.hero-title{{font-size:1.75rem;font-weight:800;margin:0 0 .35rem;
             background:linear-gradient(90deg,{P['primary']},{P['purple']});
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
.hero-sub{{color:{P['muted']};font-size:.88rem;margin:0;line-height:1.6;}}

/* cards */
.kpi{{background:{P['card']};border:1px solid {P['border']};border-radius:12px;
      padding:1.1rem 1.4rem;text-align:center;margin-bottom:.5rem;}}
.kpi-val{{font-size:1.65rem;font-weight:800;font-family:'JetBrains Mono',monospace;
          color:{P['primary']};}}
.kpi-lbl{{font-size:.7rem;color:{P['muted']};text-transform:uppercase;
          letter-spacing:.09em;margin-top:.25rem;}}

.feature-card{{background:{P['card']};border:1px solid {P['border']};
               border-radius:12px;padding:1.3rem;height:100%;}}
.fc-icon{{font-size:1.7rem;margin-bottom:.5rem;}}
.fc-title{{font-weight:700;font-size:.92rem;margin:.3rem 0;}}
.fc-desc{{color:{P['muted']};font-size:.8rem;line-height:1.55;}}

/* clean report */
.clean-row{{display:flex;align-items:center;gap:.6rem;
            padding:.45rem .8rem;border-radius:8px;
            background:rgba(79,142,247,.06);margin-bottom:.35rem;
            font-size:.83rem;color:{P['text']};}}
.clean-badge{{background:{P['primary']};color:#fff;border-radius:6px;
              padding:.15rem .55rem;font-size:.72rem;font-weight:700;
              white-space:nowrap;}}
.clean-badge.ok{{background:{P['success']};}}
.clean-badge.warn{{background:{P['warning']};}}

/* result banners */
.res-high{{background:linear-gradient(135deg,rgba(231,76,60,.18),rgba(231,76,60,.05));
           border:1px solid rgba(231,76,60,.45);border-radius:14px;
           padding:1.6rem 2rem;text-align:center;}}
.res-low {{background:linear-gradient(135deg,rgba(39,174,96,.18),rgba(39,174,96,.05));
           border:1px solid rgba(39,174,96,.45);border-radius:14px;
           padding:1.6rem 2rem;text-align:center;}}
.res-icon{{font-size:2.8rem;margin-bottom:.4rem;}}
.res-title{{font-size:1.35rem;font-weight:800;margin-bottom:.25rem;}}
.res-sub{{color:{P['muted']};font-size:.88rem;}}

/* info strip */
.info-strip{{background:rgba(79,142,247,.07);border-left:3px solid {P['primary']};
             border-radius:0 8px 8px 0;padding:.65rem 1rem;
             font-size:.83rem;color:{P['muted']};margin:.3rem 0 1rem;}}

/* sidebar */
section[data-testid="stSidebar"]{{background:{P['card']};
                                   border-right:1px solid {P['border']};}}
/* buttons */
.stButton>button{{background:linear-gradient(135deg,{P['primary']},#7C69EF)!important;
                  color:#fff!important;border:none!important;border-radius:9px!important;
                  font-weight:700!important;letter-spacing:.02em!important;
                  padding:.62rem 2rem!important;transition:opacity .18s!important;}}
.stButton>button:hover{{opacity:.82!important;}}
.stProgress>div>div{{background:{P['primary']}!important;}}
div[data-testid="stMetric"]{{background:{P['card']};border:1px solid {P['border']};
                              border-radius:10px;padding:.7rem 1rem;}}
</style>
""", unsafe_allow_html=True)

# ================================================================
# SESSION STATE
# ================================================================
for k,v in dict(df_raw=None, df_clean=None, clean_report=None,
                model=None, scaler=None, feature_columns=None,
                raw_X=None, model_results=None,
                best_model_name=None, all_trained_models=None).items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================================================================
# ██████  DATA CLEANING ENGINE  ██████████████████████████████████
# ================================================================
def clean_dataframe(df: pd.DataFrame):
    """
    Full automatic cleaning pipeline.
    Returns (df_cleaned, report_list)
    report_list = [ {"col": ..., "action": ..., "detail": ...}, ... ]
    """
    df   = df.copy()
    rep  = []   # cleaning report

    # 1. normalise column names
    df.columns = (df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(r"[^a-z0-9]+", "_", regex=True)
                    .str.strip("_"))

    # 2. drop pure-ID / geo junk columns
    to_drop = [c for c in df.columns if c in JUNK_COLS]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)
        rep.append({"action":"Dropped ID/geo columns",
                    "detail": ", ".join(to_drop), "badge":"warn"})

    # 3. drop columns that are 100 % null
    all_null = [c for c in df.columns if df[c].isnull().all()]
    if all_null:
        df.drop(columns=all_null, inplace=True)
        rep.append({"action":"Dropped all-null columns",
                    "detail": ", ".join(all_null), "badge":"warn"})

    # 4. drop columns with > 70 % missing (not the target)
    high_null = [c for c in df.columns
                 if c not in TARGET_COLS
                 and df[c].isnull().mean() > 0.70]
    if high_null:
        df.drop(columns=high_null, inplace=True)
        rep.append({"action":"Dropped >70 % missing columns",
                    "detail": ", ".join(high_null), "badge":"warn"})

    # 5. drop constant columns
    const_cols = [c for c in df.columns
                  if c not in TARGET_COLS and df[c].nunique(dropna=False) <= 1]
    if const_cols:
        df.drop(columns=const_cols, inplace=True)
        rep.append({"action":"Dropped constant columns",
                    "detail": ", ".join(const_cols), "badge":"warn"})

    # 6a. replace "nan"/"none"/""/" " string literals with real NaN (Excel exports etc.)
    NAN_STRINGS = {"nan","none","null","n/a","na","N/A","NA","<na>","<NA>"}
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].where(~df[col].isin(NAN_STRINGS) & (df[col] != ""), other=np.nan)

    # 6b. fix mixed-type numeric columns (e.g. "Total Charges" = " ")
    for col in df.columns:
        if col in TARGET_COLS:
            continue
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notnull().mean() > 0.60:
                df[col] = converted
                rep.append({"action": "Converted to numeric",
                            "detail": col, "badge": "ok"})

    # 7. fill missing values — numeric → median, categorical → mode
    num_cols_now = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols_now = df.select_dtypes(exclude=np.number).columns.tolist()

    filled_num, filled_cat = [], []
    for col in num_cols_now:
        if col in TARGET_COLS:
            continue
        n_miss = int(df[col].isnull().sum())
        if n_miss > 0:
            med = df[col].median()
            df[col].fillna(med if pd.notnull(med) else 0, inplace=True)
            filled_num.append(f"{col}({n_miss})")
    for col in cat_cols_now:
        if col in TARGET_COLS:
            continue
        n_miss = int(df[col].isnull().sum())
        if n_miss > 0:
            mode_val = df[col].mode()
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col].fillna(fill_val, inplace=True)
            filled_cat.append(f"{col}({n_miss})")

    if filled_num:
        rep.append({"action":"Filled numeric NaNs with median",
                    "detail": ", ".join(filled_num), "badge":"ok"})
    if filled_cat:
        rep.append({"action":"Filled categorical NaNs with mode",
                    "detail": ", ".join(filled_cat), "badge":"ok"})

    # 8. drop duplicate rows
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        df.drop_duplicates(inplace=True)
        rep.append({"action":"Removed duplicate rows",
                    "detail": f"{n_dup} duplicates removed", "badge":"warn"})

    # 9. fix target column — must be int 0/1
    if "churn_value" in df.columns:
        df["churn_value"] = pd.to_numeric(df["churn_value"], errors="coerce")
        df.dropna(subset=["churn_value"], inplace=True)
        df["churn_value"] = df["churn_value"].astype(int)
        rep.append({"action":"Target column validated",
                    "detail": "churn_value → int 0/1", "badge":"ok"})

    # 10. drop high-cardinality text columns (> 50 unique, not numeric)
    # except the target — these are usually free-text and can't be encoded usefully
    cat_cols_now2 = df.select_dtypes(exclude=np.number).columns.tolist()
    high_card = [c for c in cat_cols_now2
                 if c not in TARGET_COLS and df[c].nunique() > 50]
    if high_card:
        df.drop(columns=high_card, inplace=True)
        rep.append({"action":"Dropped high-cardinality text columns (>50 unique)",
                    "detail": ", ".join(high_card), "badge":"warn"})

    if not rep:
        rep.append({"action":"Dataset already clean", "detail":"No issues found","badge":"ok"})

    return df, rep


# ================================================================
# HELPERS
# ================================================================
def build_features(df_clean: pd.DataFrame):
    """Returns (X_raw, y, err).  df_clean already has normalised col names."""
    # drop remaining leakage cols except churn_value
    drop = [c for c in TARGET_COLS if c in df_clean.columns and c != "churn_value"]
    X = df_clean.drop(columns=drop + ["churn_value"], errors="ignore")
    if "churn_value" not in df_clean.columns:
        return None, None, "'churn_value' column missing after cleaning."
    y = df_clean["churn_value"].astype(int)
    return X, y, None

def encode_align(X_raw: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    X_enc = pd.get_dummies(X_raw)
    for col in feature_columns:
        if col not in X_enc.columns:
            X_enc[col] = 0
    return X_enc[feature_columns]

def plt_dark():
    plt.rcParams.update({
        "figure.facecolor": P["bg"],  "axes.facecolor":   P["card"],
        "axes.edgecolor":   P["border"],"axes.labelcolor": P["muted"],
        "xtick.color":      P["muted"],"ytick.color":      P["muted"],
        "text.color":       P["text"], "grid.color":       P["border"],
        "grid.linestyle":   "--","grid.alpha":.4,"axes.grid":True,
        "axes.spines.top":  False,"axes.spines.right":False,
    })
plt_dark()

def hero(title, sub=""):
    st.markdown(f"""
    <div class='hero'>
        <div class='hero-title'>{title}</div>
        <p class='hero-sub'>{sub}</p>
    </div>""", unsafe_allow_html=True)

def kpi_row(items):
    """items = [(value, label, colour?), ...]"""
    cols = st.columns(len(items))
    for col,(val,lbl,*rest) in zip(cols,items):
        clr = rest[0] if rest else P["primary"]
        col.markdown(f"""
        <div class='kpi'>
            <div class='kpi-val' style='color:{clr}'>{val}</div>
            <div class='kpi-lbl'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:.8rem 0 1.4rem'>
        <div style='font-size:2.2rem'>📊</div>
        <div style='font-weight:800;font-size:1rem;
                    background:linear-gradient(90deg,{P['primary']},{P['purple']});
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            Churn Predictor
        </div>
        <div style='color:{P['muted']};font-size:.72rem;margin-top:.15rem'>Pro Edition</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("nav", label_visibility="collapsed", options=[
        "🏠  Home", "📂  Upload & Clean", "🔍  Explore Data",
        "🤖  Train Model", "🔮  Predict"
    ])

    st.markdown("---")
    def _dot(ok): return "🟢" if ok else "⚪"
    st.markdown(f"""
    <div style='font-size:.8rem;color:{P['muted']};line-height:2.2'>
        {_dot(st.session_state.df_raw   is not None)} &nbsp;Dataset uploaded<br>
        {_dot(st.session_state.df_clean is not None)} &nbsp;Data cleaned<br>
        {_dot(st.session_state.model    is not None)} &nbsp;Model trained<br>
        {_dot(st.session_state.model    is not None)} &nbsp;Ready to predict
    </div>""", unsafe_allow_html=True)

# ================================================================
# HOME
# ================================================================
if page == "🏠  Home":
    hero("Customer Churn Predictor Pro",
         "Upload your dataset → auto-clean → explore → train 4 ML models → predict churn risk")

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(num,title,desc) in zip([c1,c2,c3,c4,c5],[
        ("01","Upload","Drop a CSV or Excel file — any messy format works."),
        ("02","Auto Clean","NaNs, types, duplicates & outliers fixed automatically."),
        ("03","Explore","Charts, distributions, correlations at a glance."),
        ("04","Train","4 ML models trained & ranked by ROC-AUC automatically."),
        ("05","Predict","Enter one customer's details for instant risk scoring."),
    ]):
        col.markdown(f"""
        <div class='feature-card'>
            <div style='font-size:1.6rem;font-weight:800;
                        background:linear-gradient(90deg,{P['primary']},{P['purple']});
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>{num}</div>
            <div class='fc-title'>{title}</div>
            <div class='fc-desc'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🤖 ML Models Used")
    m1,m2,m3,m4 = st.columns(4)
    for col,(ico,name,desc) in zip([m1,m2,m3,m4],[
        ("🌲","Random Forest",      "Robust ensemble, handles noise"),
        ("⚡","XGBoost",            "State-of-the-art boosting"),
        ("📈","Logistic Regression","Fast interpretable baseline"),
        ("🚀","AdaBoost",           "Adaptive boosting"),
    ]):
        col.markdown(f"""
        <div class='feature-card' style='text-align:center'>
            <div class='fc-icon'>{ico}</div>
            <div class='fc-title'>{name}</div>
            <div class='fc-desc'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📋 Expected Dataset Format")
    st.markdown(f"""<div class='info-strip'>
        Needs a <b>Churn Value</b> column (0 = stays, 1 = churns).
        Everything else is cleaned automatically — nulls, mixed types, duplicates, high-cardinality columns.
    </div>""", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "CustomerID":      ["001","002","003"],
        "Tenure Months":   [12,5,24],
        "Monthly Charges": [65.5,89.0,45.2],
        "Total Charges":   ["786","445"," "],   # ← messy on purpose
        "Contract":        ["Month-to-month","Two year","One year"],
        "Internet Service":["Fiber optic","DSL","No"],
        "Churn Value":     [1,0,0],
    }), use_container_width=True, hide_index=True)


# ================================================================
# UPLOAD & CLEAN
# ================================================================
elif page == "📂  Upload & Clean":
    hero("📂 Upload & Auto-Clean",
         "Upload any CSV or Excel file — the cleaning engine handles the rest")

    file = st.file_uploader("Drag & drop or browse", type=["csv","xlsx"],
                             label_visibility="collapsed")
    if file:
        with st.spinner("Reading file…"):
            try:
                if file.name.lower().endswith(".csv"):
                    df_raw = None
                    for enc in ("utf-8","latin-1","cp1252"):
                        try:
                            file.seek(0)
                            df_raw = pd.read_csv(file, encoding=enc)
                            break
                        except UnicodeDecodeError:
                            continue
                    if df_raw is None:
                        st.error("❌ Cannot decode CSV. Save it as UTF-8 and retry.")
                        st.stop()
                else:
                    file.seek(0)
                    df_raw = pd.read_excel(file, engine="openpyxl")

                if df_raw.empty or df_raw.shape[1] < 2:
                    st.error("❌ File appears empty or has only 1 column.")
                    st.stop()

                st.session_state.df_raw = df_raw

            except Exception as e:
                st.error(f"❌ File read error: {e}")
                st.stop()

        # ── run cleaning engine ──────────────────────────────────
        with st.spinner("🧹 Cleaning dataset…"):
            df_clean, report = clean_dataframe(df_raw)
            st.session_state.df_clean    = df_clean
            st.session_state.clean_report = report

        # ── stats ────────────────────────────────────────────────
        st.success(f"✅ **{file.name}** loaded and cleaned!")
        raw_nulls    = int(df_raw.isnull().sum().sum())
        clean_nulls  = int(df_clean.isnull().sum().sum())
        churn_ok = "churn_value" in df_clean.columns

        kpi_row([
            (f"{df_raw.shape[0]:,}",        "Raw rows",        P["primary"]),
            (f"{df_clean.shape[0]:,}",       "Clean rows",      P["success"]),
            (f"{df_raw.shape[1]}→{df_clean.shape[1]}","Columns kept", P["warning"]),
            (f"{raw_nulls:,}→{clean_nulls}", "Missing cells",   P["danger"]),
            ("✅" if churn_ok else "❌",      "Target detected", P["success"] if churn_ok else P["danger"]),
        ])

        if not churn_ok:
            st.error("❌ 'Churn Value' column not found. Training won't work.")

        # ── cleaning report ──────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🧹 Cleaning Report")
        for r in report:
            badge_cls = r.get("badge","ok")
            st.markdown(f"""
            <div class='clean-row'>
                <span class='clean-badge {badge_cls}'>{r['action']}</span>
                <span style='color:{P['muted']}'>{r['detail']}</span>
            </div>""", unsafe_allow_html=True)

        # ── side-by-side preview ─────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["✅ Cleaned Dataset", "📄 Raw Dataset"])
        with tab1:
            st.dataframe(df_clean.head(15), use_container_width=True, hide_index=True)
        with tab2:
            st.dataframe(df_raw.head(15),   use_container_width=True, hide_index=True)

        # ── column type summary ──────────────────────────────────
        st.markdown("#### 📋 Column Summary (after cleaning)")
        st.dataframe(pd.DataFrame({
            "Column":    df_clean.columns,
            "Type":      df_clean.dtypes.astype(str).values,
            "Non-null":  df_clean.notnull().sum().values,
            "Nulls":     df_clean.isnull().sum().values,
            "Unique":    df_clean.nunique().values,
            "Sample":    [str(df_clean[c].dropna().iloc[0])
                          if df_clean[c].notnull().any() else "—"
                          for c in df_clean.columns],
        }), use_container_width=True, hide_index=True)


# ================================================================
# EXPLORE DATA
# ================================================================
elif page == "🔍  Explore Data":
    df_clean = st.session_state.df_clean
    if df_clean is None:
        st.warning("⚠️ Please upload and clean a dataset first.")
        st.stop()

    hero("🔍 Exploratory Data Analysis", "Visualise patterns before training")

    # churn stats
    if "churn_value" in df_clean.columns:
        total   = len(df_clean)
        churned = int(df_clean["churn_value"].sum())
        stayed  = total - churned
        kpi_row([
            (f"{total:,}",            "Total Customers",  P["primary"]),
            (f"{stayed:,}",           "Retained",         P["success"]),
            (f"{churned:,}",          "Churned",          P["danger"]),
            (f"{churned/total*100:.1f}%","Churn Rate",    P["warning"]),
            (f"{df_clean.shape[1]}",  "Features",         P["purple"]),
        ])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Churn Distribution")
        fig, axes = plt.subplots(1,2, figsize=(11,4), facecolor=P["bg"])
        axes[0].pie([stayed,churned], labels=["Retained","Churned"],
                    autopct="%1.1f%%",
                    colors=[P["success"],P["danger"]], startangle=90,
                    wedgeprops=dict(width=0.55,edgecolor=P["bg"],linewidth=3),
                    textprops={"color":P["text"],"fontsize":11})
        axes[0].set_title("Churn Split", color=P["text"], fontweight="bold",pad=14)
        bars = axes[1].bar(["Retained","Churned"],[stayed,churned],
                            color=[P["success"],P["danger"]],
                            width=0.42, edgecolor="none")
        for bar,v in zip(bars,[stayed,churned]):
            axes[1].text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+total*0.01,
                         f"{v:,}", ha="center", fontweight="bold",
                         color=P["text"], fontsize=11)
        axes[1].set_title("Count", color=P["text"], fontweight="bold",pad=14)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    # numeric distributions
    st.markdown("#### Numeric Feature Distributions")
    num_cols = [c for c in df_clean.select_dtypes(include=np.number).columns
                if c not in TARGET_COLS]
    if num_cols:
        sel = st.multiselect("Select columns", num_cols,
                             default=num_cols[:min(4,len(num_cols))])
        if sel:
            nc = min(len(sel),4)
            nr = (len(sel)+nc-1)//nc
            fig,axes = plt.subplots(nr,nc,figsize=(5.5*nc,4*nr),
                                    facecolor=P["bg"])
            axes = np.array(axes).flatten()
            for idx,col in enumerate(sel):
                ax = axes[idx]
                if "churn_value" in df_clean.columns:
                    for v,clr,lbl in [(0,P["success"],"Retained"),
                                      (1,P["danger"],"Churned")]:
                        ax.hist(df_clean[df_clean["churn_value"]==v][col].dropna(),
                                alpha=0.65,color=clr,label=lbl,bins=30,edgecolor="none")
                    ax.legend(fontsize=8)
                else:
                    ax.hist(df_clean[col].dropna(),bins=30,
                            color=P["primary"],edgecolor="none",alpha=0.8)
                ax.set_title(col.replace("_"," "),fontweight="bold",color=P["text"])
            for j in range(len(sel),len(axes)): axes[j].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

    # categorical vs churn
    st.markdown("#### Categorical Feature vs Churn")
    cat_cols = [c for c in df_clean.select_dtypes(exclude=np.number).columns
                if c not in TARGET_COLS and df_clean[c].nunique() <= 20]
    if cat_cols and "churn_value" in df_clean.columns:
        sel_cat = st.selectbox("Select categorical column", cat_cols)
        ct = (df_clean.groupby([sel_cat,"churn_value"])
                       .size().unstack(fill_value=0))
        ct.columns = ["Retained","Churned"] if 0 in ct.columns else ct.columns
        fig,ax = plt.subplots(figsize=(max(8,len(ct)*1.3),4),
                              facecolor=P["bg"])
        x = np.arange(len(ct)); w = 0.38
        ax.bar(x-w/2, ct.iloc[:,0], width=w, color=P["success"],
               label=ct.columns[0], edgecolor="none")
        ax.bar(x+w/2, ct.iloc[:,1], width=w, color=P["danger"],
               label=ct.columns[1], edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(ct.index, rotation=30, ha="right")
        ax.legend()
        ax.set_title(f"{sel_cat.replace('_',' ')} vs Churn",
                     fontweight="bold", color=P["text"])
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    # correlation heatmap
    st.markdown("#### Correlation Heatmap")
    num_df = df_clean.select_dtypes(include=np.number)
    if len(num_df.columns) > 1:
        corr = num_df.corr()
        mask = np.triu(np.ones_like(corr,dtype=bool))
        h = max(6, len(corr)*0.65)
        fig,ax = plt.subplots(figsize=(max(8,len(corr)*0.8),h),
                              facecolor=P["bg"])
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax,
                    linewidths=0.4, linecolor=P["bg"],
                    annot_kws={"size":8}, cbar_kws={"shrink":.8})
        ax.set_title("Feature Correlations",fontweight="bold",color=P["text"])
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()


# ================================================================
# TRAIN MODEL
# ================================================================
elif page == "🤖  Train Model":
    df_clean = st.session_state.df_clean
    if df_clean is None:
        st.warning("⚠️ Please upload and clean a dataset first.")
        st.stop()

    hero("🤖 Model Training", "Auto-train 4 ML models and pick the best one")

    X_raw, y, err = build_features(df_clean)
    if err:
        st.error(f"❌ {err}"); st.stop()

    kpi_row([
        (f"{X_raw.shape[1]}",   "Features",        P["primary"]),
        (f"{X_raw.shape[0]:,}", "Samples",          P["purple"]),
        (f"{y.mean()*100:.1f}%","Churn Rate",       P["warning"]),
        (f"{int(y.sum()):,}",   "Positive Samples", P["danger"]),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ⚙️ Training Settings")
    s1,s2,s3 = st.columns(3)
    test_size   = s1.slider("Test split", 0.10, 0.40, 0.20, 0.05)
    apply_smote = s2.checkbox("Apply SMOTE", value=True,
                              help="Oversample minority class")
    n_est       = s3.select_slider("Estimators (RF/XGB/Ada)",
                                   options=[50,100,150,200,300], value=150)

    if st.button("🚀  Train All Models", use_container_width=True):

        # ── encode ───────────────────────────────────────────────
        with st.spinner("Encoding features…"):
            # Step 1: force all object cols to string so get_dummies works cleanly
            X_raw_clean = X_raw.copy()
            for col in X_raw_clean.select_dtypes(include="object").columns:
                X_raw_clean[col] = X_raw_clean[col].astype(str).str.strip()
                X_raw_clean[col].replace({"nan":"Unknown","none":"Unknown",
                                          "":"Unknown","<na>":"Unknown"}, inplace=True)

            # Step 2: for numeric cols, fill NaN with median before encoding
            for col in X_raw_clean.select_dtypes(include=np.number).columns:
                if X_raw_clean[col].isnull().any():
                    X_raw_clean[col].fillna(X_raw_clean[col].median(), inplace=True)

            # Step 3: one-hot encode
            X_enc = pd.get_dummies(X_raw_clean)

            # Step 4: final catch-all — fill any remaining NaN with 0
            if X_enc.isnull().any().any():
                X_enc.fillna(0, inplace=True)

            # Step 5: ensure all columns are float (some bool dummies cause issues)
            X_enc = X_enc.astype(float)

            X_train,X_test,y_train,y_test = train_test_split(
                X_enc, y, test_size=test_size, random_state=42, stratify=y)

        # ── scale ────────────────────────────────────────────────
        with st.spinner("Scaling…"):
            scaler    = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

        # ── SMOTE ────────────────────────────────────────────────
        if apply_smote:
            with st.spinner("Applying SMOTE…"):
                try:
                    X_train_s, y_train = SMOTE(random_state=42).fit_resample(
                        X_train_s, y_train)
                except Exception as se:
                    st.warning(f"SMOTE skipped: {se}")

        # ── train ─────────────────────────────────────────────────
        models_cfg = {
            "Random Forest":       RandomForestClassifier(
                                       n_estimators=n_est,random_state=42,n_jobs=-1),
            "XGBoost":             XGBClassifier(
                                       n_estimators=n_est,eval_metric="logloss",
                                       random_state=42,verbosity=0,n_jobs=-1),
            "Logistic Regression": LogisticRegression(
                                       max_iter=3000,random_state=42,n_jobs=-1),
            "AdaBoost":            AdaBoostClassifier(
                                       n_estimators=min(n_est,200),random_state=42),
        }

        results, trained_models = {}, {}
        prog   = st.progress(0, text="Starting…")
        status = st.empty()

        for i,(name,mdl) in enumerate(models_cfg.items()):
            status.info(f"⏳ Training **{name}**  ({i+1}/{len(models_cfg)})…")
            prog.progress((i+1)/len(models_cfg))
            try:
                mdl.fit(X_train_s, y_train)
                pred = mdl.predict(X_test_s)
                prob = mdl.predict_proba(X_test_s)[:,1]
                results[name] = {
                    "Accuracy": round(accuracy_score(y_test,pred),4),
                    "ROC-AUC":  round(roc_auc_score(y_test,prob),4),
                }
                trained_models[name] = mdl
            except Exception as me:
                st.warning(f"⚠️ {name} failed: {me}")

        prog.empty(); status.empty()

        if not results:
            st.error("❌ All models failed. Check your dataset."); st.stop()

        # ── results ───────────────────────────────────────────────
        st.markdown("#### 📊 Model Comparison")
        results_df = (pd.DataFrame([{"Model":k,**v} for k,v in results.items()])
                        .sort_values("ROC-AUC",ascending=False)
                        .reset_index(drop=True))

        st.dataframe(
            results_df.style
              .highlight_max(subset=["Accuracy","ROC-AUC"],color="#152a1e")
              .format({"Accuracy":"{:.4f}","ROC-AUC":"{:.4f}"}),
            use_container_width=True, hide_index=True)

        # bar chart
        fig,axes = plt.subplots(1,2,figsize=(12,4),facecolor=P["bg"])
        for ax,metric in zip(axes,["Accuracy","ROC-AUC"]):
            clrs = [P["success"] if i==0 else P["primary"] for i in range(len(results_df))]
            bars = ax.barh(results_df["Model"],results_df[metric],
                           color=clrs,edgecolor="none",height=0.5)
            ax.set_xlim(results_df[metric].min()*0.96,1.03)
            ax.set_title(metric,fontweight="bold",color=P["text"])
            for bar,val in zip(bars,results_df[metric]):
                ax.text(val+0.002,bar.get_y()+bar.get_height()/2,
                        f"{val:.4f}",va="center",
                        color=P["text"],fontweight="bold",fontsize=9)
        plt.tight_layout()
        st.pyplot(fig,use_container_width=True); plt.close()

        best_name  = results_df.iloc[0]["Model"]
        best_model = trained_models[best_name]
        st.success(f"🏆 **Best Model: {best_name}** — ROC-AUC: {results_df.iloc[0]['ROC-AUC']}")

        # confusion matrix + metrics
        st.markdown(f"#### Confusion Matrix & Metrics — {best_name}")
        bp = best_model.predict(X_test_s)
        cm = confusion_matrix(y_test,bp)
        fig,axes = plt.subplots(1,2,figsize=(12,4.2),facecolor=P["bg"])
        ConfusionMatrixDisplay(cm,display_labels=["Retained","Churned"]).plot(
            ax=axes[0],cmap="Blues",colorbar=False)
        axes[0].set_title("Confusion Matrix",fontweight="bold",color=P["text"])
        mvals = {
            "Precision": precision_score(y_test,bp,zero_division=0),
            "Recall":    recall_score(y_test,   bp,zero_division=0),
            "F1 Score":  f1_score(y_test,        bp,zero_division=0),
            "Accuracy":  accuracy_score(y_test,  bp),
        }
        b2 = axes[1].barh(list(mvals),list(mvals.values()),
                           color=P["primary"],edgecolor="none",height=0.45)
        axes[1].set_xlim(0,1.14)
        axes[1].set_title("Classification Metrics",fontweight="bold",color=P["text"])
        for bar,val in zip(b2,mvals.values()):
            axes[1].text(val+0.01,bar.get_y()+bar.get_height()/2,
                         f"{val:.4f}",va="center",
                         color=P["text"],fontweight="bold",fontsize=9)
        plt.tight_layout()
        st.pyplot(fig,use_container_width=True); plt.close()

        # feature importance
        if hasattr(best_model,"feature_importances_"):
            st.markdown("#### 🌟 Top 15 Feature Importances")
            imp   = pd.Series(best_model.feature_importances_,index=X_enc.columns)
            top15 = imp.nlargest(15).sort_values()
            clrs  = [P["primary"]]*len(top15); clrs[-1] = P["success"]
            fig,ax = plt.subplots(figsize=(9,5),facecolor=P["bg"])
            ax.barh(top15.index,top15.values,color=clrs,edgecolor="none")
            ax.set_title("Feature Importances (Top 15)",
                         fontweight="bold",color=P["text"])
            ax.set_xlabel("Importance Score")
            plt.tight_layout()
            st.pyplot(fig,use_container_width=True); plt.close()

        # save to session
        st.session_state.model              = best_model
        st.session_state.scaler             = scaler
        st.session_state.feature_columns    = X_enc.columns.tolist()
        st.session_state.raw_X              = X_raw
        st.session_state.model_results      = results_df
        st.session_state.best_model_name    = best_name
        st.session_state.all_trained_models = trained_models


# ================================================================
# PREDICT
# ================================================================
elif page == "🔮  Predict":
    model      = st.session_state.model
    scaler     = st.session_state.scaler
    feat_cols  = st.session_state.feature_columns
    raw_X      = st.session_state.raw_X
    all_models = st.session_state.all_trained_models

    if model is None:
        st.warning("⚠️ Please train the model first.")
        st.stop()

    hero("🔮 Predict Customer Churn",
         "Enter customer attributes to get an instant churn probability")

    chosen_name  = st.selectbox("Select model",
                                list(all_models.keys()) if all_models else
                                [st.session_state.best_model_name], index=0)
    chosen_model = all_models[chosen_name] if all_models else model
    st.markdown(f"<div class='info-strip'>Using <b>{chosen_name}</b></div>",
                unsafe_allow_html=True)

    st.markdown("#### Enter Customer Details")
    num_cols = raw_X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = raw_X.select_dtypes(exclude=np.number).columns.tolist()
    inp = {}

    left,right = st.columns(2)
    with left:
        if num_cols:
            st.markdown("**🔢 Numeric Features**")
            for col in num_cols:
                d = pd.to_numeric(raw_X[col],errors="coerce").dropna()
                if d.empty: inp[col]=0.0; continue
                inp[col] = st.number_input(
                    col.replace("_"," ").title(),
                    min_value=float(d.min()), max_value=float(d.max()),
                    value=round(float(d.mean()),2), key=f"n_{col}")
    with right:
        if cat_cols:
            st.markdown("**🏷️ Categorical Features**")
            for col in cat_cols:
                opts = sorted(raw_X[col].dropna().unique().tolist())
                if not opts: inp[col]=""; continue
                inp[col] = st.selectbox(col.replace("_"," ").title(),
                                        options=opts, key=f"c_{col}")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮  Get Churn Prediction", use_container_width=True):
        try:
            inp_df      = pd.DataFrame([inp])
            inp_enc     = encode_align(inp_df, feat_cols)
            # final NaN guard on predict input
            inp_enc.fillna(0, inplace=True)
            inp_scaled  = scaler.transform(inp_enc)

            pred = int(chosen_model.predict(inp_scaled)[0])
            prob = float(chosen_model.predict_proba(inp_scaled)[0][1])

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📋 Prediction Result")

            if pred == 1:
                st.markdown(f"""
                <div class='res-high'>
                    <div class='res-icon'>⚠️</div>
                    <div class='res-title' style='color:{P["danger"]}'>HIGH CHURN RISK</div>
                    <div class='res-sub'>This customer is likely to leave</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='res-low'>
                    <div class='res-icon'>✅</div>
                    <div class='res-title' style='color:{P["success"]}'>LOW CHURN RISK</div>
                    <div class='res-sub'>This customer is likely to stay</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            risk   = "High 🔴" if prob>.7 else "Medium 🟡" if prob>.4 else "Low 🟢"
            action = ("Immediate retention action" if prob>.7 else
                      "Monitor & engage proactively" if prob>.4 else
                      "No action needed")
            kpi_row([
                (f"{prob*100:.1f}%", "Churn Probability", P["danger"] if prob>.5 else P["success"]),
                (risk,               "Risk Level",         P["primary"]),
                (action,             "Recommended Action", P["warning"]),
            ])

            # gauge
            st.markdown("<br>", unsafe_allow_html=True)
            bc = P["danger"] if prob>.7 else P["warning"] if prob>.4 else P["success"]
            fig,ax = plt.subplots(figsize=(9,1.9),facecolor=P["bg"])
            ax.barh(0,1,   height=0.38,color=P["card"],edgecolor=P["border"],linewidth=1)
            ax.barh(0,prob,height=0.38,color=bc,       edgecolor="none")
            ax.axvline(prob,color=P["text"],linewidth=2.5,alpha=0.9)
            ax.text(prob,0.27,f"{prob*100:.1f}%",
                    ha="center",va="bottom",fontweight="bold",
                    color=P["text"],fontsize=13)
            for xv,lbl in [(0.4,"Medium"),(0.7,"High")]:
                ax.axvline(xv,color=P["muted"],linewidth=1,linestyle="--",alpha=0.55)
                ax.text(xv,-.28,lbl,ha="center",color=P["muted"],fontsize=7.5)
            ax.set_xlim(0,1); ax.set_ylim(-.5,.5); ax.set_yticks([])
            ax.set_xticks([0,.25,.5,.75,1])
            ax.set_xticklabels(["0%","25%","50%","75%","100%"])
            ax.set_title("Churn Probability Gauge",
                         fontweight="bold",color=P["text"],pad=10)
            plt.tight_layout()
            st.pyplot(fig,use_container_width=True); plt.close()

            # download
            st.download_button(
                "📥  Download Prediction as CSV",
                data=pd.DataFrame([{**inp,
                    "Model":chosen_name,
                    "Churn_Prediction":pred,
                    "Churn_Probability":round(prob,4),
                    "Risk_Level":risk.split()[0]}]).to_csv(index=False),
                file_name="churn_prediction.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.info("Tip: Re-train the model if you changed the dataset.")
