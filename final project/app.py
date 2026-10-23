import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from sklearn.base import BaseEstimator, TransformerMixin
class FunctionTransformerToDense(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.toarray() if hasattr(X, "toarray") else X

st.set_page_config(page_title="Amazon Product Analytics", page_icon="🛍️", layout="wide")

# Visual styling only: keeps the existing dataset processing and ML/OLAP logic unchanged.

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root { --ink:#17233c; --muted:#64748b; --blue:#356df3; --pale:#eef4ff; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f5f7fb; color: var(--ink); }
[data-testid="stHeader"] { background: rgba(245,247,251,.92); }
.block-container { padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1500px; }
h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: #17233c !important; letter-spacing: -0.035em; }
h1 { font-size: clamp(1.8rem, 3vw, 2.55rem) !important; font-weight: 800 !important; }
h2 { font-weight: 750 !important; }
h3 { font-weight: 700 !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#111d36 0%,#182b4c 100%); }
[data-testid="stSidebar"] * { color: #eef4ff !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #c4d2ea !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius: 10px; padding: 8px 10px; margin: 2px 0; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(255,255,255,.09); }
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] { gap: 3px; }
[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e7ecf5; padding: 18px 20px; border-radius: 16px; box-shadow: 0 5px 18px rgba(24,42,75,.045); }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: .88rem; font-weight: 600; }
[data-testid="stMetricValue"] { color: #17233c !important; font-family: 'Manrope',sans-serif; font-weight: 800; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid #e4eaf4; border-radius: 12px; overflow: hidden; }
.stButton > button, .stDownloadButton > button { border-radius: 10px; border: 0; font-weight: 700; padding: .55rem 1rem; transition: all .15s ease; }
.stButton > button[kind="primary"] { background: #356df3; color: white; box-shadow: 0 5px 12px rgba(53,109,243,.2); }
.stButton > button[kind="primary"]:hover { background: #2457d6; transform: translateY(-1px); }
.stDownloadButton > button { background: #eaf0ff; color: #2457d6; }
[data-testid="stSelectbox"], [data-testid="stMultiSelect"], [data-testid="stSlider"], [data-testid="stNumberInput"] { background: transparent; }
div[data-baseweb="select"] > div { border-radius: 10px; border-color: #dbe3f0; }
.stAlert { border-radius: 12px; }
hr { border-color: #e5eaf3; }
section[data-testid="stMain"] .stMarkdown p { line-height: 1.65; }
.dashboard-hero { background: linear-gradient(115deg,#172746 0%,#234b89 58%,#356df3 100%); padding: 26px 30px; border-radius: 20px; color: white; margin: 0 0 24px 0; box-shadow: 0 12px 30px rgba(31,65,128,.15); }
.dashboard-hero .eyebrow { color: #b9d0ff; text-transform: uppercase; letter-spacing: .13em; font-size: .72rem; font-weight: 800; margin-bottom: 9px; }
.dashboard-hero .hero-title { font-family: 'Manrope',sans-serif; font-size: clamp(1.5rem,3vw,2.2rem); font-weight: 800; line-height: 1.2; letter-spacing: -.04em; color: #fff; }
.dashboard-hero .hero-sub { color: #d9e6ff; margin-top: 10px; font-size: .95rem; }
.section-kicker { color:#356df3; font-size:.75rem; text-transform:uppercase; letter-spacing:.12em; font-weight:800; margin-bottom:4px; }
@media (max-width: 700px) { .block-container { padding: 1rem .8rem 2rem; } .dashboard-hero { padding: 22px; } }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dashboard-hero">
  <div class="eyebrow">E-commerce intelligence • Data mining</div>
  <div class="hero-title">Product Recommendation &amp; Analytics</div>
  <div class="hero-sub">Explore product performance, customer ratings, category patterns, and machine-learning insights.</div>
</div>
""", unsafe_allow_html=True)

REQUIRED_HINT = "Expected columns include product_name, category, discounted_price, actual_price, discount_percentage, rating, and rating_count."

@st.cache_data
def make_demo_data():
    rng = np.random.default_rng(42)
    categories = ["Electronics", "Home & Kitchen", "Computers & Accessories", "Toys & Games", "Office Products"]
    products = []
    for i in range(250):
        cat = rng.choice(categories)
        price = float(np.round(rng.uniform(150, 18000), 2))
        discount = int(rng.integers(0, 61))
        discounted = round(price * (1-discount/100), 2)
        rating = float(np.clip(np.round(rng.normal(3.9 + (discount % 10)/40, 0.55), 1), 1, 5))
        count = int(rng.integers(1, 12000))
        products.append({
            "product_id": f"DEMO{i+1:04d}",
            "product_name": f"Demo {cat} Product {i+1}",
            "category": cat,
            "discounted_price": discounted,
            "actual_price": price,
            "discount_percentage": discount,
            "rating": rating,
            "rating_count": count
        })
    return pd.DataFrame(products)

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    aliases = {
        "discounted_price": ["discounted_price", "discountedprice", "discount_price"],
        "actual_price": ["actual_price", "actualprice", "original_price"],
        "discount_percentage": ["discount_percentage", "discountpercentage", "discount_percent"],
        "rating_count": ["rating_count", "ratingcount", "reviews", "review_count"],
        "product_name": ["product_name", "productname", "name", "title"],
        "category": ["category", "main_category", "product_category"],
        "rating": ["rating", "product_rating"]
    }
    for target, names in aliases.items():
        if target not in df.columns:
            for name in names:
                if name in df.columns:
                    df = df.rename(columns={name: target})
                    break
    return df

def clean_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

DATA_FILE = "amazon.csv"
try:
    raw = pd.read_csv(DATA_FILE)
    df = normalize_columns(raw)
    data_source = "Real Amazon dataset (amazon.csv)"
except FileNotFoundError:
    st.error("The real dataset file 'amazon.csv' was not found. Place your original Amazon Sales Dataset CSV in the same folder as app.py, then restart Streamlit. The dashboard does not use demo data.")
    st.stop()
except Exception as e:
    st.error(f"Could not read amazon.csv: {e}")
    st.stop()

for col in ["discounted_price", "actual_price", "discount_percentage", "rating", "rating_count"]:
    if col in df.columns:
        df[col] = clean_numeric(df[col])
if "category" not in df.columns:
    df["category"] = "Unknown"
if "product_name" not in df.columns:
    df["product_name"] = [f"Product {i+1}" for i in range(len(df))]
if "rating" not in df.columns:
    st.error("The dataset must contain a rating column. " + REQUIRED_HINT)
    st.stop()
for col in ["discounted_price", "actual_price", "discount_percentage", "rating_count"]:
    if col not in df.columns:
        df[col] = np.nan
df["rating"] = df["rating"].fillna(df["rating"].median())
df["category"] = df["category"].fillna("Unknown").astype(str)
df["product_name"] = df["product_name"].fillna("Unnamed product").astype(str)
df = df.dropna(subset=["rating"])
if len(df) < 10:
    st.warning("The dataset has fewer than 10 usable rows. Results may be unreliable.")

st.sidebar.markdown("""
<div style="padding: 8px 0 18px 0;">
  <div style="font-size:1.6rem;line-height:1.2;">🛍️ <b>AMAZON</b></div>
  <div style="font-size:.78rem;letter-spacing:.12em;color:#a9c2ed!important;text-transform:uppercase;margin-top:5px;">Analytics workspace</div>
</div>
<div style="height:1px;background:rgba(255,255,255,.16);margin-bottom:16px;"></div>
""", unsafe_allow_html=True)
st.sidebar.success(f"Data source: {data_source}\n\nRows loaded: {len(df):,}")
page = st.sidebar.radio("Choose a section", ["Overview", "Data Warehouse Schemas", "OLAP Operations", "K-Means Clustering", "Classification", "Association Rules (Apriori)", "Rating Prediction", "Data Preview"])

if page == "Overview":
    st.subheader("Dataset overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Products / rows", f"{len(df):,}")
    c2.metric("Average rating", f"{df['rating'].mean():.2f} / 5")
    c3.metric("Categories", f"{df['category'].nunique():,}")
    c4.metric("Average discount", f"{df['discount_percentage'].mean():.1f}%" if df["discount_percentage"].notna().any() else "N/A")
    left,right = st.columns(2)
    with left:
        st.write("**Products by category**")
        counts = df["category"].value_counts().head(12)
        st.bar_chart(counts)
    with right:
        st.write("**Rating distribution**")
        st.bar_chart(df["rating"].round().value_counts().sort_index())
    st.write("**Dataset summary**")
    st.dataframe(df[["product_name","category","discounted_price","actual_price","discount_percentage","rating","rating_count"]].describe(include="all").T, use_container_width=True)

elif page == "Data Warehouse Schemas":
    st.subheader("Data warehouse design from Review 1")
    st.write("The schema is built from the real Amazon CSV columns. The dataset is product-review data, so it does not contain a full sales transaction table.")
    st.markdown("### Star Schema")
    st.code("""FACT_PRODUCT_RATING\n- rating_id (primary key)\n- user_key (foreign key)\n- product_key (foreign key)\n- category_key (foreign key)\n- rating, discounted_price, actual_price, rating_count\n\nDIM_USER(user_key, user_id, user_name)\nDIM_PRODUCT(product_key, product_id, product_name, product_link)\nDIM_CATEGORY(category_key, main_category, sub_category)""", language="text")
    st.markdown("### Star Schema Diagram")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    def draw_table(x, y, w, title, fields, head_color, body_color):
        row_h, head_h = 0.36, 0.5
        h = head_h + row_h * len(fields)
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=body_color, edgecolor="#17233c", linewidth=1.5, zorder=2))
        ax.add_patch(plt.Rectangle((x, y + h - head_h), w, head_h, facecolor=head_color, edgecolor="#17233c", linewidth=1.5, zorder=3))
        ax.text(x + w/2, y + h - head_h/2, title, ha="center", va="center", color="white", fontsize=11, fontweight="bold", zorder=4)
        for i, f in enumerate(fields):
            ax.text(x + 0.15, y + h - head_h - row_h*(i + 0.5), f, ha="left", va="center", fontsize=9.5, color="#17233c", zorder=4)
        return h
    fact_fields = ["rating_id (PK)", "user_key (FK)", "product_key (FK)", "category_key (FK)", "rating", "discounted_price", "actual_price", "rating_count"]
    fx, fy, fw = 4.5, 2.4, 3.0
    fh = draw_table(fx, fy, fw, "FACT_PRODUCT_RATING", fact_fields, "#356df3", "#eef4ff")
    mid_y = fy + fh/2
    draw_table(0.3, mid_y - 0.77, 3.2, "DIM_USER", ["user_key (PK)", "user_id", "user_name"], "#2a9d8f", "#e8f6f4")
    draw_table(8.5, mid_y - 0.95, 3.2, "DIM_PRODUCT", ["product_key (PK)", "product_id", "product_name", "product_link"], "#e07a1f", "#fff3e6")
    draw_table(fx, 0.2, fw, "DIM_CATEGORY", ["category_key (PK)", "main_category", "sub_category"], "#8e5bd6", "#f3ecfc")
    ax.plot([3.5, fx], [mid_y, mid_y], color="#17233c", linewidth=1.8, zorder=1)
    ax.plot([fx + fw, 8.5], [mid_y, mid_y], color="#17233c", linewidth=1.8, zorder=1)
    ax.plot([fx + fw/2, fx + fw/2], [fy, 1.78], color="#17233c", linewidth=1.8, zorder=1)
    ax.text(3.6, mid_y + 0.15, "1", fontsize=10, fontweight="bold"); ax.text(fx - 0.3, mid_y + 0.15, "*", fontsize=12, fontweight="bold")
    ax.text(fx + fw + 0.1, mid_y + 0.15, "*", fontsize=12, fontweight="bold"); ax.text(8.2, mid_y + 0.15, "1", fontsize=10, fontweight="bold")
    ax.text(fx + fw/2 + 0.12, fy - 0.3, "*", fontsize=12, fontweight="bold"); ax.text(fx + fw/2 + 0.12, 1.85, "1", fontsize=10, fontweight="bold")
    ax.set_title("Star Schema: Product Rating Fact with User, Product and Category Dimensions", fontsize=12, fontweight="bold", color="#17233c")
    st.pyplot(fig)
    plt.close(fig)
    work = df.copy()
    work["main_category"] = work["category"].astype(str).str.split("|").str[0]
    work["sub_category"] = work["category"].astype(str).str.split("|").str[1:].apply(lambda x: " | ".join(x) if x else "Unknown")
    st.markdown("### Star Schema dimension previews")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.write("**DIM_USER**")
        user_cols = [c for c in ["user_id", "user_name"] if c in work.columns]
        if user_cols:
            st.dataframe(work[user_cols].drop_duplicates().head(15), use_container_width=True)
        else:
            st.info("User fields are not available in this CSV.")
    with c2:
        st.write("**DIM_PRODUCT**")
        product_cols = [c for c in ["product_id", "product_name", "product_link"] if c in work.columns]
        st.dataframe(work[product_cols].drop_duplicates().head(15), use_container_width=True)
    with c3:
        st.write("**DIM_CATEGORY**")
        st.dataframe(work[["main_category", "sub_category"]].drop_duplicates().head(15), use_container_width=True)
    st.caption("This page presents the Star Schema design and dimension previews from the CSV. It does not claim to create or load a SQLite database.")

elif page == "OLAP Operations":
    st.subheader("OLAP operations")
    st.write("Explore product ratings and prices at different category levels using Roll-up, Drill-down, Slice, Dice, and Pivot.")
    olap = df.copy()
    olap["main_category"] = olap["category"].astype(str).str.split("|").str[0]
    olap["sub_category"] = olap["category"].astype(str).str.split("|").str[1:].apply(lambda x: " | ".join(x) if x else "Unknown")
    operation = st.selectbox("Choose OLAP operation", ["Roll-up", "Drill-down", "Slice", "Dice", "Pivot"])
    if operation == "Roll-up":
        st.write("**Roll-up:** summarize detailed products into main categories.")
        out = olap.groupby("main_category", dropna=False).agg(product_rows=("product_name","count"), average_rating=("rating","mean"), average_discount=("discount_percentage","mean"), average_price=("discounted_price","mean")).reset_index()
        st.dataframe(out.round(2), use_container_width=True)
    elif operation == "Drill-down":
        st.write("**Drill-down:** move from main category into sub-category details.")
        mains = sorted(olap["main_category"].dropna().unique().tolist())
        chosen = st.selectbox("Select main category", mains)
        out = olap[olap["main_category"] == chosen].groupby("sub_category", dropna=False).agg(product_rows=("product_name","count"), average_rating=("rating","mean"), average_price=("discounted_price","mean")).reset_index()
        st.dataframe(out.round(2), use_container_width=True)
    elif operation == "Slice":
        st.write("**Slice:** analyze one selected main category.")
        chosen = st.selectbox("Select category", sorted(olap["main_category"].dropna().unique().tolist()))
        st.dataframe(olap[olap["main_category"] == chosen][["product_name","sub_category","rating","discounted_price","discount_percentage"]], use_container_width=True)
    elif operation == "Dice":
        st.write("**Dice:** filter by category and minimum rating together.")
        chosen = st.multiselect("Select main categories", sorted(olap["main_category"].dropna().unique().tolist()), default=sorted(olap["main_category"].dropna().unique().tolist())[:1])
        minimum = st.slider("Minimum rating", 1.0, 5.0, 4.0, 0.5)
        out = olap[olap["main_category"].isin(chosen) & (olap["rating"] >= minimum)]
        st.write(f"Rows matching filters: {len(out):,}")
        st.dataframe(out[["product_name","main_category","sub_category","rating","discounted_price"]], use_container_width=True)
    else:
        st.write("**Pivot:** compare average rating across main categories and sub-categories.")
        pivot = pd.pivot_table(olap, index="main_category", columns="sub_category", values="rating", aggfunc="mean")
        st.dataframe(pivot.round(2), use_container_width=True)

elif page == "K-Means Clustering":
    st.subheader("K-Means product clustering")
    st.write("Choose one or more K values. K is the number of groups the algorithm creates.")
    numeric_candidates = [c for c in ["discounted_price","actual_price","discount_percentage","rating","rating_count"] if c in df.columns and df[c].notna().any()]
    default_features = [c for c in ["discounted_price","discount_percentage","rating","rating_count"] if c in numeric_candidates]
    features = st.multiselect("Select numeric features", numeric_candidates, default=default_features or numeric_candidates)
    ks = st.multiselect("Select K values to compare", [2,3,4,5,6,7,8], default=[3])
    if st.button("Run K-Means", type="primary"):
        if len(features) < 2:
            st.error("Choose at least two numeric features.")
        elif not ks:
            st.error("Choose at least one K value.")
        else:
            X = df[features].copy()
            for col in features:
                X[col] = X[col].fillna(X[col].median())
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            all_results=[]
            for k in ks:
                if k >= len(df):
                    st.warning(f"Skipped K={k}: K must be smaller than the number of rows.")
                    continue
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(Xs)
                sil = silhouette_score(Xs, labels) if len(set(labels)) > 1 and len(df) > k else np.nan
                all_results.append({"K":k, "Silhouette score": round(float(sil),4) if not np.isnan(sil) else np.nan, "Inertia": round(float(model.inertia_),2)})
                result = df[["product_name","category","rating"]].copy()
                result["cluster"] = labels
                with st.expander(f"Results for K = {k}", expanded=(k==ks[0])):
                    st.write(f"Silhouette score: **{sil:.3f}** (closer to 1 generally indicates better-separated clusters)")
                    st.dataframe(result.head(30), use_container_width=True)
                    if len(features) >= 2:
                        fig, ax = plt.subplots()
                        ax.scatter(Xs[:,0], Xs[:,1], c=labels)
                        ax.set_xlabel(features[0] + " (standardized)")
                        ax.set_ylabel(features[1] + " (standardized)")
                        ax.set_title(f"K-Means clusters (K={k}; first two selected features)")
                        st.pyplot(fig)
                        plt.close(fig)
            if all_results:
                st.write("**Compare selected K values**")
                st.dataframe(pd.DataFrame(all_results), use_container_width=True)
                st.caption("Use the Elbow/inertia trend and silhouette score together; do not select K from one metric alone.")

elif page == "Classification":
    st.subheader("Product rating classification")
    st.write("The numeric rating is grouped into Low (<3), Medium (3 to <4), and High (4 to 5) classes.")
    models_chosen = st.multiselect("Choose model(s)", ["Decision Tree", "Gaussian Naive Bayes"], default=["Decision Tree","Gaussian Naive Bayes"])
    test_size = st.slider("Test data percentage", 10, 40, 20, 5)
    features = [c for c in ["discounted_price","actual_price","discount_percentage","rating_count"] if df[c].notna().any()]
    features += ["category"]
    if st.button("Run classification", type="primary"):
        if not models_chosen:
            st.error("Select at least one classification model.")
        else:
            y = pd.cut(df["rating"], bins=[-np.inf,3,4,np.inf], labels=["Low","Medium","High"], right=False)
            valid = y.notna()
            X = df.loc[valid, features].copy()
            y = y[valid].astype(str)
            if y.nunique() < 2:
                st.error("Classification requires at least two rating classes in the dataset.")
            else:
                numeric = [c for c in features if c != "category"]
                categorical = ["category"]
                pre = ColumnTransformer([
                    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),("scale",StandardScaler())]), numeric),
                    ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]), categorical)
                ])
                strat = y if y.value_counts().min() >= 2 else None
                X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size/100,random_state=42,stratify=strat)
                for name in models_chosen:
                    estimator = DecisionTreeClassifier(max_depth=5, random_state=42) if name=="Decision Tree" else GaussianNB()
                    # GaussianNB needs dense encoded features.
                    model = Pipeline([("prep",pre),("dense", FunctionTransformerToDense()),("model",estimator)]) if name=="Gaussian Naive Bayes" else Pipeline([("prep",pre),("model",estimator)])
                    try:
                        model.fit(X_train,y_train)
                        pred=model.predict(X_test)
                        st.markdown(f"### {name}")
                        st.metric("Accuracy", f"{accuracy_score(y_test,pred):.3f}")
                        labels=["Low","Medium","High"]
                        cm=confusion_matrix(y_test,pred,labels=labels)
                        st.write("**Confusion matrix** (rows = actual class, columns = predicted class)")
                        st.dataframe(pd.DataFrame(cm,index=labels,columns=labels),use_container_width=True)
                        st.text(classification_report(y_test,pred,labels=labels,zero_division=0))
                        if name=="Decision Tree":
                            with st.expander("View decision tree"):
                                prep = model.named_steps["prep"]
                                feature_names = prep.get_feature_names_out()
                                fig,ax=plt.subplots(figsize=(16,7))
                                plot_tree(model.named_steps["model"],feature_names=feature_names, class_names=model.named_steps["model"].classes_,filled=True,max_depth=3,fontsize=7,ax=ax)
                                st.pyplot(fig)
                                plt.close(fig)
                    except Exception as e:
                        st.error(f"{name} could not run on this dataset: {e}")
    st.caption("For a fair comparison, both models use the same train/test split. Rating is used to create the target classes, so this is an educational classification demonstration rather than a validated recommendation engine.")

elif page == "Association Rules (Apriori)":
    st.subheader("Association rule mining")
    st.write("Apriori needs transaction-style data showing which items occurred together. A typical product-review CSV may not contain real shopping baskets, so this section builds category co-occurrence baskets as a proxy.")
    min_support = st.slider("Minimum support", 0.01, 0.50, 0.05, 0.01)
    min_conf = st.slider("Minimum confidence", 0.10, 1.00, 0.30, 0.05)
    if st.button("Generate Apriori rules", type="primary"):
        try:
            from mlxtend.frequent_patterns import apriori, association_rules
            baskets = df.groupby("product_name")["category"].apply(lambda s: list(set(s.astype(str))))
            # If each product has only one category, rules may be empty; make a category/price-level proxy.
            trans = []
            for _,row in df.iterrows():
                items=[f"Category={row['category']}"]
                if pd.notna(row["discount_percentage"]):
                    items.append("High discount" if row["discount_percentage"]>=30 else "Low discount")
                if pd.notna(row["rating"]):
                    items.append("High rating" if row["rating"]>=4 else "Lower rating")
                trans.append(list(set(items)))
            from mlxtend.preprocessing import TransactionEncoder
            te=TransactionEncoder()
            arr=te.fit(trans).transform(trans)
            basket_df=pd.DataFrame(arr,columns=te.columns_)
            frequent=apriori(basket_df,min_support=min_support,use_colnames=True)
            if frequent.empty:
                st.warning("No frequent itemsets found. Try lowering minimum support.")
            else:
                rules=association_rules(frequent,metric="confidence",min_threshold=min_conf)
                if rules.empty:
                    st.warning("No rules met the confidence threshold. Try lowering minimum confidence.")
                else:
                    rules["antecedents"]=rules["antecedents"].apply(lambda x:", ".join(sorted(x)))
                    rules["consequents"]=rules["consequents"].apply(lambda x:", ".join(sorted(x)))
                    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]].sort_values("lift",ascending=False).head(30),use_container_width=True)
                    st.caption("Important: these are co-occurrence patterns in product attributes, not proof that customers bought items together. Real basket data is needed for genuine cross-selling rules.")
        except ImportError:
            st.error("Install mlxtend by running: pip install mlxtend")

elif page == "Rating Prediction":
    st.subheader("Random Forest rating prediction")
    st.write("Predict ratings from category, price, discount, and rating-count features.")
    features = [c for c in ["discounted_price","actual_price","discount_percentage","rating_count"] if df[c].notna().any()] + ["category"]
    test_size=st.slider("Test data percentage ",10,40,20,5)
    if st.button("Train Random Forest regressor",type="primary"):
        X=df[features].copy()
        y=df["rating"].copy()
        numeric=[c for c in features if c!="category"]
        pre=ColumnTransformer([
            ("num",Pipeline([("imputer",SimpleImputer(strategy="median"))]),numeric),
            ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),["category"])
        ])
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size/100,random_state=42)
        model=Pipeline([("prep",pre),("model",RandomForestRegressor(n_estimators=150,random_state=42))])
        model.fit(X_train,y_train)
        pred=model.predict(X_test)
        mae=mean_absolute_error(y_test,pred)
        rmse=float(np.sqrt(mean_squared_error(y_test,pred)))
        r2=r2_score(y_test,pred)
        a,b,c=st.columns(3)
        a.metric("MAE",f"{mae:.3f}")
        b.metric("RMSE",f"{rmse:.3f}")
        c.metric("R² score",f"{r2:.3f}")
        results=pd.DataFrame({"Actual rating":y_test.values,"Predicted rating":np.round(pred,2)})
        st.dataframe(results.head(30),use_container_width=True)
        fig,ax=plt.subplots()
        ax.scatter(y_test,pred)
        ax.set_xlabel("Actual rating")
        ax.set_ylabel("Predicted rating")
        ax.set_title("Actual vs predicted ratings")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("These metrics measure prediction performance on the held-out test split. Results depend on dataset quality and features.")

else:
    st.subheader("Data preview")
    st.write(f"Source: **{data_source}**")
    st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    st.dataframe(df, use_container_width=True, height=500)
    st.download_button("Download currently loaded data as CSV", df.to_csv(index=False).encode("utf-8"), file_name="loaded_products.csv", mime="text/csv")

