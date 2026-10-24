"""
Predicting Automobile Fuel Efficiency (MPG) Using Machine Learning
Run:  pip install -r requirements.txt
      streamlit run app.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(page_title="MPG Predictor", page_icon="⛽", layout="wide")

FEATURES = ["cylinders", "displacement", "horsepower", "weight",
            "acceleration", "model year", "origin"]
ORIGIN = {1: "USA", 2: "Europe", 3: "Japan"}


# ------------------------------------------------------------------ data
@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Horsepower contains '?' -> convert to numeric and fill with median."""
    df = df.copy()
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())
    df = df.drop_duplicates()
    return df


def get_models():
    return {
        "Linear Regression": Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
        "Ridge Regression": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                       max_depth=3, random_state=42),
    }


@st.cache_resource
def train_all(df: pd.DataFrame, test_size: float, seed: int):
    X, y = df[FEATURES], df["mpg"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    rows, fitted = [], {}
    for name, model in get_models().items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        cv = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
        rows.append({
            "Model": name,
            "R² (test)": r2_score(y_te, pred),
            "MAE": mean_absolute_error(y_te, pred),
            "RMSE": np.sqrt(mean_squared_error(y_te, pred)),
            "CV R² (5-fold)": cv,
        })
        fitted[name] = model
    res = pd.DataFrame(rows).sort_values("R² (test)", ascending=False).reset_index(drop=True)
    return res, fitted, (X_te, y_te)


# --------------------------------------------------------------- sidebar
st.sidebar.title("⛽ MPG Predictor")
page = st.sidebar.radio("Navigate", ["🏠 Overview", "📊 Data Exploration",
                                     "🤖 Model Training & Evaluation", "🔮 Predict MPG"])
upload = st.sidebar.file_uploader("Use your own CSV (optional)", type="csv")
src = upload if upload is not None else "auto-mpg.csv"

try:
    raw = load_data(src)
    df = clean_data(raw)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

missing_cols = [c for c in FEATURES + ["mpg"] if c not in df.columns]
if missing_cols:
    st.error(f"Dataset is missing columns: {missing_cols}")
    st.stop()

st.sidebar.markdown("---")
test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)
results, fitted, (X_test, y_test) = train_all(df, test_size, int(seed))

# -------------------------------------------------------------- overview
if page == "🏠 Overview":
    st.title("Predicting Automobile Fuel Efficiency (MPG) Using Machine Learning")
    st.write("Predict a car's **miles per gallon** from its engine and design attributes "
             "using regression models trained on the classic Auto-MPG dataset.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cars", len(df))
    c2.metric("Features", len(FEATURES))
    c3.metric("Avg MPG", f"{df['mpg'].mean():.1f}")
    best = results.iloc[0]
    c4.metric(f"Best model: {best['Model']}", f"R² {best['R² (test)']:.3f}")

    st.subheader("Workflow")
    st.markdown("""
1. **Load** the dataset (`auto-mpg.csv`)
2. **Clean** – `horsepower` has `?` values → converted to numbers, filled with the median
3. **Explore** – distributions, correlations, MPG by origin / year
4. **Train** – Linear, Ridge, Decision Tree, Random Forest, Gradient Boosting
5. **Evaluate** – R², MAE, RMSE and 5-fold cross-validation
6. **Predict** – enter car specs and get an MPG estimate
""")
    st.subheader("Data sample")
    st.dataframe(df.head(10), use_container_width=True)

# ------------------------------------------------------------ exploration
elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration")
    t1, t2, t3, t4 = st.tabs(["Dataset", "Distributions", "Correlations", "Relationships"])

    with t1:
        st.write(f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
        st.write("Missing `?` horsepower values in raw file:",
                 int((raw["horsepower"].astype(str).str.strip() == "?").sum()))
        st.dataframe(df, use_container_width=True, height=300)
        st.dataframe(df.describe().T, use_container_width=True)

    with t2:
        col = st.selectbox("Column", ["mpg"] + FEATURES)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.histplot(df[col], kde=True, ax=ax, color="#2a9d8f")
        st.pyplot(fig)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        tmp = df.assign(Origin=df["origin"].map(ORIGIN))
        sns.boxplot(data=tmp, x="Origin", y="mpg", ax=ax, palette="Set2")
        ax.set_title("MPG by origin")
        st.pyplot(fig)

    with t3:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[["mpg"] + FEATURES].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        st.caption("Weight, displacement, horsepower and cylinders correlate strongly and negatively with MPG.")

    with t4:
        x = st.selectbox("X axis", FEATURES, index=3)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=df.assign(Origin=df["origin"].map(ORIGIN)),
                        x=x, y="mpg", hue="Origin", ax=ax)
        st.pyplot(fig)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        df.groupby("model year")["mpg"].mean().plot(marker="o", ax=ax, color="#e76f51")
        ax.set_ylabel("Average MPG")
        ax.set_title("Average MPG by model year")
        st.pyplot(fig)

# --------------------------------------------------------------- training
elif page == "🤖 Model Training & Evaluation":
    st.title("🤖 Model Training & Evaluation")
    st.dataframe(results.style.format({c: "{:.3f}" for c in results.columns[1:]})
                 .highlight_max(subset=["R² (test)", "CV R² (5-fold)"], color="#b7e4c7")
                 .highlight_min(subset=["MAE", "RMSE"], color="#b7e4c7"),
                 use_container_width=True)

    name = st.selectbox("Inspect model", results["Model"], index=0)
    model = fitted[name]
    pred = model.predict(X_test)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test, pred, alpha=0.7, color="#264653")
        lim = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
        ax.plot(lim, lim, "r--")
        ax.set_xlabel("Actual MPG"); ax.set_ylabel("Predicted MPG")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(y_test - pred, kde=True, ax=ax, color="#e9c46a")
        ax.set_title("Residuals")
        st.pyplot(fig)

    est = model.named_steps["m"] if isinstance(model, Pipeline) else model
    if hasattr(est, "feature_importances_"):
        imp = pd.Series(est.feature_importances_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        imp.plot.barh(ax=ax, color="#2a9d8f")
        ax.set_title("Feature importance")
        st.pyplot(fig)
    elif hasattr(est, "coef_"):
        coef = pd.Series(est.coef_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        coef.plot.barh(ax=ax, color="#2a9d8f")
        ax.set_title("Standardised coefficients")
        st.pyplot(fig)

# ------------------------------------------------------------- prediction
else:
    st.title("🔮 Predict MPG")
    model_name = st.selectbox("Model", results["Model"], index=0)
    model = fitted[model_name]

    c1, c2, c3 = st.columns(3)
    with c1:
        cylinders = st.selectbox("Cylinders", [3, 4, 5, 6, 8], index=1)
        displacement = st.slider("Displacement (cu in)", 60.0, 460.0, 150.0)
        horsepower = st.slider("Horsepower", 40.0, 250.0, 95.0)
    with c2:
        weight = st.slider("Weight (lbs)", 1500, 5200, 2800)
        acceleration = st.slider("Acceleration (0-60 s)", 8.0, 25.0, 15.0)
    with c3:
        year = st.slider("Model year (19xx)", 70, 82, 76)
        origin = st.selectbox("Origin", list(ORIGIN), format_func=ORIGIN.get)

    row = pd.DataFrame([[cylinders, displacement, horsepower, weight,
                         acceleration, year, origin]], columns=FEATURES)
    if st.button("Predict", type="primary"):
        mpg = float(model.predict(row)[0])
        st.success(f"Estimated fuel efficiency: **{mpg:.1f} MPG**  (≈ {mpg * 0.425:.1f} km/L)")
        if mpg >= 30:
            st.info("🌱 Fuel-efficient")
        elif mpg >= 20:
            st.info("🙂 Average efficiency")
        else:
            st.warning("⛽ Gas guzzler")

        st.subheader("All models compared")
        allp = pd.DataFrame({"Model": list(fitted),
                             "Predicted MPG": [float(m.predict(row)[0]) for m in fitted.values()]})
        st.dataframe(allp.style.format({"Predicted MPG": "{:.1f}"}), use_container_width=True)
