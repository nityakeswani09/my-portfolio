
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Bloom AI | Iris Intelligence",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM CSS: COLORFUL GRADIENT DESIGN
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

:root {
    --pink: #f472b6;
    --purple: #8b5cf6;
    --blue: #38bdf8;
    --orange: #fb923c;
    --ink: #30264b;
}

.stApp {
    background:
        radial-gradient(ellipse at 5% 0%, #fce7f3 0%, transparent 34%),
        radial-gradient(ellipse at 95% 10%, #ede9fe 0%, transparent 32%),
        linear-gradient(145deg, #fffafd 0%, #f8f7ff 48%, #f0f9ff 100%);
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Outfit', sans-serif !important;
    color: var(--ink);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f3e8ff, #fce7f3 60%, #e0f2fe);
    border-right: 1px solid #ffffff;
}

.hero {
    padding: 2.5rem;
    border-radius: 28px;
    color: white;
    background: linear-gradient(
        120deg, #7c3aed 0%, #c026d3 43%,
        #ec4899 72%, #fb923c 100%
    );
    box-shadow: 0 18px 45px rgba(139, 92, 246, .20);
    margin-bottom: 1.5rem;
}

.hero h1 {
    color: white !important;
    font-size: clamp(2.1rem, 5vw, 3.8rem);
    line-height: 1.05;
    margin: .5rem 0 1rem;
}

.hero p {
    color: #fff4ff;
    font-size: 1.05rem;
    max-width: 720px;
}

.eyebrow {
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: .76rem;
    font-weight: 800;
    color: #fce7f3;
}

.section-tag {
    color: #9333ea;
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.metric-card {
    background: rgba(255,255,255,.86);
    border: 1px solid #e9ddff;
    border-radius: 20px;
    padding: 1rem 1.1rem;
    box-shadow: 0 8px 24px rgba(90, 65, 140, .06);
}

.result-card {
    padding: 1.6rem;
    border-radius: 22px;
    background: linear-gradient(120deg, #ede9fe, #fce7f3, #ffedd5);
    border: 1px solid #e9d5ff;
    margin: .8rem 0 1rem;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(110deg, #7c3aed, #db2777, #f97316);
    color: white;
    border: 0;
    border-radius: 13px;
    font-weight: 700;
    min-height: 2.8rem;
    transition: transform .15s ease;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    color: white;
    border: 0;
    transform: translateY(-2px);
    box-shadow: 0 8px 22px #c084fc66;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,.84);
    border: 1px solid #e9ddff;
    padding: 1rem;
    border-radius: 18px;
}

[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,.65);
    border-radius: 18px;
}

hr {
    border-color: #eadfff;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD DATA AND TRAIN MODEL
# --------------------------------------------------
FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

@st.cache_data
def load_data():
    return pd.read_csv("iris.csv")

@st.cache_resource
def build_model(data):
    X = data[FEATURES]
    y = data["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return model, X_test, y_test, predictions


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Could not find iris.csv. Place it in the same folder as app.py."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not read iris.csv: {exc}")
    st.stop()

required_columns = FEATURES + ["species"]

if not all(col in df.columns for col in required_columns):
    st.error(
        "CSV column names do not match the expected Iris format. "
        "Check the four measurement columns and the species column."
    )
    st.stop()

if df.empty:
    st.error("Your dataset is empty.")
    st.stop()

df = df[required_columns].copy()
df["species"] = df["species"].astype(str).str.strip()
df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=required_columns, inplace=True)

if df.empty or df["species"].nunique() < 2:
    st.error("The dataset needs valid measurements and at least two species.")
    st.stop()

if (df[FEATURES] < 0).any().any():
    st.error("Flower measurements must not be negative.")
    st.stop()

if df["species"].value_counts().min() < 2:
    st.error(
        "Each species needs at least two valid records for a stratified split."
    )
    st.stop()

model, X_test, y_test, predictions = build_model(df)
accuracy = accuracy_score(y_test, predictions)
species_names = sorted(df["species"].unique().tolist())

# Consistent vibrant palette for charts
palette = px.colors.qualitative.Bold
color_map = {
    name: palette[i % len(palette)]
    for i, name in enumerate(species_names)
}


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        "<h2 style='margin-bottom:0'>🌸 Bloom AI</h2>",
        unsafe_allow_html=True,
    )
    st.caption("IRIS INTELLIGENCE LAB")
    st.divider()

    page = st.radio(
        "EXPLORE",
        [
            "🏠 Home",
            "🔮 Prediction Lab",
            "📊 Data Explorer",
            "🧠 Model Insights",
            "ℹ️ About",
        ],
        label_visibility="visible",
    )

    st.divider()
    st.markdown("**Your ML workspace**")
    st.caption(f"🌱 {len(df)} valid records")
    st.caption(f"🌈 {len(species_names)} species")
    st.caption("🌳 Decision Tree Classifier")

    st.markdown("---")
    st.caption("Made with Python + Streamlit")


# --------------------------------------------------
# REUSABLE HERO
# --------------------------------------------------
def hero(title, subtitle, eyebrow="BLOOM AI / IRIS INTELLIGENCE"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "🏠 Home":
    hero(
        "Every flower has a story. 🌷",
        "Discover the Iris flower family with machine learning. "
        "Explore real measurements, predict species, and understand "
        "how a Decision Tree makes its decisions.",
    )

    st.markdown("### Your ML garden at a glance")

    a, b, c, d = st.columns(4)
    a.metric("Flower Records", f"{len(df):,}")
    b.metric("Input Features", len(FEATURES))
    c.metric("Iris Species", len(species_names))
    d.metric("Test Accuracy", f"{accuracy:.1%}")

    st.write("")
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### 🌈 Meet the Iris family")
        counts = (
            df["species"]
            .value_counts()
            .rename_axis("Species")
            .reset_index(name="Records")
        )
        fig = px.pie(
            counts,
            names="Species",
            values="Records",
            color="Species",
            color_discrete_map=color_map,
            hole=.58,
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color="white", width=3)),
        )
        fig.update_layout(
            margin=dict(t=20, b=20, l=10, r=10),
            legend_title="Species",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### ✨ What can you do here?")
        st.markdown("""
        <div class="metric-card">
            <b>🔮 Predict a species</b><br>
            <span>Enter four measurements and see the model's prediction.</span>
            <hr>
            <b>📊 Explore the dataset</b><br>
            <span>Compare species and discover patterns in flower dimensions.</span>
            <hr>
            <b>🧠 Understand the model</b><br>
            <span>Inspect accuracy, confusion matrix and classification metrics.</span>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        if st.button("Open Prediction Lab →", use_container_width=True):
            st.session_state["navigation_request"] = "🔮 Prediction Lab"
            st.rerun()

    st.markdown("### 🌿 Flower measurement preview")
    st.dataframe(
        df.head(8),
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# PREDICTION LAB
# --------------------------------------------------
elif page == "🔮 Prediction Lab":
    hero(
        "Meet your next flower. 🔮",
        "Adjust the measurements below. Bloom AI will use your trained "
        "Decision Tree to predict which Iris species best matches them.",
        "BLOOM AI / PREDICTION STUDIO",
    )

    st.markdown("### Enter flower measurements")

    with st.form("flower_form"):
        col1, col2 = st.columns(2)

        with col1:
            sepal_length = st.slider(
                "🌿 Sepal length (cm)",
                0.1, 10.0, 5.1, 0.1,
            )
            sepal_width = st.slider(
                "🍃 Sepal width (cm)",
                0.1, 10.0, 3.5, 0.1,
            )

        with col2:
            petal_length = st.slider(
                "🌸 Petal length (cm)",
                0.1, 10.0, 1.4, 0.1,
            )
            petal_width = st.slider(
                "🌺 Petal width (cm)",
                0.1, 10.0, 0.2, 0.1,
            )

        predict_clicked = st.form_submit_button(
            "✨ Identify This Flower",
            use_container_width=True,
        )

    if predict_clicked:
        sample = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=FEATURES,
        )

        predicted_species = model.predict(sample)[0]
        probabilities = model.predict_proba(sample)[0]
        confidence = float(np.max(probabilities))

        st.session_state["last_prediction"] = {
            "species": predicted_species,
            "confidence": confidence,
            "measurements": sample.iloc[0].to_dict(),
        }

    if "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]
        predicted_species = result["species"]

        st.markdown(
            f"""
            <div class="result-card">
                <div class="section-tag">YOUR FLOWER IDENTIFICATION</div>
                <h2>🌸 {predicted_species}</h2>
                <p>
                    The Decision Tree selected <b>{predicted_species}</b>
                    as the predicted Iris species.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.metric(
            "Highest model probability",
            f"{result['confidence']:.1%}",
        )
        st.caption(
            "This is the model's probability estimate, not a guarantee "
            "that the prediction is correct."
        )

        saved_sample = pd.DataFrame(
            [result["measurements"]],
            columns=FEATURES,
        )
        saved_probabilities = model.predict_proba(saved_sample)[0]

        prob_df = pd.DataFrame({
            "Species": model.classes_,
            "Probability": saved_probabilities,
        }).sort_values("Probability", ascending=True)

        fig = px.bar(
            prob_df,
            x="Probability",
            y="Species",
            orientation="h",
            color="Species",
            color_discrete_map=color_map,
            text=prob_df["Probability"].map(lambda x: f"{x:.1%}"),
            title="Model probabilities by species",
        )
        fig.update_layout(
            xaxis=dict(tickformat=".0%", range=[0, 1]),
            showlegend=False,
            margin=dict(t=55, b=20, l=10, r=20),
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Measurements used")
        st.dataframe(saved_sample, use_container_width=True, hide_index=True)


# --------------------------------------------------
# DATA EXPLORER
# --------------------------------------------------
elif page == "📊 Data Explorer":
    hero(
        "The science of petals. 📊",
        "Explore the measurements behind the predictions. Filter species, "
        "compare dimensions and uncover patterns in the Iris dataset.",
        "BLOOM AI / DATA DISCOVERY",
    )

    selected_species = st.multiselect(
        "Filter by species",
        options=species_names,
        default=species_names,
    )

    filtered = df[df["species"].isin(selected_species)]

    if filtered.empty:
        st.info("Select at least one species to display the charts.")
        st.stop()

    x_feature = st.selectbox(
        "Horizontal axis",
        FEATURES,
        index=2,
    )
    y_feature = st.selectbox(
        "Vertical axis",
        FEATURES,
        index=3,
    )

    fig = px.scatter(
        filtered,
        x=x_feature,
        y=y_feature,
        color="species",
        color_discrete_map=color_map,
        hover_data=FEATURES,
        title=f"{x_feature} vs {y_feature}",
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=11, opacity=.82))
    fig.update_layout(
        legend_title="Species",
        margin=dict(t=55, b=20, l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        feature_to_compare = st.selectbox(
            "Compare distributions",
            FEATURES,
            key="distribution_feature",
        )
        fig = px.histogram(
            filtered,
            x=feature_to_compare,
            color="species",
            color_discrete_map=color_map,
            barmode="overlay",
            opacity=.75,
            marginal="box",
            title="Measurement distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Average measurements by species")
        if filtered.empty:
            st.info("No data available.")
        else:
            means = (
                filtered.groupby("species")[FEATURES]
                .mean()
                .reset_index()
                .melt(
                    id_vars="species",
                    var_name="Feature",
                    value_name="Mean",
                )
            )
            fig = px.bar(
                means,
                x="Feature",
                y="Mean",
                color="species",
                barmode="group",
                color_discrete_map=color_map,
                title="Species measurement comparison",
            )
            fig.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Filtered records")
    st.caption(f"Showing {len(filtered)} of {len(df)} records.")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="iris_filtered.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# MODEL INSIGHTS
# --------------------------------------------------
elif page == "🧠 Model Insights":
    hero(
        "Inside the Decision Tree. 🧠",
        "Understand how well your classifier performs on unseen test "
        "records, and inspect the mistakes it makes.",
        "BLOOM AI / MODEL EVALUATION",
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy", f"{accuracy:.2%}")
    col2.metric("Test Records", len(y_test))
    col3.metric("Tree Depth", model.get_depth())

    st.caption(
        "The model uses an 80/20 train-test split with a fixed random seed."
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_,
    )

    left, right = st.columns([1.15, 1])

    with left:
        fig = px.imshow(
            cm,
            x=model.classes_,
            y=model.classes_,
            text_auto=True,
            color_continuous_scale=[
                "#fce7f3", "#c084fc", "#7c3aed"
            ],
            labels={
                "x": "Predicted species",
                "y": "Actual species",
                "color": "Count",
            },
            title="Confusion matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        report = classification_report(
            y_test,
            predictions,
            labels=model.classes_,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        st.markdown("#### Classification report")
        st.dataframe(
            report_df.round(3),
            use_container_width=True,
        )

    st.markdown("### 🌳 Feature importance")
    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#f9a8d4", "#a78bfa", "#38bdf8"],
        title="Which measurements matter most to this tree?",
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View model configuration"):
        st.json(model.get_params())

    with st.expander("View test predictions"):
        comparison = pd.DataFrame({
            "Actual species": y_test.to_numpy(),
            "Predicted species": predictions,
        })
        comparison["Correct"] = (
            comparison["Actual species"]
            == comparison["Predicted species"]
        )
        st.dataframe(
            comparison.reset_index(drop=True),
            use_container_width=True,
        )


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------
elif page == "ℹ️ About":
    hero(
        "Small dataset. Big discoveries. 🌈",
        "A machine learning project that turns Iris flower measurements "
        "into an interactive learning experience.",
        "BLOOM AI / ABOUT THE PROJECT",
    )

    st.markdown("### The machine learning workflow")

    steps = [
        ("01", "Load", "Read flower measurements from iris.csv."),
        ("02", "Prepare", "Separate four features from the species label."),
        ("03", "Split", "Reserve 20% of records for testing."),
        ("04", "Train", "Fit a DecisionTreeClassifier on training data."),
        ("05", "Predict", "Estimate the species of a new flower."),
        ("06", "Evaluate", "Review accuracy, confusion matrix and metrics."),
    ]

    for number, title, description in steps:
        st.markdown(
            f"""
            <div class="metric-card" style="margin-bottom:10px">
                <span style="color:#9333ea;font-weight:800">{number}</span>
                &nbsp; <b>{title}</b><br>
                <span>{description}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### The four features")
    feature_info = pd.DataFrame({
        "Feature": FEATURES,
        "Description": [
            "Length of the outer leaf-like part",
            "Width of the outer leaf-like part",
            "Length of the flower petal",
            "Width of the flower petal",
        ],
        "Unit": ["cm", "cm", "cm", "cm"],
    })
    st.dataframe(feature_info, use_container_width=True, hide_index=True)

    st.info(
        "This is an educational ML demonstration. Its accuracy depends "
        "on the dataset, the split, and the trained model."
    )

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#8b7aa8'>"
    "🌸 Bloom AI · Built with Python, Streamlit, Plotly & Scikit-learn"
    "</p>",
    unsafe_allow_html=True,
)
