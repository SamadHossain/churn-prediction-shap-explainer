"""
Customer Churn Prediction + "Why They're Leaving" Explainer
Streamlit dashboard.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Churn Explainer", layout="wide")

# ---------------------------------------------------------------------------
# Load artifacts (cached so they're only loaded once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    explainer = joblib.load("shap_explainer.pkl")
    return model, explainer

@st.cache_data
def load_data():
    df = pd.read_csv("scored_customers.csv")
    return df

model, explainer = load_artifacts()
df = load_data()

SEGMENT_LABELS = {
    0: "Segment 0",
    1: "Segment 1",
    2: "Segment 2",
    3: "Segment 3",
}

# ---------------------------------------------------------------------------
# Sidebar - filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

risk_threshold = st.sidebar.slider(
    "Churn risk threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="Customers above this predicted probability are flagged as at-risk."
)

segment_filter = st.sidebar.multiselect(
    "Customer segment", options=sorted(df["segment"].unique()),
    default=sorted(df["segment"].unique()),
    format_func=lambda x: SEGMENT_LABELS.get(x, f"Segment {x}")
)

filtered_df = df[
    (df["predicted_proba"] >= 0) & (df["segment"].isin(segment_filter))
].copy()

# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------
st.title("Customer Churn Prediction + \"Why They're Leaving\" Explainer")
st.caption("Predicts which customers are likely to churn, and explains why — powered by XGBoost + SHAP.")

at_risk = filtered_df[filtered_df["predicted_proba"] >= risk_threshold]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers shown", f"{len(filtered_df):,}")
col2.metric("Flagged at-risk", f"{len(at_risk):,}")
col3.metric("At-risk rate", f"{len(at_risk) / max(len(filtered_df), 1):.1%}")
col4.metric("Actual churn rate (test set)", f"{filtered_df['actual_churn'].mean():.1%}")

st.divider()

# ---------------------------------------------------------------------------
# At-risk customer table
# ---------------------------------------------------------------------------
st.subheader("At-Risk Customers")

display_cols = ["predicted_proba", "tenure", "MonthlyCharges", "TotalCharges", "segment", "actual_churn"]
table_df = at_risk[display_cols].sort_values("predicted_proba", ascending=False).reset_index()
table_df.rename(columns={"index": "customer_id", "predicted_proba": "churn_risk"}, inplace=True)
table_df["segment"] = table_df["segment"].map(SEGMENT_LABELS)

st.dataframe(
    table_df.style.format({"churn_risk": "{:.1%}", "MonthlyCharges": "${:.2f}", "TotalCharges": "${:.2f}"}),
    use_container_width=True,
    height=350,
)

st.divider()

# ---------------------------------------------------------------------------
# Per-customer explanation
# ---------------------------------------------------------------------------
st.subheader("Why Is This Customer at Risk?")

if len(at_risk) == 0:
    st.info("No customers meet the current risk threshold and segment filters.")
else:
    selected_id = st.selectbox(
        "Select a customer ID to explain",
        options=table_df["customer_id"].tolist(),
    )

    feature_cols = [c for c in df.columns if c not in
                    ["actual_churn", "predicted_proba", "segment"]]
    customer_features = df.loc[[selected_id], feature_cols]

    proba = df.loc[selected_id, "predicted_proba"]
    st.markdown(f"**Predicted churn probability: {proba:.1%}**")

    shap_values_single = explainer.shap_values(customer_features)

    left, right = st.columns([1, 1])

    with left:
        st.markdown("**Top factors driving this prediction**")
        contributions = pd.Series(shap_values_single[0], index=feature_cols)
        top_features = contributions.abs().sort_values(ascending=False).head(6).index

        summary_rows = []
        for feat in top_features:
            effect = contributions[feat]
            direction = "⬆ increases risk" if effect > 0 else "⬇ decreases risk"
            summary_rows.append({
                "Feature": feat,
                "Value": customer_features.iloc[0][feat],
                "Effect": direction,
                "Impact": f"{effect:+.3f}",
            })
        st.table(pd.DataFrame(summary_rows))

    with right:
        st.markdown("**SHAP waterfall**")
        fig, ax = plt.subplots(figsize=(6, 4))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values_single[0],
                base_values=explainer.expected_value,
                data=customer_features.iloc[0],
                feature_names=feature_cols,
            ),
            show=False,
        )
        st.pyplot(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Segment overview
# ---------------------------------------------------------------------------
st.subheader("Segment Overview")

segment_summary = df.groupby("segment").agg(
    customers=("actual_churn", "count"),
    avg_tenure=("tenure", "mean"),
    avg_monthly_charges=("MonthlyCharges", "mean"),
    churn_rate=("actual_churn", "mean"),
).round(2)
segment_summary.index = segment_summary.index.map(SEGMENT_LABELS)

st.bar_chart(segment_summary["churn_rate"])
st.dataframe(segment_summary, use_container_width=True)
