"""Interactive Credit Risk Scoring Dashboard."""

import sys
from pathlib import Path

# Ensure app/ is on the path when Streamlit runs from project root
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scoring import (
    MODEL_METRICS,
    RAW_COLUMNS,
    REPORTS_DIR,
    build_feature_row,
    engineer_features,
    load_model,
    load_sample_data,
    predict_with_thresholds,
    predict_batch,
    risk_color,
    sensitivity_analysis,
)

st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .decision-pill {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-weight: 600;
        color: white;
        font-size: 1rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

RISK_ORDER = ["Very Low Risk", "Low Risk", "Medium Risk", "High Risk"]
DECISION_ORDER = ["Auto-Approve", "Manual Review", "Auto-Decline"]

SENSITIVITY_CONFIG = {
    "LIMIT_BAL": {"min": 5_000, "max": 500_000, "step": 5_000},
    "AGE": {"min": 18, "max": 80, "step": 1},
    "PAY_0": {"min": -2, "max": 9, "step": 1},
    "PAY_2": {"min": -2, "max": 9, "step": 1},
    "BILL_AMT1": {"min": 0, "max": 100_000, "step": 1_000},
    "PAY_AMT1": {"min": 0, "max": 50_000, "step": 500},
}


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_dataset():
    return load_sample_data()


def default_inputs() -> dict:
    return {
        "LIMIT_BAL": 140_000,
        "SEX": 1,
        "EDUCATION": 2,
        "MARRIAGE": 2,
        "AGE": 35,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 12_000,
        "BILL_AMT2": 11_500,
        "BILL_AMT3": 10_800,
        "BILL_AMT4": 9_200,
        "BILL_AMT5": 8_700,
        "BILL_AMT6": 7_900,
        "PAY_AMT1": 3_000,
        "PAY_AMT2": 2_800,
        "PAY_AMT3": 2_500,
        "PAY_AMT4": 2_200,
        "PAY_AMT5": 2_000,
        "PAY_AMT6": 1_800,
    }


def init_session_state():
    if "inputs" not in st.session_state:
        st.session_state.inputs = default_inputs()
    if "scenario_a" not in st.session_state:
        st.session_state.scenario_a = None
    if "scenario_b" not in st.session_state:
        st.session_state.scenario_b = None
    if "prev_score" not in st.session_state:
        st.session_state.prev_score = None


def render_plotly_gauge(score: int):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": st.session_state.prev_score or score, "relative": False},
        title={"text": "Credit Score"},
        gauge={
            "axis": {"range": [300, 900]},
            "bar": {"color": "#1e40af"},
            "steps": [
                {"range": [300, 549], "color": "#fecaca"},
                {"range": [550, 649], "color": "#fde68a"},
                {"range": [650, 749], "color": "#bbf7d0"},
                {"range": [750, 900], "color": "#86efac"},
            ],
            "threshold": {
                "line": {"color": "#111827", "width": 4},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_payment_timeline(inputs: dict):
    months = ["M1", "M2", "M3", "M4", "M5", "M6"]
    bills = [inputs[f"BILL_AMT{i}"] for i in range(1, 7)]
    payments = [inputs[f"PAY_AMT{i}"] for i in range(1, 7)]
    df = pd.DataFrame({"Month": months, "Bill": bills, "Payment": payments})
    fig = px.bar(
        df,
        x="Month",
        y=["Bill", "Payment"],
        barmode="group",
        title="Bill vs Payment (6-month history)",
        color_discrete_map={"Bill": "#ef4444", "Payment": "#22c55e"},
    )
    fig.update_layout(height=320, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)


def render_live_results(model, inputs: dict, approve_min: int, review_min: int):
    features = build_feature_row(inputs)
    result = predict_with_thresholds(model, features, approve_min, review_min)

    delta_score = None
    if st.session_state.prev_score is not None:
        delta_score = result["credit_score"] - st.session_state.prev_score
    st.session_state.prev_score = result["credit_score"]

    c1, c2 = st.columns(2)
    c1.metric("Default Probability", f"{result['default_probability']:.1%}")
    c2.metric("Credit Score", result["credit_score"], delta=delta_score)

    color = risk_color(result["risk_category"])
    st.markdown(
        f'<span class="decision-pill" style="background:{color};">'
        f'{result["decision"]} · {result["risk_category"]}</span>',
        unsafe_allow_html=True,
    )

    render_plotly_gauge(result["credit_score"])

    eng = features[["TOTAL_BILL", "TOTAL_PAYMENT", "UTILIZATION_RATIO", "AVG_DELAY"]].T
    eng.columns = ["Value"]
    st.dataframe(eng, use_container_width=True)

    return result, features


def render_input_form():
    inputs = st.session_state.inputs

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🎲 Random applicant", use_container_width=True):
            sample = get_dataset().sample(1).iloc[0]
            st.session_state.inputs = {col: int(sample[col]) for col in RAW_COLUMNS}
            st.rerun()
    with c2:
        if st.button("↺ Reset defaults", use_container_width=True):
            st.session_state.inputs = default_inputs()
            st.rerun()

    tab_demo, tab_pay, tab_bills = st.tabs(["Demographics", "Repayment Status", "Bills & Payments"])

    with tab_demo:
        inputs["LIMIT_BAL"] = st.slider(
            "Credit limit", 5_000, 500_000, int(inputs["LIMIT_BAL"]), 5_000,
        )
        inputs["AGE"] = st.slider("Age", 18, 80, int(inputs["AGE"]))
        col1, col2, col3 = st.columns(3)
        with col1:
            inputs["SEX"] = st.selectbox(
                "Sex", [1, 2], index=0 if int(inputs["SEX"]) == 1 else 1,
                format_func=lambda x: "Male" if x == 1 else "Female",
            )
        with col2:
            edu_opts = [1, 2, 3, 4]
            inputs["EDUCATION"] = st.selectbox(
                "Education", edu_opts,
                index=edu_opts.index(int(inputs["EDUCATION"])) if int(inputs["EDUCATION"]) in edu_opts else 1,
                format_func=lambda x: {1: "Grad school", 2: "University", 3: "High school", 4: "Other"}[x],
            )
        with col3:
            mar_opts = [1, 2, 3]
            inputs["MARRIAGE"] = st.selectbox(
                "Marriage", mar_opts,
                index=mar_opts.index(int(inputs["MARRIAGE"])) if int(inputs["MARRIAGE"]) in mar_opts else 1,
                format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}[x],
            )

    with tab_pay:
        st.caption("−2 = no consumption · −1 = paid in full · 0 = revolving · 1–9 = delay months")
        pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        cols = st.columns(3)
        for i, col in enumerate(pay_cols):
            with cols[i % 3]:
                inputs[col] = st.slider(col, -2, 9, int(inputs[col]), key=f"slider_{col}")

    with tab_bills:
        st.markdown("**Bill amounts**")
        bill_cols = st.columns(3)
        for i in range(1, 7):
            col_name = f"BILL_AMT{i}"
            with bill_cols[(i - 1) % 3]:
                inputs[col_name] = st.number_input(
                    col_name, 0, 200_000, int(inputs[col_name]), 500, key=f"bill_{col_name}",
                )
        st.markdown("**Payment amounts**")
        pay_cols_amt = st.columns(3)
        for i in range(1, 7):
            col_name = f"PAY_AMT{i}"
            with pay_cols_amt[(i - 1) % 3]:
                inputs[col_name] = st.number_input(
                    col_name, 0, 200_000, int(inputs[col_name]), 500, key=f"pay_{col_name}",
                )

    st.session_state.inputs = inputs
    return inputs


def page_live_scoring(model, approve_min: int, review_min: int):
    st.header("Live Credit Scoring")
    st.caption("Scores update instantly as you adjust inputs — no button needed.")

    form_col, result_col = st.columns([1.1, 1], gap="large")

    with form_col:
        inputs = render_input_form()
        render_payment_timeline(inputs)

    with result_col:
        st.subheader("Real-time Result")
        result, _ = render_live_results(model, inputs, approve_min, review_min)

        if st.button("💾 Save as Scenario A", use_container_width=True):
            st.session_state.scenario_a = {"inputs": dict(inputs), "result": result}
            st.toast("Scenario A saved!", icon="✅")

        if st.button("💾 Save as Scenario B", use_container_width=True):
            st.session_state.scenario_b = {"inputs": dict(inputs), "result": result}
            st.toast("Scenario B saved!", icon="✅")


def page_what_if(model, approve_min: int, review_min: int):
    st.header("What-If Analysis")
    st.caption("See how changing a single variable impacts the credit score.")

    inputs = st.session_state.inputs
    feature = st.selectbox(
        "Variable to explore",
        list(SENSITIVITY_CONFIG.keys()),
        format_func=lambda x: x.replace("_", " "),
    )
    cfg = SENSITIVITY_CONFIG[feature]
    values = list(range(cfg["min"], cfg["max"] + 1, cfg["step"]))
    if len(values) > 80:
        step = max(cfg["step"], (cfg["max"] - cfg["min"]) // 60)
        values = list(range(cfg["min"], cfg["max"] + 1, step))

    with st.spinner("Computing sensitivity curve..."):
        curve = sensitivity_analysis(model, inputs, feature, values, approve_min, review_min)

    current_val = inputs[feature]
    fig = px.line(
        curve,
        x=feature,
        y="credit_score",
        markers=True,
        title=f"Credit Score vs {feature.replace('_', ' ')}",
        hover_data=["default_probability", "decision"],
    )
    fig.add_vline(
        x=current_val,
        line_dash="dash",
        line_color="#1e40af",
        annotation_text="Current",
        annotation_position="top",
    )
    fig.add_hline(y=approve_min, line_dash="dot", line_color="#22c55e",
                  annotation_text="Auto-Approve threshold")
    fig.add_hline(y=review_min, line_dash="dot", line_color="#f59e0b",
                  annotation_text="Manual Review threshold")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        prob_fig = px.area(
            curve, x=feature, y="default_probability",
            title="Default Probability",
            color_discrete_sequence=["#ef4444"],
        )
        prob_fig.add_vline(x=current_val, line_dash="dash", line_color="#1e40af")
        prob_fig.update_layout(height=300)
        st.plotly_chart(prob_fig, use_container_width=True)
    with c2:
        st.subheader("Score at key values")
        highlights = curve.iloc[:: max(1, len(curve) // 8)]
        st.dataframe(
            highlights[[feature, "credit_score", "default_probability", "decision"]],
            use_container_width=True,
            hide_index=True,
        )


def page_compare_scenarios():
    st.header("Compare Scenarios")
    st.caption("Side-by-side comparison of saved what-if scenarios.")

    a, b = st.session_state.scenario_a, st.session_state.scenario_b
    if a is None or b is None:
        st.info("Save scenarios from the **Live Scoring** page using the Scenario A / B buttons.")
        if a:
            st.success("Scenario A is saved. Save Scenario B to compare.")
        elif b:
            st.success("Scenario B is saved. Save Scenario A to compare.")
        return

    col_a, col_b = st.columns(2)
    for col, label, scenario in [(col_a, "Scenario A", a), (col_b, "Scenario B", b)]:
        with col:
            st.subheader(label)
            r = scenario["result"]
            st.metric("Credit Score", r["credit_score"])
            st.metric("Default Probability", f"{r['default_probability']:.1%}")
            st.markdown(
                f'<span class="decision-pill" style="background:{risk_color(r["risk_category"])};">'
                f'{r["decision"]} · {r["risk_category"]}</span>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("Input Differences")
    diff_rows = []
    for field in RAW_COLUMNS:
        va, vb = a["inputs"][field], b["inputs"][field]
        if va != vb:
            diff_rows.append({"Field": field, "Scenario A": va, "Scenario B": vb, "Delta": vb - va})
    if diff_rows:
        st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)
    else:
        st.write("No input differences between scenarios.")

    score_delta = b["result"]["credit_score"] - a["result"]["credit_score"]
    prob_delta = b["result"]["default_probability"] - a["result"]["default_probability"]
    m1, m2 = st.columns(2)
    m1.metric("Score change (B − A)", score_delta)
    m2.metric("Probability change (B − A)", f"{prob_delta:+.1%}")


def page_batch_scoring(model, approve_min: int, review_min: int):
    st.header("Batch Scoring")
    st.caption("Upload data or explore a scored sample with interactive filters.")

    source = st.radio("Data source", ["Sample dataset", "Upload CSV"], horizontal=True)

    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is None:
            return
        df = pd.read_csv(uploaded)
        for drop_col in ["ID", "default", "default.payment.next.month"]:
            if drop_col in df.columns:
                df = df.drop(columns=[drop_col])
        raw = df[RAW_COLUMNS] if all(c in df.columns for c in RAW_COLUMNS) else df
    else:
        n = st.slider("Records to score", 100, 2000, 500, 100)
        raw = get_dataset().sample(n, random_state=42)[RAW_COLUMNS]

    results = predict_batch(model, raw)
    for i, row in results.iterrows():
        score = row["credit_score"]
        results.at[i, "decision"] = (
            "Auto-Approve" if score >= approve_min
            else "Manual Review" if score >= review_min
            else "Auto-Decline"
        )
    output = pd.concat([raw.reset_index(drop=True), results], axis=1)

    f1, f2, f3 = st.columns(3)
    risk_filter = f1.multiselect("Filter risk band", RISK_ORDER, default=RISK_ORDER)
    decision_filter = f2.multiselect("Filter decision", DECISION_ORDER, default=DECISION_ORDER)
    score_range = f3.slider("Score range", 300, 900, (300, 900))

    filtered = output[
        output["risk_category"].isin(risk_filter)
        & output["decision"].isin(decision_filter)
        & output["credit_score"].between(score_range[0], score_range[1])
    ]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filtered records", len(filtered))
    m2.metric("Avg score", int(filtered["credit_score"].mean()) if len(filtered) else "—")
    m3.metric("Decline rate", f"{(filtered['decision'] == 'Auto-Decline').mean():.1%}" if len(filtered) else "—")
    m4.metric("Avg default prob.", f"{filtered['default_probability'].mean():.1%}" if len(filtered) else "—")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            filtered, x="credit_score", nbins=30, color="risk_category",
            title="Score distribution (interactive)",
            category_orders={"risk_category": RISK_ORDER},
        )
        fig.update_layout(height=350, barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        pie = px.pie(
            filtered, names="decision", title="Decision breakdown",
            category_orders={"decision": DECISION_ORDER},
            color="decision",
            color_discrete_map={
                "Auto-Approve": "#22c55e",
                "Manual Review": "#f59e0b",
                "Auto-Decline": "#ef4444",
            },
        )
        pie.update_layout(height=350)
        st.plotly_chart(pie, use_container_width=True)

    st.dataframe(
        filtered.sort_values("credit_score", ascending=False),
        use_container_width=True,
        height=400,
    )
    st.download_button(
        "⬇️ Download filtered results",
        filtered.to_csv(index=False),
        "credit_scores.csv",
        "text/csv",
        use_container_width=True,
    )


def page_portfolio_analytics(model, approve_min: int, review_min: int):
    st.header("Portfolio Analytics")
    st.caption("Explore portfolio risk with interactive filters and drill-downs.")

    n = st.slider("Portfolio size", 200, 5000, 1500, 100)
    sample = get_dataset().sample(n, random_state=11)
    raw = sample[RAW_COLUMNS].reset_index(drop=True)
    eng = engineer_features(raw)
    results = predict_batch(model, raw)
    results["decision"] = results["credit_score"].apply(
        lambda s: "Auto-Approve" if s >= approve_min else "Manual Review" if s >= review_min else "Auto-Decline"
    )
    portfolio = pd.concat([eng, results], axis=1)
    if "default" in sample.columns:
        portfolio["actual_default"] = sample["default"].values

    score_band = st.select_slider(
        "Focus score band",
        options=["All", "Prime (≥750)", "Near-Prime (650–749)", "Subprime (550–649)", "Decline (<550)"],
    )
    band_map = {
        "Prime (≥750)": portfolio["credit_score"] >= 750,
        "Near-Prime (650–749)": portfolio["credit_score"].between(650, 749),
        "Subprime (550–649)": portfolio["credit_score"].between(550, 649),
        "Decline (<550)": portfolio["credit_score"] < 550,
    }
    view = portfolio if score_band == "All" else portfolio[band_map[score_band]]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", len(view))
    c2.metric("Avg score", int(view["credit_score"].mean()))
    c3.metric("Decline %", f"{(view['decision'] == 'Auto-Decline').mean():.1%}")
    if "actual_default" in view.columns:
        c4.metric("Actual default %", f"{view['actual_default'].mean():.1%}")

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            view["risk_category"].value_counts().reset_index(),
            x="count", y="risk_category", orientation="h",
            title="Risk band counts",
            labels={"risk_category": "Risk Band", "count": "Count"},
            category_orders={"risk_category": RISK_ORDER},
        )
        fig.update_layout(height=350, yaxis={"categoryorder": "array", "categoryarray": RISK_ORDER[::-1]})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        scatter = px.scatter(
            view.sample(min(800, len(view)), random_state=3),
            x="UTILIZATION_RATIO",
            y="credit_score",
            color="risk_category",
            size="default_probability",
            hover_data=["decision", "default_probability"],
            title="Utilization vs Credit Score",
            category_orders={"risk_category": RISK_ORDER},
        )
        scatter.update_layout(height=350)
        st.plotly_chart(scatter, use_container_width=True)

    if "actual_default" in view.columns:
        st.subheader("Default rate by risk band")
        default_by_band = (
            view.groupby("risk_category")["actual_default"]
            .mean()
            .reindex(RISK_ORDER)
            .reset_index()
        )
        default_by_band.columns = ["risk_category", "default_rate"]
        fig = px.bar(
            default_by_band, x="risk_category", y="default_rate",
            title="Observed default rate by band",
            color="risk_category",
            category_orders={"risk_category": RISK_ORDER},
        )
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def page_model_insights(model, approve_min: int, review_min: int):
    st.header("Model Insights")

    metrics_df = pd.DataFrame(
        {"Model": list(MODEL_METRICS.keys()), "ROC-AUC": list(MODEL_METRICS.values())}
    ).sort_values("ROC-AUC", ascending=True)

    fig = px.bar(
        metrics_df, x="ROC-AUC", y="Model", orientation="h",
        title="Model comparison (ROC-AUC)", text="ROC-AUC",
        color="ROC-AUC", color_continuous_scale="Blues",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(height=350, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Decision threshold simulator")
    st.caption("Drag thresholds in the sidebar — this chart updates live.")
    sim = get_dataset().sample(800, random_state=99)[RAW_COLUMNS]
    sim_results = predict_batch(model, sim)
    sim_results["sim_decision"] = sim_results["credit_score"].apply(
        lambda s: "Auto-Approve" if s >= approve_min else "Manual Review" if s >= review_min else "Auto-Decline"
    )
    counts = sim_results["sim_decision"].value_counts().reindex(DECISION_ORDER).fillna(0)
    fig2 = px.funnel(
        x=counts.values, y=counts.index,
        title="Simulated decision funnel (800 applicants)",
    )
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)

    for title, filename in [
        ("Feature Importance", "feature_importance.png"),
        ("SHAP Summary", "shap_summary.png"),
        ("Credit Score Distribution", "credit_score_distribution.png"),
    ]:
        path = REPORTS_DIR / filename
        if path.exists():
            with st.expander(title, expanded=False):
                st.image(str(path), use_container_width=True)


def sidebar_controls():
    st.sidebar.title("🏦 Credit Risk")
    st.sidebar.markdown("Interactive scoring powered by XGBoost")

    st.sidebar.subheader("Decision thresholds")
    approve_min = st.sidebar.slider("Auto-approve min score", 650, 850, 700, 10)
    review_min = st.sidebar.slider("Manual review min score", 500, approve_min, 600, 10)
    st.sidebar.caption(f"Decline: score < {review_min}")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Live Scoring",
            "What-If Analysis",
            "Compare Scenarios",
            "Batch Scoring",
            "Portfolio Analytics",
            "Model Insights",
        ],
    )
    return page, approve_min, review_min


def main():
    init_session_state()
    page, approve_min, review_min = sidebar_controls()

    try:
        model = get_model()
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        st.info("Run `notebooks/Credit_Risk_Model.ipynb` to train the model first.")
        return

    pages = {
        "Live Scoring": lambda: page_live_scoring(model, approve_min, review_min),
        "What-If Analysis": lambda: page_what_if(model, approve_min, review_min),
        "Compare Scenarios": page_compare_scenarios,
        "Batch Scoring": lambda: page_batch_scoring(model, approve_min, review_min),
        "Portfolio Analytics": lambda: page_portfolio_analytics(model, approve_min, review_min),
        "Model Insights": lambda: page_model_insights(model, approve_min, review_min),
    }
    pages[page]()


if __name__ == "__main__":
    main()
