"""
Doctor view — real-time patient risk monitor.
Shows 5 sample patients ranked by XGBoost risk score,
with a SHAP waterfall chart for each.
"""
import streamlit as st
import plotly.graph_objects as go
import api_client as api

# ── Sample patient roster ─────────────────────────────────────────────────────
PATIENTS = [
    {
        "id": "PT-001", "name": "Maria Chen",       "age": 83, "ward": "ICU",
        "vitals": {
            "heart_rate_mean": 87.52, "systolic_bp_mean": 110.55, "spo2_mean": 95.52,
            "temperature_c_mean": 37.42, "respiratory_rate_mean": 23.87,
            "heart_rate_max": 125.0, "systolic_bp_min": 42.0, "spo2_min": 87.0,
            "temperature_c_max": 38.29, "respiratory_rate_max": 43.0,
            "heart_rate_last": 97.0, "spo2_last": 94.0,
            "creatinine_last": 1.7, "glucose_last": 124.0, "hemoglobin_last": 7.4,
            "wbc_last": 6.0, "lactate_last": 1.0,
            "temperature_c_was_missing": 1.0, "age_at_admission": 83.0,
        },
    },
    {
        "id": "PT-002", "name": "James Rodriguez",  "age": 58, "ward": "Cardiology",
        "vitals": {
            "heart_rate_mean": 95.0, "systolic_bp_mean": 95.0, "spo2_mean": 93.0,
            "temperature_c_mean": 38.1, "respiratory_rate_mean": 20.0,
            "heart_rate_max": 110.0, "systolic_bp_min": 75.0, "spo2_min": 91.0,
            "temperature_c_max": 38.8, "respiratory_rate_max": 26.0,
            "heart_rate_last": 100.0, "spo2_last": 92.0,
            "creatinine_last": 1.4, "glucose_last": 145.0, "hemoglobin_last": 10.2,
            "wbc_last": 12.0, "lactate_last": 2.2,
            "temperature_c_was_missing": 0.0, "age_at_admission": 58.0,
        },
    },
    {
        "id": "PT-003", "name": "Ahmed Hassan",     "age": 72, "ward": "Pulmonology",
        "vitals": {
            "heart_rate_mean": 88.0, "systolic_bp_mean": 135.0, "spo2_mean": 91.0,
            "temperature_c_mean": 37.8, "respiratory_rate_mean": 22.0,
            "heart_rate_max": 105.0, "systolic_bp_min": 85.0, "spo2_min": 88.0,
            "temperature_c_max": 38.5, "respiratory_rate_max": 30.0,
            "heart_rate_last": 92.0, "spo2_last": 90.0,
            "creatinine_last": 1.1, "glucose_last": 115.0, "hemoglobin_last": 11.5,
            "wbc_last": 9.5, "lactate_last": 1.6,
            "temperature_c_was_missing": 0.0, "age_at_admission": 72.0,
        },
    },
    {
        "id": "PT-004", "name": "Sarah Johnson",    "age": 45, "ward": "General",
        "vitals": {
            "heart_rate_mean": 78.0, "systolic_bp_mean": 118.0, "spo2_mean": 97.0,
            "temperature_c_mean": 37.0, "respiratory_rate_mean": 16.0,
            "heart_rate_max": 88.0, "systolic_bp_min": 105.0, "spo2_min": 95.0,
            "temperature_c_max": 37.5, "respiratory_rate_max": 20.0,
            "heart_rate_last": 76.0, "spo2_last": 97.0,
            "creatinine_last": 0.8, "glucose_last": 105.0, "hemoglobin_last": 13.5,
            "wbc_last": 7.0, "lactate_last": 1.0,
            "temperature_c_was_missing": 0.0, "age_at_admission": 45.0,
        },
    },
    {
        "id": "PT-005", "name": "Emma Williams",    "age": 28, "ward": "Emergency",
        "vitals": {
            "heart_rate_mean": 72.0, "systolic_bp_mean": 122.0, "spo2_mean": 99.0,
            "temperature_c_mean": 36.8, "respiratory_rate_mean": 14.0,
            "heart_rate_max": 82.0, "systolic_bp_min": 112.0, "spo2_min": 97.0,
            "temperature_c_max": 37.2, "respiratory_rate_max": 18.0,
            "heart_rate_last": 70.0, "spo2_last": 99.0,
            "creatinine_last": 0.7, "glucose_last": 88.0, "hemoglobin_last": 14.2,
            "wbc_last": 5.5, "lactate_last": 0.6,
            "temperature_c_was_missing": 0.0, "age_at_admission": 28.0,
        },
    },
]


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_all_risks(token: str) -> list[dict]:
    """Fetch risk scores for all sample patients (cached 60 s)."""
    results = []
    for p in PATIENTS:
        try:
            resp       = api.alert(token, p["vitals"])
            pred       = resp["prediction"][0]
            risk_score = pred["probabilityDecimal"]
            results.append({
                **p,
                "risk_score": risk_score,
                "risk_pct":   round(risk_score * 100, 1),
                "risk_level": pred["riskLevel"],
                "explanation": resp.get("explanation", []),
            })
        except Exception as exc:
            results.append({**p, "risk_score": -1.0, "risk_pct": "?", "risk_level": "ERROR", "explanation": [], "error": str(exc)})
    return sorted(results, key=lambda x: x["risk_score"], reverse=True)


def _risk_icon(level: str) -> str:
    return {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(level, "⚪")


def _shap_chart(explanation: list) -> go.Figure:
    """Horizontal bar chart of SHAP contributions."""
    labels = [f["label"] for f in explanation]
    shaps  = [f["shap"]  for f in explanation]
    values = [f["value"] for f in explanation]
    colors = ["#d32f2f" if v > 0 else "#1565c0" for v in shaps]

    fig = go.Figure(go.Bar(
        x            = shaps,
        y            = labels,
        orientation  = "h",
        marker_color = colors,
        text         = [str(v) for v in values],
        textposition = "outside",
        hovertemplate = "<b>%{y}</b><br>Value: %{text}<br>SHAP: %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.update_layout(
        title      = "Risk Factor Contributions (SHAP)",
        xaxis_title= "← decreases risk   |   increases risk →",
        yaxis      = dict(autorange="reversed"),
        height     = 340,
        margin     = dict(l=10, r=30, t=40, b=0),
    )
    return fig


def render(token: str) -> None:
    st.title("👨‍⚕️ Patient Risk Monitor")

    # ── Controls ──────────────────────────────────────────────────────
    col_h, col_th = st.columns([4, 1])
    with col_th:
        threshold = st.slider(
            "Alert threshold", 0.0, 1.0, 0.70, 0.05,
            help="Patients at or above this score trigger an alert",
        )
    with col_h:
        if st.button("🔄 Refresh scores"):
            st.cache_data.clear()

    # ── Load risk scores ──────────────────────────────────────────────
    with st.spinner("Scoring all patients via XGBoost…"):
        patients = _fetch_all_risks(token)

    alerts_count = sum(1 for p in patients if p["risk_score"] >= threshold)
    st.subheader(f"{'⚠️' if alerts_count else '✅'}  {alerts_count} alert{'s' if alerts_count != 1 else ''} — {len(patients)} patients monitored")
    st.markdown("---")

    # ── Patient cards ─────────────────────────────────────────────────
    for p in patients:
        icon  = _risk_icon(p["risk_level"])
        alert = p["risk_score"] >= threshold
        label = f"{icon}  **{p['name']}** — {p['risk_pct']}%  {'⚠️ ALERT' if alert else ''}  |  {p['ward']}  |  Age {p['age']}"

        with st.expander(label, expanded=(p["risk_level"] == "HIGH")):
            col_meta, col_chart = st.columns([1, 2])

            with col_meta:
                st.metric("Risk Score",  f"{p['risk_pct']}%")
                st.metric("Risk Level",  p["risk_level"])
                st.metric("Ward",        p["ward"])
                st.metric("Age",         f"{p['age']} yrs")

                # Progress bar
                bar_colour = "red" if p["risk_level"] == "HIGH" else "orange" if p["risk_level"] == "MODERATE" else "green"
                st.markdown(f"""
                <div style="background:#eee;border-radius:6px;height:12px;margin-top:8px">
                  <div style="background:{bar_colour};width:{p['risk_pct']}%;height:12px;border-radius:6px"></div>
                </div>
                """, unsafe_allow_html=True)

            with col_chart:
                if p.get("explanation"):
                    st.plotly_chart(_shap_chart(p["explanation"]), use_container_width=True)
                elif p.get("error"):
                    st.warning(f"Could not score: {p['error']}")
                else:
                    st.info("No SHAP explanation available.")
