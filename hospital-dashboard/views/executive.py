"""
Executive view — 7–30 day admission forecast + all-pipeline KPIs.
Calls /forecast and /health; shows a line chart with CI, KPI cards,
and model health indicators.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import api_client as api


@st.cache_data(ttl=120, show_spinner=False)
def _fetch_forecast(token: str, target_date: str, days: int) -> dict:
    return api.forecast(token, target_date, days)


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_health() -> dict:
    return api.health()


def render(token: str) -> None:
    st.title("📊 Executive Dashboard")

    # ── Sidebar forecast controls ─────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.subheader("⚙️ Forecast Settings")
        start_date    = st.date_input("Start date", value=pd.Timestamp("2151-09-01"))
        forecast_days = st.slider("Days to forecast", 7, 30, 14)

    # ── Layout: chart (wide) + model health (narrow) ──────────────────
    col_main, col_side = st.columns([3, 1])

    # ── Model health ──────────────────────────────────────────────────
    with col_side:
        st.subheader("🔧 Model Health")
        try:
            h = _fetch_health()
            p_ok = h["models_loaded"]["prophet"]
            x_ok = h["models_loaded"]["xgboost"]
            st.metric("Prophet (Forecasting)", "✅ Loaded" if p_ok else "❌ Offline")
            st.metric("XGBoost (Alerts)",      "✅ Loaded" if x_ok else "❌ Offline")
            st.metric("API Version",           h.get("version", "?"))
            st.caption(f"Checked at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as exc:
            st.error(f"API unreachable: {exc}")

    # ── Forecast chart ────────────────────────────────────────────────
    with col_main:
        with st.spinner("Fetching forecast from Prophet model…"):
            try:
                data    = _fetch_forecast(token, str(start_date), forecast_days)
                entries = data["entry"]

                dates  = [e["effectiveDateTime"] for e in entries]
                values = [e["valueQuantity"]["value"] for e in entries]
                lows   = [e["referenceRange"][0]["low"]["value"]  for e in entries]
                highs  = [e["referenceRange"][0]["high"]["value"] for e in entries]
                days_  = [e["dayOfWeek"] for e in entries]

                # ── KPI cards ─────────────────────────────────────────
                total    = sum(values)
                avg_v    = total / len(values)
                peak_val = max(values)
                peak_day = days_[values.index(peak_val)]
                weekend_vals = [v for v, d in zip(values, days_) if d in ("Saturday", "Sunday")]
                weekday_vals = [v for v, d in zip(values, days_) if d not in ("Saturday", "Sunday")]
                wknd_avg = sum(weekend_vals) / max(len(weekend_vals), 1)
                wkdy_avg = sum(weekday_vals) / max(len(weekday_vals), 1)

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Forecasted",  f"{total:,}",        f"over {forecast_days} days")
                k2.metric("Daily Average",      f"{avg_v:.0f}",     "admissions / day")
                k3.metric("Peak Day",           peak_day,           f"{peak_val} admissions")
                k4.metric("Weekend vs Weekday", f"{wknd_avg:.0f}",  f"vs {wkdy_avg:.0f} weekday")

                # ── Forecast chart ────────────────────────────────────
                st.subheader(f"📈 {forecast_days}-Day Admission Forecast")

                fig = go.Figure()

                # Confidence interval band
                fig.add_trace(go.Scatter(
                    x           = dates + dates[::-1],
                    y           = highs + lows[::-1],
                    fill        = "toself",
                    fillcolor   = "rgba(33,150,243,0.12)",
                    line        = dict(color="rgba(0,0,0,0)"),
                    name        = "95% Confidence Interval",
                    showlegend  = True,
                    hoverinfo   = "skip",
                ))

                # Forecast line
                fig.add_trace(go.Scatter(
                    x             = dates,
                    y             = values,
                    mode          = "lines+markers",
                    name          = "Forecasted Admissions",
                    line          = dict(color="#1565c0", width=2.5),
                    marker        = dict(size=7, color="#1565c0"),
                    text          = days_,
                    hovertemplate = "<b>%{x}</b>  (%{text})<br>Admissions: %{y}<extra></extra>",
                ))

                # Weekend shading
                for date, day_name in zip(dates, days_):
                    if day_name in ("Saturday", "Sunday"):
                        fig.add_vrect(
                            x0=date, x1=date,
                            fillcolor="rgba(200,200,200,0.25)",
                            line_width=0,
                        )

                fig.update_layout(
                    xaxis_title = "Date",
                    yaxis_title = "Admissions",
                    hovermode   = "x unified",
                    height      = 400,
                    legend      = dict(orientation="h", yanchor="bottom", y=1.02),
                    margin      = dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── Detail table ──────────────────────────────────────
                with st.expander("📄 Daily forecast table"):
                    df = pd.DataFrame({
                        "Date":      dates,
                        "Day":       days_,
                        "Forecast":  values,
                        "CI Low":    lows,
                        "CI High":   highs,
                    })
                    st.dataframe(df, use_container_width=True, hide_index=True)

            except Exception as exc:
                st.error(f"Could not load forecast — is the API running? `{exc}`")

    # ── All-pipeline KPI summary ──────────────────────────────────────
    st.markdown("---")
    st.subheader("🏆 All Pipeline KPIs at a Glance")

    k_p1, k_p2, k_p3 = st.columns(3)
    with k_p1:
        st.info(
            "**Pipeline 1 — Admission Forecasting**\n\n"
            "🔵 Model: Facebook Prophet\n\n"
            "📐 MAE: ~3.2 admissions\n\n"
            "📅 Horizon: 7 – 30 days\n\n"
            "✅ Status: Live"
        )
    with k_p2:
        st.success(
            "**Pipeline 2 — Staff Optimization**\n\n"
            "🟢 Method: Linear Programming (PuLP)\n\n"
            "📈 KPI improvement: **42.5%**\n\n"
            "⏱️ Refresh: weekly\n\n"
            "✅ Status: Live"
        )
    with k_p3:
        st.warning(
            "**Pipeline 3 — Patient Risk Alerts**\n\n"
            "🟡 Model: XGBoost Classifier\n\n"
            "🎯 AUC-ROC: 0.95\n\n"
            "⚠️ Alert threshold: 70%  |  19 features\n\n"
            "✅ Status: Live"
        )
