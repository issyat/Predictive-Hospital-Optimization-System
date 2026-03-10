"""
Manager view — optimized staff schedule.
Visualises the /staffing response as a Gantt timeline,
a coverage heatmap, and stacked shift bars.
"""
import streamlit as st
import plotly.graph_objects as go
import api_client as api


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_schedule(token: str) -> dict:
    return api.staffing(token)


def render(token: str) -> None:
    st.title("📋 Staff Schedule Optimizer")

    with st.spinner("Loading schedule…"):
        data     = _fetch_schedule(token)
    schedule = data["schedule"]

    # ── KPI cards ─────────────────────────────────────────────────────
    total_staff  = sum(s["total"] for s in schedule)
    busiest      = max(schedule, key=lambda x: x["total"])
    avg_daily    = total_staff / len(schedule)
    morning_avg  = sum(s["morning"] for s in schedule) / len(schedule)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("KPI Improvement",   data.get("kpi_improvement", "N/A"), "vs. manual schedule")
    k2.metric("Total Staff / Week", str(total_staff),                   "shifts this week")
    k3.metric("Busiest Day",        busiest["day"],                     f"{busiest['total']} staff")
    k4.metric("Daily Average",      f"{avg_daily:.1f}",                 "staff / day")

    st.markdown("---")

    # ── Row 1: Stacked bar + Heatmap ──────────────────────────────────
    col_bar, col_heat = st.columns(2)

    with col_bar:
        st.subheader("Daily Shift Breakdown")
        days   = [s["day"] for s in schedule]
        colors = {"morning": "#42a5f5", "afternoon": "#66bb6a", "night": "#5c6bc0"}
        labels = {"morning": "🌅 Morning  (06–14h)", "afternoon": "☀️ Afternoon (14–22h)", "night": "🌙 Night     (22–06h)"}

        fig = go.Figure()
        for shift, color in colors.items():
            fig.add_trace(go.Bar(
                name         = labels[shift],
                x            = days,
                y            = [s[shift] for s in schedule],
                marker_color = color,
                text         = [s[shift] for s in schedule],
                textposition = "inside",
            ))
        fig.update_layout(
            barmode     = "stack",
            xaxis_title = "Day",
            yaxis_title = "Number of Staff",
            legend      = dict(orientation="h", yanchor="bottom", y=1.02),
            height      = 380,
            margin      = dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_heat:
        st.subheader("Coverage Heatmap")
        z    = [[s["morning"], s["afternoon"], s["night"]] for s in schedule]
        text = [[str(v) for v in row] for row in z]

        fig = go.Figure(go.Heatmap(
            z            = z,
            x            = ["🌅 Morning", "☀️ Afternoon", "🌙 Night"],
            y            = [s["day"] for s in schedule],
            colorscale   = "Blues",
            text         = text,
            texttemplate = "%{text}",
            showscale    = True,
        ))
        fig.update_layout(
            height = 380,
            margin = dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Gantt-style timeline ───────────────────────────────────
    st.subheader("Weekly Shift Timeline (Gantt)")

    SHIFTS = [
        ("Morning",   "🌅 Morning  (06–14h)", 6,  8,  "#42a5f5"),
        ("Afternoon", "☀️ Afternoon (14–22h)", 14, 8,  "#66bb6a"),
        ("Night",     "🌙 Night     (22–06h)", 22, 8,  "#5c6bc0"),
    ]

    fig = go.Figure()
    for shift_key, shift_label, start_h, duration, color in SHIFTS:
        staff_counts = [s[shift_key.lower()] for s in schedule]
        fig.add_trace(go.Bar(
            name         = shift_label,
            x            = [duration] * len(schedule),
            y            = [s["day"] for s in schedule],
            base         = [start_h]  * len(schedule),
            orientation  = "h",
            marker_color = color,
            text         = [f"{n} nurses" for n in staff_counts],
            textposition = "inside",
            hovertemplate = f"<b>%{{y}}</b><br>{shift_label}<br>Staff: %{{text}}<extra></extra>",
        ))

    fig.update_layout(
        barmode     = "overlay",
        xaxis       = dict(title="Hour of day", range=[0, 30], tickvals=list(range(0, 31, 2))),
        yaxis       = dict(autorange="reversed"),
        legend      = dict(orientation="h", yanchor="bottom", y=1.02),
        height      = 320,
        margin      = dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Detail table ──────────────────────────────────────────────────
    with st.expander("📄 Full schedule table"):
        import pandas as pd
        df = pd.DataFrame(schedule)[["date", "day", "morning", "afternoon", "night", "total"]]
        df.columns = ["Date", "Day", "Morning", "Afternoon", "Night", "Total"]
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption(f"Optimisation method: {data.get('optimization', 'N/A')}  ·  Generated at: {data.get('generated_at', 'N/A')}")
