"""
Main Streamlit entry point.
Handles login, role selection, and dispatches to the correct view.
"""
import streamlit as st
import api_client as api

st.set_page_config(
    page_title = "Hospital AI Dashboard",
    page_icon  = "🏥",
    layout     = "wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; }
.risk-high   { color: #d32f2f; font-weight: bold; }
.risk-mod    { color: #f57c00; font-weight: bold; }
.risk-low    { color: #388e3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Login screen ──────────────────────────────────────────────────────────────
if "token" not in st.session_state:
    st.title("🏥 Hospital Predictive Optimization System")
    st.markdown("---")

    col_l, col_mid, col_r = st.columns([1, 2, 1])
    with col_mid:
        st.subheader("Sign In")
        role     = st.selectbox("Role", ["Doctor", "Manager", "Executive"])
        username = st.text_input("Username", value=f"{role.lower()}_user")

        if st.button("Sign In", use_container_width=True, type="primary"):
            with st.spinner("Authenticating…"):
                try:
                    token = api.get_token(username)
                    st.session_state.token    = token
                    st.session_state.role     = role
                    st.session_state.username = username
                    st.rerun()
                except Exception as exc:
                    st.error(f"Login failed — is the API running?\n\n`{exc}`")
    st.stop()

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Hospital AI")
    st.markdown(f"**User:** `{st.session_state.username}`")
    st.markdown(f"**Role:** `{st.session_state.role}`")
    st.markdown("---")

    VIEW_ICONS = {
        "Doctor — Patient Risk":      "👨‍⚕️",
        "Manager — Staffing":         "📋",
        "Executive — Forecasting":    "📊",
    }
    view = st.radio("View", list(VIEW_ICONS.keys()))

    st.markdown("---")
    if st.button("Sign Out"):
        for key in ["token", "role", "username"]:
            st.session_state.pop(key, None)
        st.rerun()

# ── View dispatch ─────────────────────────────────────────────────────────────
token = st.session_state.token

if "Doctor" in view:
    from views.doctor import render
    render(token)
elif "Manager" in view:
    from views.manager import render
    render(token)
else:
    from views.executive import render
    render(token)
