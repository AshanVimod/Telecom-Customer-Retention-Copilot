"""Streamlit page for a retention officer.

Run from the project root:  streamlit run src/app/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from agent import run_agent  # noqa: E402
from tools import CUSTOMERS, get_churn_risk  # noqa: E402

st.set_page_config(page_title="Retention Copilot", layout="wide")
st.title("Telecom Retention Copilot")
st.caption("Demo with public data and fictional policies. All outputs are drafts for an officer to review.")


@st.cache_data(ttl=600, show_spinner="Scoring customer...")
def churn_risk(customer_id: str) -> dict:
    return get_churn_risk(customer_id)


customer_id = st.sidebar.selectbox("Customer", CUSTOMERS.index[:300])
language = st.sidebar.radio("SMS language", ["English", "Sinhala", "Tamil"])
extra = st.sidebar.text_area("Extra instruction (optional)", placeholder="e.g. Give them 50% off")

left, right = st.columns([1, 2])

with left:
    st.subheader("Churn risk")
    try:
        risk = churn_risk(customer_id)
    except Exception as e:
        st.error(f"Could not reach the churn endpoint: {e}")
        st.stop()
    st.metric("Churn probability", f"{risk['churn_probability']:.0%}", risk["risk_level"], delta_color="off")
    st.write("**Top reasons**")
    for reason in risk["top_reasons"]:
        st.write(f"- {reason}")
    st.write("**Profile**")
    st.json(risk["profile"])

with right:
    st.subheader("Recommendation")
    if st.button("Generate retention recommendation", type="primary"):
        ask = f"Assess customer {customer_id} and draft the SMS"
        ask += "." if language == "English" else f" in {language}."
        if extra.strip():
            ask += f" Officer's note: {extra.strip()}"
        with st.spinner("Agent working..."):
            answer, trace = run_agent(ask)
        st.markdown(answer)
        with st.expander("Agent steps (tool calls)"):
            st.json(trace)
    else:
        st.info("Pick a customer, then generate a recommendation.")
