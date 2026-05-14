import streamlit as st
import pandas as pd
import requests

API_URL = "https://resiliencelens-api3.jollyriver-33d3f101.eastasia.azurecontainerapps.io"

st.set_page_config(
    page_title="ResilienceLens Scrum Board",
    page_icon="🧾",
    layout="wide"
)

st.title("🧾 ResilienceLens Scrum Developer Ticket Board")
st.caption("ML-based upcoming failure prediction converted into Scrum-style engineering tickets")

try:
    health = requests.get(f"{API_URL}/", timeout=3)
    api_running = health.status_code == 200
except Exception:
    api_running = False

if not api_running:
    st.error("API is not running. Start it first:")
    st.code("uvicorn ticket_api:app --reload --port 8000")
    st.stop()

st.success("✅ API Connected")

st.sidebar.header("Generate ML-Based Ticket")

service = st.sidebar.selectbox(
    "Service",
    ["auth", "payments", "orders", "search", "recommendation", "analytics", "notifications", "media"]
)

cloud = st.sidebar.selectbox("Cloud", ["AWS", "GCP", "Azure"])
cloud_status = st.sidebar.selectbox("Cloud Status", ["operational", "degraded", "down"])
asn = st.sidebar.selectbox("ASN", ["AS16509", "AS15169", "AS8075"])
network_status = st.sidebar.selectbox("Network Status", ["stable", "unstable", "down"])
latency = st.sidebar.slider("Latency", 50, 600, 500)

if st.sidebar.button("🚨 Generate Scrum Ticket"):
    payload = {
        "service": service,
        "cloud": cloud,
        "cloud_status": cloud_status,
        "asn": asn,
        "network_status": network_status,
        "latency": latency
    }

    response = requests.post(f"{API_URL}/detect", json=payload)

    if response.status_code == 200:
        data = response.json()
        if data:
            st.sidebar.success(f"{len(data)} ticket(s) generated")
        else:
            st.sidebar.warning("No ticket generated. Try cloud down, network down, latency 500.")
    else:
        st.sidebar.error("Ticket generation failed")

response = requests.get(f"{API_URL}/tickets")

if response.status_code != 200:
    st.error("Could not load tickets")
    st.stop()

tickets = response.json()

if not tickets:
    st.warning("No tickets yet.")
    st.info("Use sidebar values: cloud_status = down, network_status = down, latency = 500")
    st.stop()

df = pd.DataFrame(tickets)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Tickets", len(df))
m2.metric("Open Tickets", len(df[df["status"] == "open"]))
m3.metric("P0 / P1 Tickets", len(df[df["priority"].isin(["P0", "P1"])]))
m4.metric("Critical Tickets", len(df[df["severity"] == "critical"]))

st.divider()

f1, f2, f3, f4 = st.columns(4)

with f1:
    status_filter = st.selectbox("Status", ["all"] + sorted(df["status"].unique().tolist()))

with f2:
    priority_filter = st.selectbox("Priority", ["all"] + sorted(df["priority"].unique().tolist()))

with f3:
    severity_filter = st.selectbox("Severity", ["all"] + sorted(df["severity"].unique().tolist()))

with f4:
    category_filter = st.selectbox("Category", ["all"] + sorted(df["category"].unique().tolist()))

filtered = df.copy()

if status_filter != "all":
    filtered = filtered[filtered["status"] == status_filter]

if priority_filter != "all":
    filtered = filtered[filtered["priority"] == priority_filter]

if severity_filter != "all":
    filtered = filtered[filtered["severity"] == severity_filter]

if category_filter != "all":
    filtered = filtered[filtered["category"] == category_filter]

st.subheader("📊 Ticket Summary")

c1, c2 = st.columns(2)

with c1:
    st.write("By Priority")
    st.bar_chart(df["priority"].value_counts())

with c2:
    st.write("By Category")
    st.bar_chart(df["category"].value_counts())

st.divider()

st.subheader("📌 Scrum Tickets")

def icon_priority(value):
    if value == "P0":
        return "🔴"
    if value == "P1":
        return "🟠"
    if value == "P2":
        return "🟡"
    return "🟢"


for _, ticket in filtered.iterrows():
    with st.container(border=True):
        top1, top2, top3 = st.columns([3, 1, 1])

        with top1:
            st.markdown(f"### {ticket['title']}")
            st.caption(f"{ticket['id']} | {ticket['created_at']}")

        with top2:
            st.markdown(f"**Priority:** {icon_priority(ticket['priority'])} {ticket['priority']}")
            st.markdown(f"**Severity:** {ticket['severity'].upper()}")

        with top3:
            st.markdown(f"**Status:** {ticket['status']}")
            st.markdown(f"**Team:** {ticket['assigned_team']}")

        st.markdown("#### 👤 User Story")
        st.info(ticket["user_story"])

        st.markdown("#### 🚨 Problem Summary")
        st.write(ticket["problem_summary"])

        st.markdown("#### 📉 Business Impact")
        st.write(ticket["business_impact"])

        st.markdown("#### 🧠 Predicted Root Cause")
        st.write(ticket["predicted_root_cause"])

        st.markdown("#### 🔧 Proposed Fix")
        for fix in ticket["proposed_fix"]:
            st.write(f"- {fix}")

        st.markdown("#### 🛠️ Implementation Tasks")
        for task in ticket["implementation_tasks"]:
            st.checkbox(task, key=f"{ticket['id']}_{task}")

        st.markdown("#### ✅ Acceptance Criteria")
        for criteria in ticket["acceptance_criteria"]:
            st.write(f"- {criteria}")

        st.markdown("#### 🎯 Definition of Done")
        for done in ticket["definition_of_done"]:
            st.write(f"- {done}")

        st.caption(f"ML / Risk Confidence: {ticket['ml_confidence']}")

st.divider()

with st.expander("View Raw Ticket Table"):
    st.dataframe(filtered, use_container_width=True)

st.markdown("### 📥 Export")
st.markdown(f"[Download Scrum Tickets CSV]({API_URL}/tickets/export)")