"""
ticket_api.py — ResilienceLens Scrum-Style Developer Ticket API

Run locally:
python3 -m uvicorn ticket_api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import uuid
import datetime
import io
import csv
import threading

from core.anomaly import train_anomaly_model, detect_anomaly
from core.risk_engine import calculate_risk
from core.graph_builder import build_graph
from core.ml_model import train_model
from data.cloud_status import get_cloud_status
from data.network_data import get_network_status
from utils.helpers import load_config


app = FastAPI(title="ResilienceLens Scrum Ticket API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tickets: List[dict] = []
tickets_lock = threading.Lock()

anomaly_model = None
failure_model = None
config = None
models_lock = threading.Lock()


@app.on_event("startup")
def startup():
    """
    Keep startup lightweight for Azure.
    Do NOT train ML models here.
    """
    global config
    config = load_config()
    print("✅ Scrum Ticket API started")
    print("✅ Config loaded")


def load_models():
    """
    Lazy-load models only when /detect is called.
    This avoids Azure startup timeout.
    """
    global anomaly_model, failure_model

    with models_lock:
        if anomaly_model is None:
            print("⏳ Loading anomaly model...")
            anomaly_model = train_anomaly_model()
            print("✅ Anomaly model loaded")

        if failure_model is None:
            print("⏳ Loading failure model...")
            failure_model = train_model()
            print("✅ Failure model loaded")


class Event(BaseModel):
    service: str
    cloud: str
    cloud_status: str
    asn: str
    network_status: str
    latency: int


class TicketStatusUpdate(BaseModel):
    status: str


SEVERITY_MAP = {
    "anomaly": "high",
    "cloud_outage": "critical",
    "network": "medium",
    "ml_prediction": "high",
    "topology": "critical",
}

PRIORITY_MAP = {
    "critical": "P0",
    "high": "P1",
    "medium": "P2",
    "low": "P3",
}

TEAM_MAP = {
    "auth": "Identity Backend Team",
    "payments": "Payments Backend Team",
    "orders": "Orders Backend Team",
    "search": "Search Platform Team",
    "recommendation": "ML Recommendation Team",
    "analytics": "Data Platform Team",
    "notifications": "Notification Service Team",
    "media": "Media Platform Team",
}


def get_assigned_team(service: str) -> str:
    return TEAM_MAP.get(service, "Platform / SRE Team")


def get_business_impact(service: str) -> str:
    impacts = {
        "payments": "Payment failures can cause direct revenue loss, failed transactions, and poor customer trust.",
        "auth": "Authentication failures can block users from logging in and accessing the platform.",
        "orders": "Order failures can prevent checkout completion and affect customer experience.",
        "search": "Search degradation can reduce product discovery and user engagement.",
        "recommendation": "Recommendation failures can reduce personalization and conversion.",
        "analytics": "Analytics failures can affect reporting, dashboards, and decision-making.",
        "notifications": "Notification failures can delay important user alerts and system communication.",
        "media": "Media failures can cause slow content loading and poor user experience.",
    }

    return impacts.get(
        service,
        "Service degradation can affect reliability, SLA compliance, and user experience."
    )


def make_scrum_ticket(
    category: str,
    service: str,
    title: str,
    problem_summary: str,
    root_cause: str,
    proposed_fix: List[str],
    implementation_tasks: List[str],
    acceptance_criteria: List[str],
    ml_confidence: Optional[str] = None,
) -> dict:
    severity = SEVERITY_MAP.get(category, "medium")
    priority = PRIORITY_MAP.get(severity, "P2")

    return {
        "id": f"TKT-{str(uuid.uuid4())[:8].upper()}",
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "status": "open",
        "ticket_type": "Scrum Engineering Ticket",
        "category": category,
        "severity": severity,
        "priority": priority,
        "service": service,
        "assigned_team": get_assigned_team(service),
        "ml_confidence": ml_confidence or "N/A",
        "title": title,
        "user_story": (
            f"As a platform reliability engineer, I want to proactively fix "
            f"predicted issues in the {service} service so that user-facing "
            f"failures are prevented before they occur."
        ),
        "problem_summary": problem_summary,
        "business_impact": get_business_impact(service),
        "predicted_root_cause": root_cause,
        "proposed_fix": proposed_fix,
        "implementation_tasks": implementation_tasks,
        "acceptance_criteria": acceptance_criteria,
        "definition_of_done": [
            "Code or configuration fix is implemented.",
            "Fix is tested in local or staging environment.",
            "Relevant service metrics are verified.",
            "Failure probability or risk condition is reduced.",
            "Ticket is updated with final root cause and fix summary.",
        ],
    }


def save_ticket(ticket: dict):
    with tickets_lock:
        tickets.append(ticket)


@app.post("/detect", response_model=List[dict])
def detect(event: Event):
    """
    Generate Scrum-style developer tickets from infrastructure event signals.
    """
    load_models()

    new_tickets = []
    ev = event.dict()

    # 1. Anomaly Ticket
    try:
        if detect_anomaly(anomaly_model, ev):
            ticket = make_scrum_ticket(
                category="anomaly",
                service=ev["service"],
                title=f"[P1] Reduce abnormal latency in {ev['service']} service",
                problem_summary=(
                    f"The anomaly model detected unusual latency in `{ev['service']}`. "
                    f"Current latency is {ev['latency']}ms on {ev['cloud']} with ASN {ev['asn']}."
                ),
                root_cause=(
                    "The service is likely affected by high response time caused by slow database queries, "
                    "resource saturation, inefficient API calls, traffic spike, or degraded infrastructure."
                ),
                proposed_fix=[
                    "Optimize slow API endpoints.",
                    "Identify and tune slow database queries.",
                    "Enable or adjust autoscaling policy.",
                    "Add caching for repeated requests.",
                    "Review recent deployment changes that may have increased latency.",
                ],
                implementation_tasks=[
                    "Check logs and traces for top slow endpoints.",
                    "Identify database queries taking more than expected threshold.",
                    "Add index or query optimization where required.",
                    "Check CPU, memory, and container health metrics.",
                    "Deploy fix and monitor latency after release.",
                ],
                acceptance_criteria=[
                    "P95 latency is below 200ms.",
                    "No anomaly is detected for the same service during monitoring window.",
                    "Error rate remains below 1%.",
                    "Fix evidence is attached using logs or dashboard screenshot.",
                ],
                ml_confidence="IsolationForest anomaly detected",
            )

            save_ticket(ticket)
            new_tickets.append(ticket)

    except Exception as e:
        print("Anomaly detection skipped:", str(e))

    # 2. Cloud + Network Risk Tickets
    try:
        cloud_status = get_cloud_status()
        network_status = get_network_status()

        score, issues = calculate_risk(
            config["services"],
            cloud_status,
            network_status
        )

        for issue in issues:
            if "outage" in issue:
                ticket = make_scrum_ticket(
                    category="cloud_outage",
                    service=ev["service"],
                    title=f"[P0] Add failover protection for {ev['service']} due to cloud outage risk",
                    problem_summary=(
                        f"The risk engine detected cloud outage impact. Issue: {issue}. "
                        f"Risk score is {score}. Current event cloud status is {ev['cloud_status']}."
                    ),
                    root_cause=(
                        "The service may depend on a degraded cloud region or provider, creating a "
                        "single point of failure."
                    ),
                    proposed_fix=[
                        "Enable failover to a healthy cloud region.",
                        "Add circuit breaker for unhealthy cloud dependency.",
                        "Configure health checks before routing traffic.",
                        "Add fallback path for critical service calls.",
                        "Document cloud dependency risk in architecture notes.",
                    ],
                    implementation_tasks=[
                        "Identify affected cloud region from config.",
                        "Verify service dependency on degraded region.",
                        "Add or validate failover rule.",
                        "Test failover using simulated outage.",
                        "Update incident playbook with mitigation steps.",
                    ],
                    acceptance_criteria=[
                        "Traffic can fail over to healthy region.",
                        "Health check blocks traffic to unhealthy dependency.",
                        "Service remains available during simulated outage.",
                        "No P0 cloud dependency risk remains open for this service.",
                    ],
                    ml_confidence=f"Risk score: {score}",
                )

            else:
                ticket = make_scrum_ticket(
                    category="network",
                    service=ev["service"],
                    title=f"[P2] Improve network resilience for {ev['service']} service",
                    problem_summary=(
                        f"The risk engine detected ASN/network instability. Issue: {issue}. "
                        f"Current ASN is {ev['asn']} and network status is {ev['network_status']}."
                    ),
                    root_cause=(
                        "Service requests may be affected by unstable ASN routing, packet loss, "
                        "upstream provider issues, or missing retry/fallback logic."
                    ),
                    proposed_fix=[
                        "Improve retry logic for temporary network failures.",
                        "Add timeout handling for external dependencies.",
                        "Route traffic through alternate ASN if supported.",
                        "Add monitoring for packet loss and request timeout.",
                        "Use DNS or geo-routing failover where possible.",
                    ],
                    implementation_tasks=[
                        "Check timeout and retry settings in service client code.",
                        "Validate dependency call failure handling.",
                        "Check network monitoring for ASN instability.",
                        "Add alert for repeated timeout spikes.",
                        "Test service behavior during unstable network simulation.",
                    ],
                    acceptance_criteria=[
                        "Service handles temporary network failures without crashing.",
                        "Timeout errors reduce below defined threshold.",
                        "Retry logic works without duplicate unsafe operations.",
                        "Network instability no longer causes repeated user-facing failures.",
                    ],
                    ml_confidence=f"Risk score: {score}",
                )

            save_ticket(ticket)
            new_tickets.append(ticket)

    except Exception as e:
        print("Risk engine skipped:", str(e))

    # 3. ML Failure Prediction Ticket
    try:
        import pandas as pd

        row = {
            "service": ev["service"],
            "cloud": ev["cloud"],
            "region": "us-east-1",
            "asn": ev["asn"],
            "cloud_status": ev["cloud_status"],
            "network_status": ev["network_status"],
            "latency": ev["latency"],
        }

        df_row = pd.DataFrame([row])

        # IMPORTANT:
        # This assumes core/ml_model.py returns a sklearn Pipeline
        # with preprocessing inside it.
        probability = failure_model.predict_proba(df_row)[0][1]

        if probability > 0.6:
            ticket = make_scrum_ticket(
                category="ml_prediction",
                service=ev["service"],
                title=f"[P1] Prevent predicted failure in {ev['service']} service",
                problem_summary=(
                    f"The ML model predicts {probability * 100:.1f}% probability of upcoming "
                    f"failure in `{ev['service']}`. Input signals: latency={ev['latency']}ms, "
                    f"cloud_status={ev['cloud_status']}, network_status={ev['network_status']}."
                ),
                root_cause=(
                    "The model identified a risky combination of high latency, cloud status, "
                    "and network status. This pattern is similar to previous failure cases."
                ),
                proposed_fix=[
                    "Reduce service latency by optimizing slow endpoints.",
                    "Add autoscaling rule when latency or CPU crosses threshold.",
                    "Add caching for high-frequency requests.",
                    "Strengthen retry and timeout handling for external dependencies.",
                    "Fail over traffic if cloud or network status remains degraded.",
                ],
                implementation_tasks=[
                    "Identify top 3 slow endpoints from logs/traces.",
                    "Optimize slow database queries or add required indexes.",
                    "Check autoscaling rules and update thresholds if needed.",
                    "Add Redis/cache layer for repeated read-heavy requests.",
                    "Validate retry, timeout, and fallback logic.",
                    "Deploy fix and monitor model risk score after deployment.",
                ],
                acceptance_criteria=[
                    "ML failure probability drops below 40%.",
                    "P95 latency is below 200ms.",
                    "Error rate is below 1%.",
                    "No repeated high-risk prediction occurs for the same service.",
                    "Developer documents whether the prediction was true positive or false positive.",
                ],
                ml_confidence=f"{probability * 100:.1f}%",
            )

            save_ticket(ticket)
            new_tickets.append(ticket)

    except Exception as e:
        print("ML prediction skipped:", str(e))

    # 4. Topology Risk Ticket
    try:
        graph = build_graph(config["services"])

        affected_nodes = {
            node
            for node, data in graph.nodes(data=True)
            if data.get("type") == "cloud" and ev["cloud_status"] != "operational"
        }

        if len(affected_nodes) >= 2:
            ticket = make_scrum_ticket(
                category="topology",
                service=ev["service"],
                title=f"[P0] Remove correlated topology risk affecting {ev['service']}",
                problem_summary=(
                    f"Graph analysis found {len(affected_nodes)} affected cloud nodes: "
                    f"{', '.join(list(affected_nodes)[:4])}. This indicates possible correlated failure."
                ),
                root_cause=(
                    "Multiple services may depend on shared cloud or network nodes, creating "
                    "a correlated failure path."
                ),
                proposed_fix=[
                    "Identify services sharing the same risky cloud dependency.",
                    "Add multi-region redundancy for critical services.",
                    "Reduce single-cloud dependency for high-priority flows.",
                    "Add dependency-aware health checks.",
                    "Create architecture backlog item for resilience improvement.",
                ],
                implementation_tasks=[
                    "Generate service-cloud-ASN graph.",
                    "Identify single points of failure.",
                    "Map affected services to business criticality.",
                    "Create failover plan for critical services.",
                    "Update architecture documentation.",
                ],
                acceptance_criteria=[
                    "Critical single points of failure are identified.",
                    "Failover plan is documented.",
                    "Architecture risk is added to sprint backlog.",
                    "No critical service depends on one unhealthy cloud node without fallback.",
                ],
                ml_confidence="Graph correlation detected",
            )

            save_ticket(ticket)
            new_tickets.append(ticket)

    except Exception as e:
        print("Topology analysis skipped:", str(e))

    return new_tickets


@app.get("/tickets")
def list_tickets(
    status: Optional[str] = None,
    severity: Optional[str] = None,
    category: Optional[str] = None,
    priority: Optional[str] = None,
    service: Optional[str] = None,
):
    with tickets_lock:
        result = list(tickets)

    if status:
        result = [t for t in result if t["status"] == status]

    if severity:
        result = [t for t in result if t["severity"] == severity]

    if category:
        result = [t for t in result if t["category"] == category]

    if priority:
        result = [t for t in result if t["priority"] == priority]

    if service:
        result = [t for t in result if t["service"] == service]

    return sorted(result, key=lambda t: t["created_at"], reverse=True)


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    with tickets_lock:
        for ticket in tickets:
            if ticket["id"] == ticket_id:
                return ticket

    raise HTTPException(status_code=404, detail="Ticket not found")


@app.patch("/tickets/{ticket_id}")
def update_ticket(ticket_id: str, body: TicketStatusUpdate):
    allowed_statuses = {"open", "in-progress", "resolved", "closed"}

    if body.status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Use one of: {', '.join(allowed_statuses)}"
        )

    with tickets_lock:
        for ticket in tickets:
            if ticket["id"] == ticket_id:
                ticket["status"] = body.status
                return ticket

    raise HTTPException(status_code=404, detail="Ticket not found")


@app.get("/tickets/export")
def export_tickets():
    buffer = io.StringIO()

    fieldnames = [
        "id",
        "created_at",
        "status",
        "ticket_type",
        "category",
        "severity",
        "priority",
        "service",
        "assigned_team",
        "ml_confidence",
        "title",
        "user_story",
        "problem_summary",
        "business_impact",
        "predicted_root_cause",
        "proposed_fix",
        "implementation_tasks",
        "acceptance_criteria",
        "definition_of_done",
    ]

    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()

    with tickets_lock:
        export_rows = list(tickets)

    for ticket in export_rows:
        row = ticket.copy()

        for key in [
            "proposed_fix",
            "implementation_tasks",
            "acceptance_criteria",
            "definition_of_done",
        ]:
            row[key] = " | ".join(ticket.get(key, []))

        writer.writerow({key: row.get(key, "") for key in fieldnames})

    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=scrum_developer_tickets.csv"
        },
    )


@app.get("/")
def home():
    return {
        "message": "ResilienceLens Scrum Developer Ticket API is running 🚀",
        "status": "healthy",
        "routes": [
            "/detect (POST)",
            "/tickets (GET)",
            "/tickets/{ticket_id} (GET)",
            "/tickets/{ticket_id} (PATCH)",
            "/tickets/export (GET)",
        ],
    }