"""
CreditPulse — FAISS RAG Indexer
Spec: CREDIT-003 (agent knowledge base)

Builds a FAISS vector index over financial regulation and fraud detection knowledge:
- PCI DSS compliance guidelines
- FCRA (Fair Credit Reporting Act) requirements
- Regulation E (electronic fund transfers)
- CFPB guidance on algorithmic decision-making
- Internal CreditPulse model documentation

The agent uses this to answer questions like:
  "What does PCI DSS require when flagging a transaction?"
  "What disclosures are required under FCRA when denying credit?"

Usage:
    python agent/rag/indexer.py --build
"""

import os
from pathlib import Path

import structlog
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = structlog.get_logger(__name__)

INDEX_DIR = Path("agent/rag/faiss_index")

# Regulatory and internal knowledge base (embedded directly for offline demo)
KNOWLEDGE_BASE = [
    {
        "title": "PCI DSS v4.0 — Transaction Monitoring Requirements",
        "content": """
PCI DSS Requirement 10.7: Detect and respond to failures of critical security controls.
All transaction monitoring systems must: (1) log every authorization attempt with timestamp,
merchant ID, and transaction amount; (2) retain logs for at least 12 months with 3 months
immediately available; (3) implement automated alerts for anomalous patterns including
velocity checks, amount outliers, and geographic anomalies. Flagged transactions must trigger
incident response within 4 hours for Critical severity events.
        """,
    },
    {
        "title": "FCRA — Adverse Action Notification Requirements",
        "content": """
Under the Fair Credit Reporting Act (FCRA), when a creditor takes an adverse action based
wholly or partly on information in a consumer report, they must: (1) provide written notice
to the consumer within a reasonable time; (2) identify the consumer reporting agency that
furnished the report; (3) state that the consumer reporting agency did not make the adverse
decision; (4) provide the consumer's right to obtain a free copy of their report within 60 days;
(5) provide the consumer's right to dispute inaccurate information. For AI-driven decisions,
the CFPB expects specific explanations of the top factors that led to the adverse action.
        """,
    },
    {
        "title": "Regulation E — Unauthorized Electronic Fund Transfer Liability",
        "content": """
Under Regulation E (Electronic Fund Transfer Act), consumer liability for unauthorized
transactions is limited: (1) $50 if reported within 2 business days of learning of the loss;
(2) $500 if reported after 2 but within 60 days; (3) unlimited if reported after 60 days.
Financial institutions must investigate claims within 10 business days (or 45 for new accounts).
Provisional credit must be given within 5 days if investigation extends beyond 10 days.
Fraud detection systems must support the investigation timeline with audit trails.
        """,
    },
    {
        "title": "CFPB Guidance on Algorithmic Decision-Making in Credit",
        "content": """
The Consumer Financial Protection Bureau (CFPB) has issued guidance requiring that when
AI/ML models are used for credit decisions: (1) the specific reasons for adverse actions must
be explainable — vague references to 'a complex algorithm' are insufficient; (2) lenders must
be able to identify the specific factors that contributed most to the credit decision;
(3) protected characteristics (race, sex, religion, national origin, familial status, disability)
must not be used as inputs even indirectly; (4) proxy discrimination — using zip code, surname,
or other correlated variables — violates the Equal Credit Opportunity Act (ECOA).
SHAP values and counterfactual explanations are cited as acceptable explainability approaches.
        """,
    },
    {
        "title": "CreditPulse — Fraud Risk Score Interpretation Guide",
        "content": """
CreditPulse computes a composite fraud risk score (0–100) as follows:
  Score = 0.65 × XGBoost_fraud_probability × 100 + 0.35 × IsolationForest_anomaly × 100

Risk bands:
  0–25:  CLEAR — auto-approve, standard monitoring
  26–50: LOW_RISK — approve with enhanced monitoring
  51–75: REVIEW — human review required before processing
  76–100: FRAUD — block and trigger dispute investigation

Top contributing factors (SHAP) include: transaction amount relative to account history,
transaction velocity in the past hour, merchant category risk profile, and whether the
merchant is a foreign entity. Counterfactual explanations are generated via Dice-ML.
        """,
    },
    {
        "title": "CreditPulse — Model Governance and Fairness Policy",
        "content": """
CreditPulse enforces automated fairness gates before any model version can be promoted
to production. The fairness gate (implemented as a Kiro hook in .kiro/hooks/fairness-gate.sh)
runs Fairlearn demographic parity and equal opportunity checks across account age groups,
region types (urban/suburban/rural), and account types. All fairness metrics must be < 0.05
(5 percentage point difference). Models that fail the gate cannot be promoted without
human-in-the-loop (HITL) override, which is logged to audit.fairness_overrides.
PSI drift monitoring runs weekly and triggers automatic alerts when PSI > 0.20 for any feature.
        """,
    },
    {
        "title": "XGBoost Feature Importance — CreditPulse Fraud Model",
        "content": """
The top 5 features by mean absolute SHAP value in the CreditPulse fraud detection model are:
1. txn_velocity_1h (0.342): Number of transactions in the past hour — high velocity is the
   strongest signal for card testing and account takeover fraud.
2. amount_vs_avg_ratio (0.287): Transaction amount divided by 90-day average — large deviations
   indicate potential fraud or compromised account.
3. merchant_category_encoded (0.198): Wire transfers, cryptocurrency, and gambling merchants
   have 6-8× higher base fraud rates than grocery/restaurant.
4. is_foreign_merchant (0.124): Foreign merchants have higher fraud rates due to reduced
   liability enforcement and delayed dispute resolution.
5. amount (0.089): Raw transaction amount — very large amounts are flagged but less predictive
   than the ratio feature above.
        """,
    },
    {
        "title": "Operational Runbook — High-Risk Transaction Response",
        "content": """
When CreditPulse flags a transaction as FRAUD (score > 75):
1. The transaction is blocked immediately (not just flagged).
2. A HIGH-RISK alert is published to Kafka topic alerts.fraud.
3. The account is temporarily frozen pending investigation.
4. An automated case is opened in the fraud investigation queue.
5. The customer is notified via SMS/email within 2 minutes.
6. The human fraud analyst has 4 hours to review and confirm/reverse.
7. All decisions (block, approve, escalate) are logged to audit.model_decisions.
8. If reversed, audit.hitl_overrides is updated with analyst justification.
        """,
    },
]


def build_index(embedding_model: str = "all-MiniLM-L6-v2") -> FAISS:
    """Build FAISS vector index from knowledge base documents."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    logger.info("building_rag_index", docs=len(KNOWLEDGE_BASE))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = []
    for kb_entry in KNOWLEDGE_BASE:
        chunks = splitter.split_text(kb_entry["content"].strip())
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={"title": kb_entry["title"]}))

    logger.info("chunks_created", count=len(documents))

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    index = FAISS.from_documents(documents, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.save_local(str(INDEX_DIR))
    logger.info("index_saved", path=str(INDEX_DIR), chunks=len(documents))
    return index


def load_index(embedding_model: str = "all-MiniLM-L6-v2") -> FAISS:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.build:
        index = build_index()
        print(f"Index built with {index.index.ntotal} vectors at {INDEX_DIR}")
