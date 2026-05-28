# utils/knowledge_base.py

import logging
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import config

logger = logging.getLogger(__name__)

# This is just for our project. In production, load from S3, Confluence, or OpenSearch at startup.
RUNBOOK_DOCUMENTS = [
    Document(
        page_content=(
            "Runbook: Database Connection Pool Exhaustion. "
            "Symptoms: connection timeout errors, slow queries, users unable to log in. "
            "Resolution steps: "
            "1. Check current pool size with: SHOW STATUS LIKE 'Threads_connected'. "
            "2. Identify long-running queries: SELECT * FROM information_schema.processlist WHERE time > 30. "
            "3. Kill blocking queries if safe to do so. "
            "4. Restart connection pool manager: systemctl restart app-connection-pool. "
            "5. Increase pool size in config if recurring: max_connections=200. "
            "6. Monitor for 15 minutes post-fix. "
            "Owner: platform-engineering. Escalation: #oncall-platform."
        ),
        metadata={"category": "infrastructure", "priority": "P1", "source": "runbook-db-001"}
    ),
    Document(
        page_content=(
            "Runbook: Authentication Service — JWT Signing Key Expiry. "
            "Symptoms: users cannot log in, 401 errors in auth service logs, token validation failures. "
            "Resolution steps: "
            "1. Confirm in auth service logs: grep 'JWT expired' /var/log/auth-service.log. "
            "2. Generate new signing key: openssl genrsa -out new_key.pem 2048. "
            "3. Update secret in AWS Secrets Manager: auth/jwt-signing-key. "
            "4. Redeploy auth service: kubectl rollout restart deployment/auth-service. "
            "5. Verify login flow end-to-end in staging before closing. "
            "Owner: auth-team. Escalation: #oncall-auth."
        ),
        metadata={"category": "authentication", "priority": "P2", "source": "runbook-auth-001"}
    ),
    Document(
        page_content=(
            "Runbook: Frontend Build — UI Component Regression. "
            "Symptoms: button not rendering, UI broken after deploy, CSS missing or JS errors in console. "
            "Resolution steps: "
            "1. Check recent CI/CD deployments for frontend changes. "
            "2. Reproduce in staging environment. "
            "3. Run visual regression tests: npm run test:visual. "
            "4. If confirmed regression: git revert <commit> && git push origin main. "
            "5. Hot-fix forward if revert is not possible. "
            "6. Attach browser console screenshots to JIRA ticket. "
            "Owner: frontend-team. Escalation: #oncall-frontend."
        ),
        metadata={"category": "ui_bug", "priority": "P3", "source": "runbook-fe-001"}
    ),
    Document(
        page_content=(
            "Runbook: Performance Degradation — High Latency Spike. "
            "Symptoms: API response times above 2s, timeout errors, elevated p99 latency in CloudWatch. "
            "Resolution steps: "
            "1. Check CloudWatch metrics for CPU, memory, and network on affected services. "
            "2. Check if a recent deployment coincides with the latency spike. "
            "3. Enable circuit breaker if upstream dependency is degraded. "
            "4. Scale out if resource-bound: aws autoscaling set-desired-capacity --desired-capacity 6. "
            "5. Profile slow endpoints with X-Ray traces if no resource issue found. "
            "Owner: platform-engineering. Escalation: #oncall-platform."
        ),
        metadata={"category": "performance", "priority": "P2", "source": "runbook-perf-001"}
    ),
    Document(
        page_content=(
            "Runbook: Security — Unexpected IAM Permission Change. "
            "Symptoms: access denied errors for existing users, unexpected IAM policy modification in CloudTrail. "
            "Resolution steps: "
            "1. Review CloudTrail logs for IAM events in last 24h: filter eventSource=iam.amazonaws.com. "
            "2. Identify who made the change and from which IP. "
            "3. Revert IAM policy to last known good state using version history. "
            "4. Rotate credentials for affected service accounts. "
            "5. File security incident report with InfoSec within 1 hour. "
            "Owner: security-team. Escalation: #security-incidents."
        ),
        metadata={"category": "security", "priority": "P1", "source": "runbook-sec-001"}
    ),
]


class KnowledgeBase:
    """FAISS-backed vector store for runbook retrieval (RAG layer)."""

    def __init__(self):
        logger.info("KnowledgeBase initialising | documents=%d", len(RUNBOOK_DOCUMENTS))
        self.vectorstore: FAISS | None = None
        self.embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)

    def build(self) -> None:
        """
        Embeds all runbook documents and loads them into a FAISS index.

        Deferred from __init__ so Lambda cold starts don't pay the
        embedding API cost before the first real invocation.
        """
        logger.info("Building FAISS index | documents=%d", len(RUNBOOK_DOCUMENTS))
        self.vectorstore = FAISS.from_documents(
            documents=RUNBOOK_DOCUMENTS,
            embedding=self.embeddings
        )
        logger.info("FAISS index ready")

    def search(self, query: str, top_k: int = 2) -> list[Document]:
        """
        Returns the top_k runbooks most semantically similar to query.

        top_k=2 keeps the LLM prompt size predictable while covering
        the majority of single-category incidents.

        Raises:
            RuntimeError: If called before build().
        """
        if self.vectorstore is None:
            raise RuntimeError("KnowledgeBase.build() must be called before search().")

        logger.info("KB search | query='%s' | top_k=%d", query[:80], top_k)
        results = self.vectorstore.similarity_search(query=query, k=top_k)
        logger.info(
            "KB search complete | hits=%d | sources=%s",
            len(results),
            [doc.metadata.get("source") for doc in results]
        )
        return results