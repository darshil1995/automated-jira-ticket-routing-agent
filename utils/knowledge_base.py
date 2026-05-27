"""
Knowledge Base Builder — RAG (Retrieval Augmented Generation) layer.

What is RAG?
    Instead of asking the LLM to generate a fix from memory (which risks
    hallucination), we first retrieve relevant documents from a curated
    internal knowledge base, then pass those documents as context to the LLM.
    The LLM generates its response grounded in real, verified content.

    Think of it as giving the LLM an open-book exam instead of a closed one.

In this project the knowledge base contains:
    - Internal runbooks (step-by-step resolution guides)
    - Past incident post-mortems
    - Architecture decision records (ADRs)

In production you would replace the hardcoded docs below with:
    - A fetch from Confluence / Notion / Google Drive
    - A query to an S3 bucket of markdown runbooks
    - A connection to Amazon OpenSearch for persistent vector storage

Dependencies:
    - faiss-cpu: Facebook AI Similarity Search — fast local vector index
    - langchain_openai: OpenAIEmbeddings to convert text into vectors
    - langchain_community: FAISS wrapper for LangChain
"""

import logging
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal Knowledge Base Documents
# ---------------------------------------------------------------------------

# These are your "runbooks" - the internal knowledge your Resolution Agent
# will search through when generating a fix.
# In production: fetch these from S3, Confluence, or a database at startup.
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
            "Symptoms: users cannot log in, 401 errors in auth service logs, "
            "token validation failures. "
            "Resolution steps: "
            "1. Confirm in auth service logs: grep 'JWT expired' /var/log/auth-service.log. "
            "2. Generate new signing key: openssl genrsa -out new_key.pem 2048. "
            "3. Update secret in AWS Secrets Manager: auth/jwt-signing-key. "
            "4. Redeploy auth service to pick up new key: kubectl rollout restart deployment/auth-service. "
            "5. Verify login flow works end-to-end in staging before closing. "
            "Owner: auth-team. Escalation: #oncall-auth."
        ),
        metadata={"category": "authentication", "priority": "P2", "source": "runbook-auth-001"}
    ),
    Document(
        page_content=(
            "Runbook: Frontend Build — UI Component Regression. "
            "Symptoms: button not rendering, UI broken after deploy, "
            "CSS missing or JavaScript errors in browser console. "
            "Resolution steps: "
            "1. Check recent deployments in CI/CD pipeline for frontend changes. "
            "2. Reproduce in staging environment. "
            "3. Run visual regression tests: npm run test:visual. "
            "4. If confirmed regression: revert frontend deployment. "
            "   git revert <commit> && git push origin main. "
            "5. Hot-fix forward if revert is not possible. "
            "6. Open JIRA ticket with browser console screenshots attached. "
            "Owner: frontend-team. Escalation: #oncall-frontend."
        ),
        metadata={"category": "ui_bug", "priority": "P3", "source": "runbook-fe-001"}
    ),
    Document(
        page_content=(
            "Runbook: Performance Degradation — High Latency Spike. "
            "Symptoms: API response times above 2s, timeout errors, "
            "CloudWatch showing elevated p99 latency. "
            "Resolution steps: "
            "1. Check CloudWatch metrics for CPU, memory, and network on affected services. "
            "2. Check if a recent deployment coincides with the latency spike. "
            "3. Enable circuit breaker if upstream dependency is degraded. "
            "4. Scale out affected service if resource-bound: "
            "   aws autoscaling set-desired-capacity --desired-capacity 6. "
            "5. If no resource issue: profile slow endpoints with X-Ray traces. "
            "Owner: platform-engineering. Escalation: #oncall-platform."
        ),
        metadata={"category": "performance", "priority": "P2", "source": "runbook-perf-001"}
    ),
    Document(
        page_content=(
            "Runbook: Security — Unexpected IAM Permission Change. "
            "Symptoms: access denied errors for existing users, "
            "CloudTrail showing unexpected IAM policy modification. "
            "Resolution steps: "
            "1. Review CloudTrail logs for IAM events in last 24h: "
            "   filter eventSource=iam.amazonaws.com. "
            "2. Identify who made the change and from which IP. "
            "3. Revert IAM policy to last known good state using version history. "
            "4. Rotate credentials for any affected service accounts. "
            "5. File security incident report with InfoSec team within 1 hour. "
            "Owner: security-team. Escalation: #security-incidents."
        ),
        metadata={"category": "security", "priority": "P1", "source": "runbook-sec-001"}
    ),
]


# ---------------------------------------------------------------------------
# Knowledge Base Class
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """
    Manages the FAISS vector store used by the Resolution Agent for RAG.

    On first use, it converts all runbook documents into vector embeddings
    using OpenAI's embedding model and stores them in a local FAISS index.
    On subsequent calls within the same Lambda execution context, it reuses
    the already-built index (warm start optimisation).

    Attributes:
        vectorstore (FAISS | None): The in-memory FAISS vector index.
            None until build() is called.
        embeddings (OpenAIEmbeddings): The embedding model used to convert
            text into vectors for similarity search.
    """

    def __init__(self):
        """
        Initialises the KnowledgeBase.

        Does not build the index yet — deferred until first use so that
        Lambda cold starts are not penalised by embedding calls before
        the first real invocation.
        """
        logger.info("KnowledgeBase initialising | documents=%d", len(RUNBOOK_DOCUMENTS))
        self.vectorstore = None

        # OpenAI's text-embedding-ada-002 is the default embedding model.
        # It converts a string into a 1536-dimension vector that captures
        # semantic meaning — similar text produces similar vectors.
        self.embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)

    def build(self) -> None:
        """
        Converts all runbook documents into vectors and loads them into FAISS.

        FAISS (Facebook AI Similarity Search) stores these vectors in memory
        and provides sub-millisecond nearest-neighbour search — given a query
        vector, it finds the most semantically similar documents instantly.

        This is called once at Resolution Agent startup. In production you
        would persist the FAISS index to S3 and load it here instead of
        rebuilding from scratch on every cold start.
        """
        logger.info("Building FAISS vector index | documents=%d", len(RUNBOOK_DOCUMENTS))

        self.vectorstore = FAISS.from_documents(
            documents=RUNBOOK_DOCUMENTS,
            embedding=self.embeddings
        )

        logger.info("FAISS index built successfully")

    def search(self, query: str, top_k: int = 2) -> list[Document]:
        """
        Searches the knowledge base for the most relevant runbooks.

        Converts the query string into a vector and finds the top_k
        most similar documents in the FAISS index by cosine similarity.

        Args:
            query: The issue description or triage summary to search with.
            top_k: Number of documents to return. Default 2 keeps the
                   LLM context window manageable while providing enough
                   coverage for most issues.

        Returns:
            List of Document objects ordered by relevance (most relevant first).

        Raises:
            RuntimeError: If search is called before build().
        """
        if self.vectorstore is None:
            logger.error("KnowledgeBase.search() called before build()")
            raise RuntimeError("Knowledge base not built. Call build() first.")

        logger.info("Searching knowledge base | query='%s' | top_k=%d", query[:80], top_k)

        results = self.vectorstore.similarity_search(query=query, k=top_k)

        logger.info(
            "Knowledge base search complete | results=%d | sources=%s",
            len(results),
            [doc.metadata.get("source") for doc in results]
        )

        return results