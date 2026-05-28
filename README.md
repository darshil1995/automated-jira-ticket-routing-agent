# Automated JIRA Workflow & Ticket Routing Agent

A production-grade multi-agent AI system that automatically triages incoming
system alerts, generates structured resolution plans using RAG, and enforces
data privacy compliance — before filing a JIRA ticket.

Built with LangGraph, LangChain, OpenAI GPT-4o, and AWS Lambda.

---

## Architecture

```
Incoming Alert / HTTP Request
          ↓
    API Gateway / S3
          ↓
   AWS Lambda Function
          ↓
   ┌──────────────────────────────────────────┐
   │           LangGraph Pipeline             │
   │                                          │
   │  Agent A       Agent B        Agent C    │
   │  Triage   →  Resolution  →      QA       │
   │  Agent        Agent            Agent     │
   └──────────────────────────────────────────┘
          ↓
   tickets/*.json  (Mock JIRA Output)
          ↓
   CloudWatch Logs
```

### Agent Responsibilities

| Agent | Role | Key Capability |
|---|---|---|
| **Triage Agent** | Classifies incoming issue | Tool calling — queries historical incident logs |
| **Resolution Agent** | Generates fix + ticket draft | RAG — searches internal runbook knowledge base |
| **QA Agent** | Reviews for compliance | Two-layer review — regex pre-check + LLM review |

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Multi-agent orchestration | LangGraph | Stateful directed graph with shared state between agents |
| LLM | OpenAI GPT-4o | Classification, generation, and compliance review |
| Vector search | FAISS | Semantic similarity search over internal runbooks |
| Embeddings | OpenAI text-embedding-ada-002 | Converts runbooks into searchable vectors |
| Serverless runtime | AWS Lambda | Event-driven, scales to zero, no server management |
| HTTP trigger | AWS API Gateway | Public HTTPS endpoint for webhook and dashboard triggers |
| File trigger | Amazon S3 | Automated batch alert ingestion via file upload events |
| Logging | AWS CloudWatch | Structured logs with request ID tracing across all agents |
| Language | Python 3.12 | — |

---

## Project Structure

```
automated-jira-ticket-routing-agent/
├── agents/
│   ├── __init__.py
│   ├── triage_agent.py       # Agent A: classifies issue + calls log tool
│   ├── resolution_agent.py   # Agent B: RAG search + ticket draft generation
│   └── qa_agent.py           # Agent C: privacy + compliance review
├── utils/
│   ├── __init__.py
│   ├── knowledge_base.py     # FAISS vector store + runbook documents
│   └── ticket_writer.py      # Mock JIRA ticket writer (saves to tickets/)
├── tests/
│   ├── __init__.py
│   ├── test_triage.py        # Triage Agent isolated test
│   ├── test_resolution.py    # Resolution Agent isolated test
│   ├── test_qa.py            # QA Agent isolated test
│   ├── test_pipeline.py      # Full three-agent pipeline test
│   └── test_handler.py       # Lambda handler + event parsing test
├── scripts/
│   ├── package.sh            # Builds Lambda deployment zip
│   └── deploy.sh             # Uploads and deploys to AWS Lambda
├── tickets/                  # Mock JIRA output — JSON files written here
├── sample_data/              # Sample alert inputs for demo runs
├── graph.py                  # LangGraph graph definition + pipeline entry point
├── lambda_handler.py         # AWS Lambda handler — API Gateway + S3 events
├── config.py                 # Central config — all env vars loaded here
├── requirements.txt          # Full dependencies
└── .env.example              # Environment variable template
```

---

## How It Works

### 1. Triage Agent (Agent A)

Receives a raw issue string from an HTTP request or S3 file event. Classifies
it into a category (`infrastructure`, `auth`, `ui_bug`, `performance`,
`security`) and priority (`P1`–`P4`). Calls a custom LangChain tool to query
historical incident logs for similar past events, enriching the context before
classification. Returns structured JSON that all downstream agents read from
shared LangGraph state.

```json
{
  "category": "infrastructure",
  "priority": "P1",
  "summary": "Database connection timeout on production cluster",
  "log_summary": "3 similar incidents found. Avg resolution: 2.5h.",
  "suggested_owner": "platform-engineering",
  "recommended_action": "escalate_to_oncall",
  "confidence": "high"
}
```

### 2. Resolution Agent (Agent B)

Reads the triage output from shared state. Builds a semantic search query
from the category, log summary, and recommended action. Retrieves the top 2
most relevant internal runbooks from the FAISS vector index using cosine
similarity. Injects the retrieved runbooks directly into the LLM prompt —
this is the RAG (Retrieval Augmented Generation) pattern. The LLM generates
a structured JIRA ticket draft grounded in real internal documentation rather
than hallucinated content.

```json
{
  "ticket_title": "P1 Infrastructure: DB connection pool exhaustion on prod",
  "ticket_description": "Production database is experiencing connection timeouts affecting all users.",
  "resolution_steps": [
    "1. Check connection pool size: SHOW STATUS LIKE 'Threads_connected'",
    "2. Identify long-running queries",
    "3. Restart connection pool manager",
    "4. Monitor for 15 minutes post-fix"
  ],
  "affected_systems": ["production-cluster", "api-gateway"],
  "escalation_path": "escalate_to_oncall",
  "estimated_resolution_time": "2 hours",
  "runbooks_referenced": ["runbook-db-001"]
}
```

### 3. QA Agent (Agent C)

Reviews the ticket draft in two layers before it is filed.

**Layer 1 — Static pre-checks (free, deterministic):**
Fast regex patterns scan all ticket text for obvious violations — email
addresses, internal IP addresses, AWS access key patterns, and API credential
formats. Runs before any LLM call to catch cheap wins without spending tokens.

**Layer 2 — LLM compliance review:**
Sends the full ticket to GPT-4o with the complete privacy guidelines in
context. Handles nuanced violations that regex cannot catch — indirect PII,
third-party vendor name references, speculative business impact language,
and individual blame attribution.

If static checks find a violation but the LLM approves, the static check
always overrides — deterministic rules win over probabilistic ones for
compliance decisions.

```json
{
  "approved": true,
  "violations": [],
  "violation_count": 0,
  "risk_level": "low",
  "final_ticket": { "...sanitised ticket..." },
  "qa_notes": "No PII or policy violations detected. Safe to file."
}
```

### LangGraph State Flow

LangGraph maintains a single shared state dictionary across all agents.
Each node receives the full state and returns only the keys it updates.
No agent calls another directly — they communicate exclusively through state.

```
START  →  { issue: "raw alert string" }
           ↓ triage_node
           { issue, triage: { category, priority, log_summary, ... } }
           ↓ resolution_node
           { issue, triage, resolution: { ticket_title, steps, ... } }
           ↓ qa_node
           { issue, triage, resolution, qa: { approved, final_ticket, ... } }
          END
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- OpenAI API key — [platform.openai.com](https://platform.openai.com)
- AWS account — optional, for Lambda deployment only

### Installation

```bash
git clone https://github.com/yourusername/automated-jira-ticket-routing-agent
cd automated-jira-ticket-routing-agent

python -m venv .venv

# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```bash
OPENAI_API_KEY=sk-your-key-here
LOG_LEVEL=INFO
MOCK_MODE=True        # True = free local testing, False = real LLM calls
LLM_MODEL=gpt-4o
LLM_MAX_TOKENS=1000
```

### Running Locally

**Test individual agents in isolation:**

```bash
# Test Triage Agent only
python tests/test_triage.py

# Test Resolution Agent (chains from triage output)
python tests/test_resolution.py

# Test QA Agent (chains from resolution output)
python tests/test_qa.py
```

**Run the full three-agent pipeline:**

```bash
python tests/test_pipeline.py
```

**Test the Lambda handler locally with mock AWS events:**

```bash
python tests/test_handler.py
```

### Sample Pipeline Output

Running `test_pipeline.py` with `MOCK_MODE=True` produces:

```
TEST 1: Database infrastructure issue (P1)
Triage:     infrastructure | P1
Confidence: high
Action:     escalate_to_oncall
Ticket:     P1 Infrastructure: Issue detected on production
Steps:      5 steps
Approved:   True
Risk:       low
Notes:      No PII or policy violations detected. Safe to file.

TEST 2: Authentication issue (P2)
Triage:     authentication | P2
Action:     assign_to_auth_team
Ticket:     P2 Authentication: Issue detected on production
Approved:   True
Risk:       low

TEST 3: Frontend UI bug (P3)
Triage:     ui_bug | P3
Action:     assign_to_frontend_team
Ticket:     P3 Ui_Bug: Issue detected on production
Approved:   True
Risk:       low
```

Approved tickets are saved as JSON files in the `tickets/` folder.

---

## AWS Deployment Architecture

The system is architected for serverless deployment on AWS. The Lambda
function handles two trigger types in a single handler:

**API Gateway trigger** — for synchronous HTTP requests from dashboards,
Slack bots, or monitoring webhooks:

```bash
curl -X POST https://your-api-id.execute-api.us-east-1.amazonaws.com/triage \
  -H "Content-Type: application/json" \
  -d '{"issue": "Production DB timeout errors since 2am"}'
```

**S3 trigger** — for asynchronous batch processing where monitoring systems
drop alert files into a bucket:

```bash
echo "Production DB timeout errors" > alert.txt
aws s3 cp alert.txt s3://your-trigger-bucket/alerts/alert-001.txt
# Lambda fires automatically on upload
```

**AWS services used:**

| Service | Purpose |
|---|---|
| Lambda | Serverless function runtime — scales to zero |
| API Gateway | HTTP trigger endpoint with route configuration |
| S3 | File-based event trigger + deployment artifact storage |
| CloudWatch | Structured logging, request tracing, and alerting |
| IAM | Least-privilege execution role for Lambda |

**CloudWatch log format** — every line is structured for querying:

```
2026-05-28T13:00:10 | INFO | agents.triage_agent | Triage complete | category=infrastructure | priority=P1
2026-05-28T13:00:11 | INFO | agents.resolution_agent | Runbooks retrieved | count=2 | sources=['runbook-db-001']
2026-05-28T13:00:12 | INFO | agents.qa_agent | QA review complete | approved=True | violations=0 | risk_level=low
```

Query all P1 incidents in CloudWatch Logs Insights:

```
fields @timestamp, @message
| filter @message like "priority=P1"
| sort @timestamp desc
```

---

## Key Design Decisions

**Why LangGraph over a simple function chain?**

LangGraph models the pipeline as a stateful directed graph. Each node reads
from and writes to a shared state dict without calling other agents directly.
This means agents are fully decoupled — you can add, remove, or reorder them
without touching existing code. It also gives you full pipeline state
inspection at every step, which is critical for debugging LLM systems where
intermediate outputs matter as much as the final result.

**Why RAG over pure LLM generation?**

Without RAG the LLM generates fixes from training data alone. It may
hallucinate terminal commands, reference services that don't exist in your
infrastructure, or suggest procedures that were deprecated two years ago.
RAG retrieves your actual internal runbooks before generation — the output
is grounded in verified, organisation-specific knowledge that the LLM was
never trained on.

**Why a separate QA Agent?**

LLMs can inadvertently include sensitive information in generated text — a
log summary containing a customer email, a runbook step referencing an
internal IP, a description that names a vendor without legal approval. A
dedicated compliance gate ensures nothing reaches the ticket system without
review. This mirrors how real enterprise teams operate: no raw AI output
goes to customers or shared systems without a human or automated review step.

**Why two-layer QA review?**

Regex catches deterministic violations cheaply — no API call needed for an
obvious email address pattern. The LLM handles cases regex cannot reason
about — "the description implies this customer is John Smith from Finance"
is not a regex problem. Running regex first means most clean tickets never
reach the LLM at all, keeping costs low while maintaining thorough coverage.

**Why mock mode?**

`MOCK_MODE=True` lets the entire three-agent pipeline run end-to-end during
development without spending any OpenAI credits. Every agent returns
realistic hardcoded output that matches the real JSON schema exactly — so
state passing, error handling, and the LangGraph graph structure are all
validated for free. The only thing not tested in mock mode is the LLM prompt
quality, which is tested separately by flipping `MOCK_MODE=False`.

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key from platform.openai.com |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity: DEBUG, INFO, WARNING, ERROR |
| `MOCK_MODE` | No | `True` | Skip LLM calls for free local testing |
| `LLM_MODEL` | No | `gpt-4o` | OpenAI model to use across all agents |
| `LLM_MAX_TOKENS` | No | `1000` | Maximum tokens per LLM response |

---

## Future Improvements

- **Conditional routing** — skip Resolution Agent for P4 cosmetic issues,
  escalate P1 directly to PagerDuty without waiting for ticket generation
- **Persistent vector store** — replace in-memory FAISS with Amazon OpenSearch
  so the knowledge base survives Lambda cold starts
- **Real JIRA integration** — swap `ticket_writer.py` mock with the
  Atlassian JIRA REST API v3
- **Slack notification** — post approved P1 tickets to an on-call channel
  via Slack webhook after QA approval
- **Feedback loop** — store engineer resolutions back into the knowledge
  base so the RAG index improves over time

---

## Author

Darshil Shah |
[LinkedIn](https://www.linkedin.com/in/darshil-shah-38a780a8/) |
[Github](https://github.com/darshil1995/)
