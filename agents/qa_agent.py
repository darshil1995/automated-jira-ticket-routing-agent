import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI
import config

logger = logging.getLogger(__name__)

PRIVACY_GUIDELINES = """
DATA PRIVACY AND SECURITY GUIDELINES FOR JIRA TICKETS:

1. NO CUSTOMER PII
   - Do not include customer email addresses, phone numbers, or full names
   - Do not include account IDs that map directly to individual customers
   - Replace any found PII with placeholder: [REDACTED-PII]

2. NO CREDENTIALS OR SECRETS
   - Do not include API keys, tokens, passwords, or secret values
   - Do not include AWS access key IDs or secret keys
   - Replace any found credentials with: [REDACTED-CREDENTIAL]

3. NO INTERNAL NETWORK DETAILS
   - Do not include internal IP addresses (e.g. 10.x.x.x, 192.168.x.x)
   - Do not include internal hostnames or service discovery URLs
   - Replace with: [REDACTED-INTERNAL-NETWORK]

4. NO RAW STACK TRACES
   - Do not include full file system paths from stack traces
   - Summarise the error type only, not the full trace
   - Replace with: [REDACTED-STACK-TRACE]

5. THIRD-PARTY VENDOR NAMES
   - Flag (do not redact) any third-party vendor or tool names
   - These require approval from the legal team before filing
   - Mark with: [NEEDS-LEGAL-REVIEW: <vendor name>]

6. APPROPRIATE AUDIENCE
   - Ticket must be written for a technical engineering audience
   - Must not contain speculation about business impact or financial loss
   - Must not assign blame to specific individuals by name
"""

# Compiled once at import time — regex compilation is expensive at scale
_PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "internal_ip": re.compile(r"\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b"),
    "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "api_key_pattern": re.compile(r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*\S+"),
}


def _run_static_pii_checks(text: str) -> list[str]:
    """
    Runs deterministic regex checks before the LLM review.

    Static checks are cheap and always override LLM approval —
    a regex match is a hard violation regardless of LLM judgement.
    """
    violations = []
    for pattern_name, pattern in _PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            violations.append(f"Static check: found {pattern_name} — {matches[:3]}")
            logger.warning("PII detected | type=%s | matches=%s", pattern_name, matches[:3])
    return violations


def _mock_qa_response(resolution: dict[str, Any]) -> dict[str, Any]:
    """
    Returns a mock QA result without calling OpenAI.

    Static checks still run in mock mode so the regex layer
    is exercised on every local test run.
    """
    logger.info("QAAgent running in MOCK_MODE — skipping LLM call")

    combined_text = " ".join([
        resolution.get("ticket_title", ""),
        resolution.get("ticket_description", ""),
        " ".join(resolution.get("resolution_steps", [])),
    ])

    static_violations = _run_static_pii_checks(combined_text)

    if static_violations:
        logger.warning("Static violations found in mock mode | count=%d", len(static_violations))
        return {
            "approved": False,
            "violations": static_violations,
            "violation_count": len(static_violations),
            "risk_level": "high",
            "final_ticket": None,
            "qa_notes": "Ticket REJECTED. Static pre-check found PII or credential exposure.",
        }

    return {
        "approved": True,
        "violations": [],
        "violation_count": 0,
        "risk_level": "low",
        "final_ticket": resolution,
        "qa_notes": "Ticket approved. No PII, credential, or policy violations detected. Safe to file.",
    }


class QAAgent:
    """
    Two-layer compliance reviewer: regex pre-check followed by LLM review.

    Static checks catch obvious violations cheaply. The LLM handles nuanced
    cases regex cannot reason about (indirect PII, speculative language, etc.).
    Static violations always override LLM approval — deterministic wins.
    """

    RISK_LEVELS = ["low", "medium", "high", "critical"]

    def __init__(self):
        logger.info("Initialising QAAgent | model=%s | mock=%s", config.LLM_MODEL, config.MOCK_MODE)
        self.mock_mode = config.MOCK_MODE

        # temperature=0 — compliance decisions must be consistent across identical inputs
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=0,
            api_key=config.OPENAI_API_KEY,
        )

    def _build_prompt(self, resolution: dict) -> str:
        return f"""
        You are a senior security and compliance engineer reviewing a JIRA ticket
        draft before it is filed. Your job is to ensure it complies with all
        data privacy and security guidelines.

        GUIDELINES TO ENFORCE:
        {PRIVACY_GUIDELINES}

        TICKET DRAFT TO REVIEW:
        {json.dumps(resolution, indent=2)}

        Review every field of the ticket against every guideline above.

        Return ONLY a valid JSON object with these exact keys:
        {{
            "approved": true or false,
            "violations": ["description of violation 1", "description of violation 2"],
            "violation_count": <integer>,
            "risk_level": "low | medium | high | critical",
            "final_ticket": {{...sanitised ticket with violations redacted...}},
            "qa_notes": "one paragraph summary of findings and actions taken"
        }}

        Rules for your response:
        - If no violations found: approved=true, violations=[], risk_level="low"
        - If violations found: approved=false, list each violation clearly
        - Always return final_ticket — original if approved, sanitised if rejected
        - Return ONLY the JSON. No explanation, no markdown, no code fences.
        """

    def run(self, resolution: dict[str, Any]) -> dict[str, Any]:
        """
        Runs two-layer compliance review and returns the final approval decision.

        Raises:
            ValueError: If the LLM returns malformed JSON.
        """
        logger.info("QAAgent.run() | title='%s'", resolution.get("ticket_title", "")[:60])

        if self.mock_mode:
            return _mock_qa_response(resolution)

        combined_text = " ".join([
            resolution.get("ticket_title", ""),
            resolution.get("ticket_description", ""),
            " ".join(resolution.get("resolution_steps", [])),
            " ".join(resolution.get("affected_systems", [])),
        ])

        static_violations = _run_static_pii_checks(combined_text)
        logger.info("Static pre-check complete | violations=%d", len(static_violations))

        logger.info("Calling LLM for QA review | model=%s", config.LLM_MODEL)
        response = self.llm.invoke(self._build_prompt(resolution))
        raw_content = response.content.strip()

        try:
            result = json.loads(raw_content)
        except json.JSONDecodeError as e:
            logger.error("QA LLM returned invalid JSON | error=%s | raw='%s'", str(e), raw_content[:300])
            raise ValueError(f"QAAgent received non-JSON response: {raw_content[:200]}") from e

        # Static violations are deterministic — merge them and override LLM approval if needed
        if static_violations:
            merged = list(set(result.get("violations", []) + static_violations))
            result["violations"] = merged
            result["violation_count"] = len(merged)
            if result.get("approved") is True:
                logger.warning("LLM approved but static checks found violations — overriding to rejected")
                result["approved"] = False
                result["risk_level"] = "high"

        logger.info(
            "QA complete | approved=%s | violations=%d | risk_level=%s",
            result.get("approved"), result.get("violation_count", 0), result.get("risk_level")
        )
        return result


# Singleton — reused across Lambda warm starts to avoid re-initialising the LLM client
qa_agent = QAAgent()


def run_qa(state: dict) -> dict:
    """LangGraph node — reads 'resolution' from state, writes 'qa' back."""
    logger.info("Node: qa | state_keys=%s", list(state.keys()))
    resolution = state.get("resolution", {})
    if not resolution:
        logger.warning("QA node received empty resolution state")
    return {"qa": qa_agent.run(resolution)}