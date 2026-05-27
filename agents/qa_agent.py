"""
QA Agent — Agent C in the JIRA Workflow Pipeline.

Responsibilities:
    1. Receive the structured resolution draft from Agent B
    2. Review it against a set of data privacy and security guidelines
    3. Flag any violations found (PII, credentials, internal IPs, etc.)
    4. Either approve the ticket as-is or return a sanitised version
    5. Produce the final output that gets written to the mock JIRA system

Why is a QA Agent necessary?
    LLMs can inadvertently include sensitive information in generated text.
    For example, the Resolution Agent might pull a log summary that contains
    a customer email address, an internal IP, or an API key fragment.
    Without a review gate, that sensitive data would flow directly into a
    JIRA ticket visible to the entire engineering team.

    The QA Agent acts as an automated compliance layer — it mirrors what
    a senior engineer would do before filing a ticket: check that no
    sensitive data is exposed, no policy is violated, and the content
    is appropriate for its audience.

Privacy guidelines enforced:
    - No customer PII (emails, phone numbers, full names)
    - No internal IP addresses or hostnames
    - No credentials, tokens, or API keys
    - No third-party vendor names without approval flag
    - No raw stack traces containing file paths

Typical input (from resolution state):
    {
        "ticket_title": "P1 Infrastructure: DB connection pool exhaustion",
        "ticket_description": "...",
        "resolution_steps": [...],
        "affected_systems": [...],
        "escalation_path": "escalate_to_oncall",
        "estimated_resolution_time": "2 hours",
        "runbooks_referenced": ["runbook-db-001"]
    }

Typical output:
    {
        "approved": true,
        "violations": [],
        "violation_count": 0,
        "risk_level": "low",
        "final_ticket": { ...sanitised resolution dict... },
        "qa_notes": "No issues found. Ticket approved for filing."
    }
"""

import json
import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI
import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Privacy Guidelines
# ---------------------------------------------------------------------------

# These are the rules the QA Agent enforces on every ticket.
# Defined as a module-level constant so they can be:
#   - Imported and referenced in unit tests
#   - Updated without changing agent logic
#   - Logged alongside any violation for audit trails
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


# ---------------------------------------------------------------------------
# Static Pre-checks (regex-based, runs before LLM)
# ---------------------------------------------------------------------------

# Compiled regex patterns for fast, deterministic PII detection.
# These run BEFORE the LLM call to catch obvious violations cheaply.
# The LLM handles nuanced cases that regex cannot catch.
_PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "internal_ip": re.compile(r"\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b"),
    "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "api_key_pattern": re.compile(r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*\S+"),
}


def _static_precheck(text: str) -> list[str]:
    """
    Runs fast regex-based checks on a text string before sending to the LLM.

    This is a cheap first pass — catches obvious, deterministic violations
    (email addresses, AWS key patterns, internal IPs) without spending
    API tokens. The LLM review handles everything else.

    Args:
        text: The combined text content of the ticket to check.

    Returns:
        List of violation description strings. Empty list means clean.
    """
    violations = []

    for pattern_name, pattern in _PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            violations.append(
                f"Static check: found {pattern_name} pattern — {matches[:3]}"
            )
            logger.warning(
                "PII pattern detected | type=%s | matches=%s",
                pattern_name, matches[:3]
            )

    return violations


# ---------------------------------------------------------------------------
# Mock Response
# ---------------------------------------------------------------------------

def _mock_qa_response(resolution: dict[str, Any]) -> dict[str, Any]:
    """
    Returns a hardcoded QA approval without calling OpenAI.

    In mock mode we assume the mock resolution output is clean
    (it contains no real PII since it's hardcoded test data).
    The static pre-checks still run even in mock mode — this ensures
    the regex layer is always exercised during local testing.

    Args:
        resolution: The resolution dict from Agent B.

    Returns:
        A mock QA result dict matching the real output schema.
    """
    logger.info("MOCK_MODE active — running static checks then returning mock QA response")

    # Combine all ticket text for static checking
    # Even in mock mode we run regex checks to exercise that code path
    combined_text = " ".join([
        resolution.get("ticket_title", ""),
        resolution.get("ticket_description", ""),
        " ".join(resolution.get("resolution_steps", [])),
    ])

    static_violations = _static_precheck(combined_text)

    if static_violations:
        logger.warning(
            "Static pre-check found violations in mock mode | count=%d",
            len(static_violations)
        )
        return {
            "approved": False,
            "violations": static_violations,
            "violation_count": len(static_violations),
            "risk_level": "high",
            "final_ticket": None,
            "qa_notes": (
                "Ticket REJECTED. Static pre-check found potential PII or "
                "credential exposure. Review violations before filing."
            ),
        }

    return {
        "approved": True,
        "violations": [],
        "violation_count": 0,
        "risk_level": "low",
        "final_ticket": resolution,
        "qa_notes": (
            "Ticket approved. No PII, credential, or policy violations detected. "
            "Safe to file in JIRA."
        ),
    }


# ---------------------------------------------------------------------------
# QA Agent Class
# ---------------------------------------------------------------------------

class QAAgent:
    """
    Reviews resolution drafts for data privacy and security compliance.

    Operates in two layers:
        Layer 1 — Static pre-checks: fast regex patterns that catch
                  obvious violations (emails, IPs, AWS keys) before
                  any LLM call is made. Free and deterministic.

        Layer 2 — LLM review: sends the ticket to the LLM with the
                  full privacy guidelines. Catches nuanced violations
                  that regex cannot — e.g. indirect PII, vendor name
                  references, speculative business impact language.

    The two-layer approach keeps costs low (most clean tickets are
    approved by regex alone) while ensuring thorough review of anything
    that looks potentially problematic.

    Attributes:
        llm (ChatOpenAI): The LLM used for nuanced compliance review.
        mock_mode (bool): When True, skips LLM and uses static checks only.
    """

    # Risk levels the QA agent can assign — ordered low to high
    RISK_LEVELS = ["low", "medium", "high", "critical"]

    def __init__(self):
        """
        Initialises the QA Agent.

        Uses a lower temperature than the other agents because compliance
        review requires consistent, deterministic judgement — not creativity.
        """
        logger.info(
            "Initialising QAAgent | model=%s | mock=%s",
            config.LLM_MODEL, config.MOCK_MODE
        )

        self.mock_mode = config.MOCK_MODE

        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=0,      # Strictly deterministic — compliance decisions
                                # must be consistent across identical inputs.
                                # Any randomness here is a liability.
            api_key=config.OPENAI_API_KEY,
        )

    def _build_prompt(self, resolution: dict) -> str:
        """
        Constructs the QA review prompt.

        The full privacy guidelines are injected directly into the prompt
        so the LLM has the complete compliance ruleset in context.
        The ticket content follows so the LLM can review it against
        each rule explicitly.

        Args:
            resolution: The resolution dict from Agent B.

        Returns:
            Fully formatted QA review prompt string.
        """
        # Serialise the full ticket as formatted JSON for the LLM to read
        ticket_text = json.dumps(resolution, indent=2)

        return f"""
        You are a senior security and compliance engineer reviewing a JIRA ticket
        draft before it is filed. Your job is to ensure it complies with all
        data privacy and security guidelines.

        GUIDELINES TO ENFORCE:
        {PRIVACY_GUIDELINES}

        TICKET DRAFT TO REVIEW:
        {ticket_text}

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
        - Always return final_ticket — if approved, return original ticket unchanged;
          if violations found, return sanitised version with violations redacted
        - Be specific in violation descriptions — name the field and the value
        - Return ONLY the JSON. No explanation, no markdown, no code fences.
        """

    def run(self, resolution: dict[str, Any]) -> dict[str, Any]:
        """
        Main entry point for the QA Agent.

        Orchestrates the two-layer review:
            1. Short-circuit to mock if MOCK_MODE is True
            2. Run static regex pre-checks on all ticket text
            3. If static checks pass, send to LLM for deep review
            4. Parse and return the structured compliance result

        The static pre-check result is included in the LLM prompt
        so the LLM is aware of any patterns already flagged — it can
        then focus its review on the nuanced cases.

        Args:
            resolution: Structured resolution dict from Agent B containing
                        ticket_title, ticket_description, resolution_steps,
                        affected_systems, escalation_path, etc.

        Returns:
            Dict with approved, violations, violation_count, risk_level,
            final_ticket, and qa_notes.

        Raises:
            ValueError: If the LLM returns malformed JSON.
        """
        logger.info(
            "QAAgent.run() called | ticket_title='%s'",
            resolution.get("ticket_title", "")[:60]
        )

        # --- MOCK MODE ---
        if self.mock_mode:
            return _mock_qa_response(resolution)

        # --- PRODUCTION MODE ---

        # Layer 1: Static pre-checks — fast, free, deterministic
        combined_text = " ".join([
            resolution.get("ticket_title", ""),
            resolution.get("ticket_description", ""),
            " ".join(resolution.get("resolution_steps", [])),
            " ".join(resolution.get("affected_systems", [])),
        ])

        static_violations = _static_precheck(combined_text)
        logger.info(
            "Static pre-check complete | violations_found=%d",
            len(static_violations)
        )

        # Layer 2: LLM deep review
        # We always run this regardless of static results —
        # static checks only catch obvious patterns; LLM catches nuance
        prompt = self._build_prompt(resolution)

        logger.info("Calling LLM for QA review | model=%s", config.LLM_MODEL)
        response = self.llm.invoke(prompt)
        raw_content = response.content.strip()
        logger.debug("Raw LLM QA response | content='%s'", raw_content[:200])

        # Parse JSON response
        try:
            result = json.loads(raw_content)
        except json.JSONDecodeError as e:
            logger.error(
                "QA LLM returned invalid JSON | error=%s | raw='%s'",
                str(e), raw_content[:300]
            )
            raise ValueError(
                f"QA Agent received non-JSON response: {raw_content[:200]}"
            ) from e

        # Merge any static violations the LLM may have missed
        # Static violations are deterministic — if regex found something,
        # it must be in the final violations list regardless of LLM opinion
        if static_violations:
            existing = result.get("violations", [])
            merged = list(set(existing + static_violations))
            result["violations"] = merged
            result["violation_count"] = len(merged)

            # Downgrade approval if static checks found something
            if result.get("approved") is True:
                logger.warning(
                    "LLM approved ticket but static checks found violations — overriding to rejected"
                )
                result["approved"] = False
                result["risk_level"] = "high"

        logger.info(
            "QA review complete | approved=%s | violations=%d | risk_level=%s",
            result.get("approved"),
            result.get("violation_count", 0),
            result.get("risk_level")
        )

        return result


# ---------------------------------------------------------------------------
# Module-level singleton + LangGraph node function
# ---------------------------------------------------------------------------

# Single instance reused across Lambda warm starts
qa_agent = QAAgent()


def run_qa(state: dict) -> dict:
    """
    LangGraph node function for the QA Agent.

    Reads the resolution output from shared graph state,
    runs the two-layer compliance review, and writes the
    final approved (or rejected) ticket back into state.

    This is the last node in the graph. Its output is what
    gets written to the mock JIRA system.

    Args:
        state: LangGraph shared state. Must contain "resolution" key.

    Returns:
        Dict fragment with "qa" key to merge into graph state.
    """
    logger.info("LangGraph node: qa | state_keys=%s", list(state.keys()))

    resolution = state.get("resolution", {})

    if not resolution:
        logger.warning("QA node received empty resolution state")

    qa_result = qa_agent.run(resolution)
    return {"qa": qa_result}