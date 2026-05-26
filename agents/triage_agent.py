# agents/triage_agent.py

"""
Triage Agent — Agent A in the JIRA Workflow Pipeline.

Responsibilities:
    1. Receive a raw incoming issue (system alert, user report, monitoring event)
    2. Classify it into category and priority level
    3. Call the historical log tool to find similar past incidents
    4. Return a structured triage result for the Resolution Agent to consume

This agent is intentionally narrow in scope. It does NOT attempt to fix
anything — it only understands and classifies the problem. Keeping
classification separate from resolution means each can be improved,
tested, and debugged independently.

Typical input:
    "Production database is throwing connection timeout errors since 2am"

Typical output:
    {
        "category": "infrastructure",
        "priority": "P1",
        "log_summary": "3 similar incidents found. Last occurred 2024-01-15. Avg resolution: 2h",
        "confidence": "high",
        "recommended_action": "escalate_to_oncall"
    }
"""

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain.tools import tool

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def check_historical_logs(issue_description: str) -> str:
    """
    Queries the internal incident log store for similar past issues.

    In production, this would call CloudWatch Logs Insights, an internal
    database, or a search API. For now, it returns realistic mock data
    so the pipeline runs end-to-end without external dependencies.

    Args:
        issue_description: The raw issue text to search logs for.

    Returns:
        A plain-text summary of similar historical incidents.
    """
    logger.debug("check_historical_logs called | query='%s'", issue_description[:80])

    # --- MOCK RESPONSES ---
    # Keyed by keyword so different test inputs return different results.
    # Replace this dict lookup with a real API call when going to production.
    mock_log_db = {
        "database": (
            "Found 3 similar incidents. "
            "Last occurred: 2024-01-15. "
            "Root cause: connection pool exhaustion. "
            "Resolution: restart connection pool manager. "
            "Avg resolution time: 1.5 hours."
        ),
        "auth": (
            "Found 1 similar incident. "
            "Last occurred: 2024-02-03. "
            "Root cause: expired JWT signing key. "
            "Resolution: rotate signing key and redeploy auth service. "
            "Avg resolution time: 45 minutes."
        ),
        "timeout": (
            "Found 5 similar incidents. "
            "Last occurred: 2024-03-01. "
            "Root cause: upstream service latency spike. "
            "Resolution: circuit breaker triggered, traffic rerouted. "
            "Avg resolution time: 30 minutes."
        ),
    }

    # Match the first keyword found in the issue description
    issue_lower = issue_description.lower()
    for keyword, summary in mock_log_db.items():
        if keyword in issue_lower:
            logger.info("Historical log match found | keyword='%s'", keyword)
            return summary

    # Default response when no match is found
    logger.info("No historical log match found for issue")
    return (
        "No similar incidents found in historical logs. "
        "This appears to be a new type of issue."
    )


# ---------------------------------------------------------------------------
# Mock LLM Response
# ---------------------------------------------------------------------------

def _mock_triage_response(issue: str) -> dict[str, Any]:
    """
    Returns a hardcoded realistic triage result without calling OpenAI.

    Used when MOCK_MODE=True in .env. This lets you test the entire
    pipeline structure and agent state passing for free — no API credits
    consumed. The structure mirrors exactly what the real LLM returns
    so the rest of the pipeline behaves identically in both modes.

    Args:
        issue: The raw incoming issue string.

    Returns:
        A dict matching the real triage output schema.
    """
    logger.info("MOCK_MODE active — returning mock triage response")

    issue_lower = issue.lower()

    # Return different mock outputs based on keywords
    # so your sample runs look realistic and varied
    if "database" in issue_lower or "db" in issue_lower:
        return {
            "category": "infrastructure",
            "priority": "P1",
            "log_summary": (
                "Found 3 similar incidents. Last: 2024-01-15. "
                "Root cause: connection pool exhaustion. Avg resolution: 1.5h."
            ),
            "confidence": "high",
            "recommended_action": "escalate_to_oncall",
        }
    elif "auth" in issue_lower or "login" in issue_lower or "password" in issue_lower:
        return {
            "category": "authentication",
            "priority": "P2",
            "log_summary": (
                "Found 1 similar incident. Last: 2024-02-03. "
                "Root cause: expired JWT key. Avg resolution: 45min."
            ),
            "confidence": "high",
            "recommended_action": "assign_to_auth_team",
        }
    elif "ui" in issue_lower or "button" in issue_lower or "frontend" in issue_lower:
        return {
            "category": "ui_bug",
            "priority": "P3",
            "log_summary": "No similar incidents found. New issue type.",
            "confidence": "medium",
            "recommended_action": "assign_to_frontend_team",
        }
    else:
        return {
            "category": "general",
            "priority": "P2",
            "log_summary": "No similar incidents found.",
            "confidence": "low",
            "recommended_action": "assign_to_on_call",
        }


# ---------------------------------------------------------------------------
# Main Agent Function
# ---------------------------------------------------------------------------

def run_triage(issue: str) -> dict[str, Any]:
    """
    Entry point for the Triage Agent.

    Orchestrates the full triage flow:
        1. Optionally short-circuits to mock mode for local testing
        2. Calls the historical log tool to enrich context
        3. Sends enriched context to the LLM for classification
        4. Parses and validates the structured JSON response
        5. Returns a clean dict for LangGraph to store in shared state

    Args:
        issue: Raw incoming issue text. Can be a monitoring alert,
               a user-reported bug, or a Slack message.

    Returns:
        A dict with keys: category, priority, log_summary,
        confidence, recommended_action.

    Raises:
        ValueError: If the LLM returns a response that cannot be
                    parsed as valid JSON. Caught by LangGraph and
                    logged to CloudWatch.
    """
    logger.info("Triage Agent started | issue='%s'", issue[:100])

    # --- MOCK MODE ---
    # Short-circuit here if MOCK_MODE is True.
    # The rest of this function is only reached in production mode.
    if config.MOCK_MODE:
        result = _mock_triage_response(issue)
        logger.info("Triage complete (mock) | category=%s | priority=%s",
                    result["category"], result["priority"])
        return result

    # --- PRODUCTION MODE ---

    # Step 1: Call the log tool to find historical context
    # We do this before the LLM call so the LLM has richer input
    log_summary = check_historical_logs.invoke(issue)
    logger.debug("Log tool result | summary='%s'", log_summary[:120])

    # Step 2: Build the classification prompt
    # The prompt is explicit about output format so we can reliably parse it
    prompt = f"""
    You are a senior site reliability engineer triaging an incoming issue.

    Incoming issue:
    {issue}

    Historical log context:
    {log_summary}

    Classify this issue and return ONLY a valid JSON object with these exact keys:
    {{
        "category": "infrastructure | authentication | ui_bug | performance | security | general",
        "priority": "P1 | P2 | P3 | P4",
        "log_summary": "one sentence summary of relevant historical context",
        "confidence": "high | medium | low",
        "recommended_action": "escalate_to_oncall | assign_to_auth_team | assign_to_frontend_team | assign_to_on_call"
    }}

    Priority guide:
    - P1: Production down or data loss risk
    - P2: Major feature broken, workaround exists
    - P3: Minor bug, low user impact
    - P4: Cosmetic or documentation issue

    Return ONLY the JSON. No explanation, no markdown, no code fences.
    """

    # Step 3: Call the LLM
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        temperature=0,       # Zero temperature = deterministic, consistent output
        api_key=config.OPENAI_API_KEY,
    )

    logger.info("Calling LLM for triage classification | model=%s", config.LLM_MODEL)
    response = llm.invoke(prompt)
    raw_content = response.content.strip()
    logger.debug("Raw LLM response | content='%s'", raw_content[:200])

    # Step 4: Parse and validate the JSON response
    # If the LLM returns malformed JSON, raise clearly so LangGraph
    # can catch it and log it to CloudWatch with full context
    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError as e:
        logger.error(
            "Triage LLM returned invalid JSON | error=%s | raw='%s'",
            str(e), raw_content[:300]
        )
        raise ValueError(
            f"Triage Agent received non-JSON response from LLM: {raw_content[:200]}"
        ) from e

    logger.info(
        "Triage complete | category=%s | priority=%s | confidence=%s",
        result.get("category"), result.get("priority"), result.get("confidence")
    )

    return result