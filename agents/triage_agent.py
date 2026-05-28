"""
Triage Agent — Agent A in the JIRA Workflow Pipeline.

Responsibilities:
    1. Receive a raw incoming issue (system alert, user complaint, etc.)
    2. Classify it into a category and priority level
    3. Call the historical log tool to find similar past incidents
    4. Return a structured triage result for the Resolution Agent to consume

This agent is intentionally narrow in scope. It does not attempt to fix
anything — it only understands and classifies the problem. This separation
means you can retrain or reprompt this agent independently without
affecting downstream agents.

Dependencies:
    - langchain_openai: LLM wrapper for OpenAI's GPT models
    - langchain.tools: decorator to expose Python functions as LLM-callable tools
    - config: central config module for model name, mock mode, etc.
"""

import json
import logging
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from config import OPENAI_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, MOCK_MODE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

@tool
def check_historical_logs(issue_description: str) -> str:
    """
    Queries historical incident logs to find similar past issues.

    In production this would connect to one of:
        - CloudWatch Logs Insights (AWS native)
        - An internal Elasticsearch / OpenSearch cluster
        - A JIRA historical ticket database

    For the portfolio version we return a realistic mock response
    so the pipeline runs end-to-end without external dependencies.

    Args:
        issue_description (str): The raw issue text to search logs for.

    Returns:
        str: A summary of similar past incidents including resolution time
             and suggested owner team.
    """
    logger.debug("check_historical_logs called | issue=%s", issue_description[:80])

    # --- Mock response (MOCK_MODE=True in .env during development) ---
    # Replace this block with a real API/DB call when going to production
    mock_log_data = {
        "similar_incidents_found": 3,
        "most_recent": "2024-11-14",
        "avg_resolution_time_hours": 2.5,
        "suggested_owner_team": "platform-engineering",
        "common_root_cause": "Deployment config drift after release",
        "resolution_hint": "Rollback to previous config version and restart service"
    }

    logger.info(
        "Historical log check complete | incidents_found=%d",
        mock_log_data["similar_incidents_found"]
    )

    return json.dumps(mock_log_data, indent=2)


# ---------------------------------------------------------------------------
# Agent Class
# ---------------------------------------------------------------------------

class TriageAgent:
    """
    Classifies incoming issues and enriches them with historical context.

    This agent wraps the LLM interaction for the triage step. It is
    instantiated once at graph build time and reused across invocations,
    which avoids recreating the LLM client on every Lambda call (cold
    start optimisation).

    Attributes:
        llm (ChatOpenAI): The LLM client bound with the triage tool.
        tools (list): List of LangChain tools available to this agent.
        mock_mode (bool): When True, skips LLM call and returns mock output.
    """

    # Priority levels the agent can assign — defined as a class constant
    # so they can be referenced elsewhere without magic strings
    PRIORITY_LEVELS = ["P1", "P2", "P3", "P4"]

    # Valid categories the agent can classify issues into
    CATEGORIES = [
        "infrastructure",
        "auth",
        "database",
        "ui-bug",
        "performance",
        "security",
        "other"
    ]

    def __init__(self):
        """
        Initialises the Triage Agent.

        Sets up the LLM client and binds the historical log tool to it.
        The tool binding tells the LLM it can call check_historical_logs
        during its reasoning process.
        """
        logger.info("Initialising TriageAgent | model=%s | mock=%s", LLM_MODEL, MOCK_MODE)

        self.mock_mode = MOCK_MODE
        self.tools = [check_historical_logs]

        # Bind tools to the LLM so it knows they exist and can call them
        # tool_choice="auto" lets the LLM decide when to use the tool
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=0,          # Zero temperature = deterministic output
                                    # Critical for classification tasks where
                                    # consistency matters more than creativity
            api_key=OPENAI_API_KEY
        ).bind_tools(self.tools)

    def _build_prompt(self, issue: str) -> str:
        """
        Constructs the triage prompt sent to the LLM.

        Keeping the prompt in its own method makes it easy to:
            - Unit test the prompt structure independently
            - A/B test different prompt versions
            - Log the exact prompt that produced a given output

        Args:
            issue (str): The raw incoming issue text.

        Returns:
            str: The fully formatted prompt string.
        """
        return f"""
        You are a senior site reliability engineer performing issue triage.

        Your job is to:
        1. Classify the issue into one of these categories: {self.CATEGORIES}
        2. Assign a priority: {self.PRIORITY_LEVELS}
           - P1 = production down, immediate action required
           - P2 = major feature broken, significant user impact
           - P3 = minor bug, workaround exists
           - P4 = cosmetic issue, no functional impact
        3. Use the check_historical_logs tool to find similar past incidents
        4. Return ONLY a valid JSON object with this exact structure:

        {{
            "category": "<category>",
            "priority": "<P1|P2|P3|P4>",
            "summary": "<one sentence description of the issue>",
            "log_summary": "<what the historical logs revealed>",
            "suggested_owner": "<team name>",
            "recommended_action": "<escalate_to_oncall | assign_to_auth_team | assign_to_frontend_team | assign_to_on_call>",
            "confidence": "<high|medium|low>"
        }}

        Issue to triage:
        {issue}
        """

    def _mock_response(self, issue: str) -> dict:
        """
        Returns a realistic hardcoded triage result for local development.

        Used when MOCK_MODE=True so the pipeline can be tested end-to-end
        without making any OpenAI API calls or spending credits.

        Args:
            issue (str): The raw issue (used only for logging here).

        Returns:
            dict: A mock triage result matching the real output structure.
        """
        logger.info("TriageAgent running in MOCK_MODE — skipping LLM call")
        return {
            "category": "infrastructure",
            "priority": "P1",
            "summary": "Database connection timeout detected on production cluster",
            "log_summary": "3 similar incidents found. Avg resolution: 2.5h. Common cause: config drift after deployment.",
            "suggested_owner": "platform-engineering",
            "recommended_action": "escalate_to_oncall",  # ← ADD THIS
            "confidence": "high"
        }

    def run(self, issue: str) -> dict:
        """
        Main entry point for the Triage Agent.

        Orchestrates the full triage flow:
            1. Check mock mode
            2. Build and send prompt to LLM
            3. Parse and validate the structured JSON response
            4. Return result for the LangGraph state

        Args:
            issue (str): Raw incoming issue text from the event trigger.

        Returns:
            dict: Structured triage result with category, priority, summary,
                  log context, owner suggestion, and confidence level.

        Raises:
            ValueError: If the LLM returns a response that cannot be parsed
                        as valid JSON — logged and re-raised for graph to handle.
        """
        logger.info("TriageAgent.run() called | issue_preview='%s'", issue[:80])

        # --- Short-circuit for development / testing ---
        if self.mock_mode:
            return self._mock_response(issue)

        # --- Build prompt and invoke LLM ---
        prompt = self._build_prompt(issue)
        logger.debug("Sending prompt to LLM | length=%d chars", len(prompt))

        try:
            response = self.llm.invoke(prompt)
            logger.debug("LLM raw response received | content=%s", response.content[:200])

            # Strip markdown code fences if the LLM wraps JSON in ```json ... ```
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            result = json.loads(content.strip())
            logger.info(
                "Triage complete | category=%s | priority=%s | confidence=%s",
                result.get("category"),
                result.get("priority"),
                result.get("confidence")
            )
            return result

        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON | error=%s", str(e))
            raise ValueError(f"TriageAgent returned invalid JSON: {e}") from e


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

# Instantiated once at import time so LangGraph reuses the same instance
# across the lifetime of a Lambda execution context (warm start optimisation)
triage_agent = TriageAgent()


def run_triage(state: dict) -> dict:
    """
    LangGraph node function for the Triage Agent.

    LangGraph calls node functions with the full graph state dict and
    expects back a dict of keys to update in that state. This thin wrapper
    adapts the TriageAgent class to that interface.

    Args:
        state (dict): The current LangGraph state. Must contain "issue" key.

    Returns:
        dict: Updated state fragment containing the "triage" key.
    """
    logger.info("LangGraph node: triage | state_keys=%s", list(state.keys()))
    issue = state.get("issue", "")

    if not issue:
        logger.warning("Triage node received empty issue string")

    triage_result = triage_agent.run(issue)
    return {"triage": triage_result}