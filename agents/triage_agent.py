import json
import logging
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from config import OPENAI_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, MOCK_MODE

logger = logging.getLogger(__name__)


@tool
def check_historical_logs(issue_description: str) -> str:
    """
    Queries historical incident logs for similar past issues.

    In production, replace this with a real CloudWatch Logs Insights,
    OpenSearch, or internal DB query.
    """
    logger.debug("check_historical_logs called | issue='%s'", issue_description[:80])

    mock_log_data = {
        "similar_incidents_found": 3,
        "most_recent": "2024-11-14",
        "avg_resolution_time_hours": 2.5,
        "suggested_owner_team": "platform-engineering",
        "common_root_cause": "Deployment config drift after release",
        "resolution_hint": "Rollback to previous config version and restart service",
    }

    logger.info("Log check complete | incidents_found=%d", mock_log_data["similar_incidents_found"])
    return json.dumps(mock_log_data, indent=2)


class TriageAgent:
    """Classifies incoming issues and enriches them with historical log context."""

    PRIORITY_LEVELS = ["P1", "P2", "P3", "P4"]
    CATEGORIES = ["infrastructure", "auth", "database", "ui-bug", "performance", "security", "other"]

    def __init__(self):
        logger.info("Initialising TriageAgent | model=%s | mock=%s", LLM_MODEL, MOCK_MODE)
        self.mock_mode = MOCK_MODE
        self.tools = [check_historical_logs]

        # temperature=0 for deterministic classification — consistency matters more than creativity here
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=0,
            api_key=OPENAI_API_KEY,
        ).bind_tools(self.tools)

    def _build_prompt(self, issue: str) -> str:
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
        logger.info("TriageAgent running in MOCK_MODE — skipping LLM call")
        return {
            "category": "infrastructure",
            "priority": "P1",
            "summary": "Database connection timeout detected on production cluster",
            "log_summary": "3 similar incidents found. Avg resolution: 2.5h. Common cause: config drift after deployment.",
            "suggested_owner": "platform-engineering",
            "recommended_action": "escalate_to_oncall",
            "confidence": "high",
        }

    def run(self, issue: str) -> dict:
        """
        Classifies the issue and returns a structured triage result.

        Raises:
            ValueError: If the LLM returns malformed JSON.
        """
        logger.info("TriageAgent.run() | issue='%s'", issue[:80])

        if self.mock_mode:
            return self._mock_response(issue)

        prompt = self._build_prompt(issue)

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # Strip markdown code fences if the LLM wraps the JSON response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            result = json.loads(content.strip())
            logger.info(
                "Triage complete | category=%s | priority=%s | confidence=%s",
                result.get("category"), result.get("priority"), result.get("confidence")
            )
            return result

        except json.JSONDecodeError as e:
            logger.error("Triage LLM returned invalid JSON | error=%s", str(e))
            raise ValueError(f"TriageAgent returned invalid JSON: {e}") from e


# Singleton — reused across Lambda warm starts to avoid re-initialising the LLM client
triage_agent = TriageAgent()


def run_triage(state: dict) -> dict:
    """LangGraph node — reads 'issue' from state, writes 'triage' back."""
    logger.info("Node: triage | state_keys=%s", list(state.keys()))
    issue = state.get("issue", "")
    if not issue:
        logger.warning("Triage node received empty issue string")
    return {"triage": triage_agent.run(issue)}