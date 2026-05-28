import os
import logging
from dotenv import load_dotenv

# No-op in Lambda — env vars are set via AWS console instead of .env
load_dotenv()


def _require(key: str) -> str:
    """
    Raises at startup if a required variable is missing, rather than
    failing silently mid-pipeline when the value is first used.
    """
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Check your .env file or Lambda environment config."
        )
    return value


OPENAI_API_KEY: str = _require("OPENAI_API_KEY")
AWS_REGION: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
MOCK_MODE: bool = os.getenv("MOCK_MODE", "True").lower() == "true"
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))


def configure_logging() -> logging.Logger:
    """
    Configures structured logging with a pipe-delimited format that
    CloudWatch Logs Insights can filter and query by field.
    """
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    return logging.getLogger("jira_agent")


logger = configure_logging()
logger.info("Configuration loaded | MOCK_MODE=%s | LLM_MODEL=%s", MOCK_MODE, LLM_MODEL)