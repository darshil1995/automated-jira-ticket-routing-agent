""""
Central configuration module for the JIRA Agent pipeline.

All environment variables are read here and exposed as typed constants.
Every other module imports from this file — never directly from os.environ.
This makes it trivial to see every config value the app depends on,
and easy to validate them at startup rather than failing mid-pipeline.
"""

import os
import logging
from dotenv import load_dotenv

# Load variables from .env into os.environ
# This is a no-op in production (Lambda) where env vars are set via AWS console
load_dotenv()


def _require(key: str) -> str:
    """
    Fetch a required environment variable.
    Raises a clear error at startup if it's missing,
    rather than a cryptic KeyError deep inside a pipeline run.
    """
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Check your .env file or Lambda environment config."
        )
    return value


# --- LLM ---
OPENAI_API_KEY: str = _require("OPENAI_API_KEY")

# --- AWS ---
AWS_REGION: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# --- Logging ---
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# --- App Behaviour ---
# When True, agents return mock responses instead of calling OpenAI.
# Useful for testing the pipeline structure without spending API credits.
MOCK_MODE: bool = os.getenv("MOCK_MODE", "True").lower() == "true"

# LLM model to use across all agents.
# Centralised here so you change the model once, not in every agent file.
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")

# Maximum tokens the LLM can return per call.
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))


def configure_logging() -> logging.Logger:
    """
    Sets up structured logging for the entire application.

    Returns a root logger configured with:
    - Log level from environment (DEBUG for local dev, INFO for production)
    - A consistent format that CloudWatch can parse and filter on
    - ISO timestamp for correlation across distributed Lambda invocations

    Call this once at application entry point (lambda_handler.py).
    """
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    return logging.getLogger("jira_agent")


# Module-level logger for config issues caught at import time
logger = configure_logging()
logger.info("Configuration loaded | MOCK_MODE=%s | LLM_MODEL=%s", MOCK_MODE, LLM_MODEL)