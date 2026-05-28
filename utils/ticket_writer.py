# utils/ticket_writer.py

import json
import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# In production, replace this with a JIRA REST API v3 call:
# POST https://yourcompany.atlassian.net/rest/api/3/issue
TICKETS_DIR = os.path.join(os.path.dirname(__file__), "..", "tickets")


def write_ticket(ticket: dict[str, Any]) -> str:
    """
    Persists an approved ticket as a JSON file and returns its file path.

    Raises:
        OSError: If the tickets directory cannot be created or written to.
    """
    os.makedirs(TICKETS_DIR, exist_ok=True)

    ticket_id = f"MOCK-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    filepath = os.path.join(TICKETS_DIR, f"{ticket_id}.json")

    ticket_record = {
        "ticket_id": ticket_id,
        "created_at": datetime.now().isoformat(),
        "status": "open",
        "source": "automated-jira-agent",
        "ticket": ticket,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(ticket_record, f, indent=2)

    logger.info(
        "Ticket written | id=%s | title='%s'",
        ticket_id,
        ticket.get("ticket_title", "")[:60]
    )

    return filepath