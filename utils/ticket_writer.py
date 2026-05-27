"""
Mock JIRA Ticket Writer — Simulates filing a ticket in JIRA.

In production this would call the JIRA REST API v3:
    POST https://yourcompany.atlassian.net/rest/api/3/issue

For the portfolio version we write the final approved ticket to a
local JSON file in the tickets/ directory. This gives you a tangible
output you can show in demos and include in your GitHub repo.

Each ticket gets a unique ID based on timestamp so multiple test
runs accumulate in the tickets/ folder — great for showing the
pipeline handling different issue types.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Directory where mock tickets are saved
# Created automatically if it doesn't exist
TICKETS_DIR = os.path.join(os.path.dirname(__file__), "..", "tickets")


def write_ticket(ticket: dict[str, Any]) -> str:
    """
    Writes an approved JIRA ticket to a local JSON file.

    Simulates the JIRA API call that would happen in production.
    The output file structure mirrors what the JIRA REST API would
    return as a confirmation response.

    Args:
        ticket: The final approved ticket dict from the QA Agent.
                Must contain at minimum: ticket_title, ticket_description,
                resolution_steps, affected_systems, escalation_path.

    Returns:
        The file path of the written ticket — useful for logging
        and for the Lambda response body.

    Raises:
        OSError: If the tickets directory cannot be created or
                 the file cannot be written.
    """
    # Ensure the tickets directory exists
    os.makedirs(TICKETS_DIR, exist_ok=True)

    # Generate a unique ticket ID using timestamp
    # Format: MOCK-20240115-143022
    ticket_id = f"MOCK-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Build the full ticket record — wraps the agent output with
    # metadata that a real JIRA ticket would have
    ticket_record = {
        "ticket_id": ticket_id,
        "created_at": datetime.now().isoformat(),
        "status": "open",
        "source": "automated-jira-agent",
        "ticket": ticket
    }

    # Write to tickets/ directory
    filename = f"{ticket_id}.json"
    filepath = os.path.join(TICKETS_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(ticket_record, f, indent=2)

    logger.info(
        "Ticket written | id=%s | path=%s | title='%s'",
        ticket_id,
        filepath,
        ticket.get("ticket_title", "")[:60]
    )

    return filepath