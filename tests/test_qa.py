# test_qa.py

"""
End-to-end sanity check for the full three-agent pipeline in mock mode.

Runs all three agents in sequence — exactly as LangGraph will —
and prints the output at each stage so you can verify the full
state flow from raw issue string to final approved JIRA ticket.
"""

import json
from agents.triage_agent import run_triage
from agents.resolution_agent import run_resolution
from agents.qa_agent import run_qa

# --- Stage 1: Raw issue comes in ---
state = {
    "issue": "Production database is throwing connection timeout errors. Users cannot log in."
}

# --- Stage 2: Triage Agent ---
state.update(run_triage(state))
print("\n--- Stage 1: Triage Output ---")
print(json.dumps(state["triage"], indent=2))

# --- Stage 3: Resolution Agent ---
state.update(run_resolution(state))
print("\n--- Stage 2: Resolution Output ---")
print(json.dumps(state["resolution"], indent=2))

# --- Stage 4: QA Agent ---
state.update(run_qa(state))
print("\n--- Stage 3: QA Output ---")
print(json.dumps(state["qa"], indent=2))

# --- Final summary ---
print("\n--- Pipeline Summary ---")
qa = state["qa"]
print(f"Approved:        {qa['approved']}")
print(f"Risk level:      {qa['risk_level']}")
print(f"Violations:      {qa['violation_count']}")
print(f"QA notes:        {qa['qa_notes']}")
print(f"Ticket title:    {qa['final_ticket']['ticket_title']}")