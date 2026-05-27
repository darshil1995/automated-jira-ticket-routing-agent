"""Sanity check for the Resolution Agent in mock mode."""

import json
from agents.triage_agent import run_triage
from agents.resolution_agent import run_resolution

# Step 1: Run triage first, just like LangGraph would
triage_state = {
    "issue": "Production database is throwing connection timeout errors. Users cannot log in."
}
triage_output = run_triage(triage_state)
print("\n--- Triage Output ---")
print(json.dumps(triage_output, indent=2))

# Step 2: Feed real triage output into resolution — no manual keys
resolution_state = {"triage": triage_output["triage"]}
resolution_output = run_resolution(resolution_state)
print("\n--- Resolution Output ---")
print(json.dumps(resolution_output, indent=2))