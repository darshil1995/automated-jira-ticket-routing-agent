"""Quick sanity check for the Triage Agent in mock mode."""

from agents.triage_agent import run_triage

# Simulate what LangGraph will pass as state
test_state = {
    "issue": "Production database is throwing connection timeout errors. Users cannot log in."
}

result = run_triage(test_state)
print("\n--- Triage Output ---")
import json
print(json.dumps(result, indent=2))