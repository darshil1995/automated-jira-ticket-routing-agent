# test_triage.py
"""
Quick local test for the Triage Agent.
Run with: python test_triage.py
No AWS or OpenAI credentials needed when MOCK_MODE=True.
"""

from agents.triage_agent import run_triage

# Three different issue types to verify mock routing works
test_issues = [
    "Production database is throwing connection timeout errors since 2am",
    "Users cannot log in — authentication service returning 401",
    "Export CSV button on the dashboard is not responding",
]

for issue in test_issues:
    print(f"\nIssue: {issue}")
    print("-" * 60)
    result = run_triage(issue)
    for key, value in result.items():
        print(f"  {key}: {value}")