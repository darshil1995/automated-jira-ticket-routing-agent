"""
Full end-to-end pipeline test.

Tests the complete graph invocation via run_pipeline() — the same
function that lambda_handler.py will call. This is the closest thing
to a production run without actually deploying to AWS.

Run this to verify the entire system works before moving to Lambda.
"""

from graph import run_pipeline

# --- Test case 1: Database infrastructure issue ---
print("\n" + "="*60)
print("TEST 1: Database infrastructure issue (P1)")
print("="*60)

result = run_pipeline(
    "Production database is throwing connection timeout errors. "
    "Users cannot log in. Started at 2am after deployment."
)

print(f"\nTriage:     {result['triage']['category']} | {result['triage']['priority']}")
print(f"Confidence: {result['triage']['confidence']}")
print(f"Action:     {result['triage']['recommended_action']}")
print(f"\nTicket:     {result['resolution']['ticket_title']}")
print(f"Steps:      {len(result['resolution']['resolution_steps'])} steps")
print(f"\nApproved:   {result['qa']['approved']}")
print(f"Risk:       {result['qa']['risk_level']}")
print(f"Notes:      {result['qa']['qa_notes']}")

# --- Test case 2: Auth issue ---
print("\n" + "="*60)
print("TEST 2: Authentication issue (P2)")
print("="*60)

result2 = run_pipeline(
    "Users are getting 401 errors when trying to log in. "
    "Password reset emails are not being sent."
)

print(f"\nTriage:     {result2['triage']['category']} | {result2['triage']['priority']}")
print(f"Action:     {result2['triage']['recommended_action']}")
print(f"\nTicket:     {result2['resolution']['ticket_title']}")
print(f"\nApproved:   {result2['qa']['approved']}")
print(f"Risk:       {result2['qa']['risk_level']}")

# --- Test case 3: UI bug ---
print("\n" + "="*60)
print("TEST 3: Frontend UI bug (P3)")
print("="*60)

result3 = run_pipeline(
    "The export CSV button on the reports page is not working. "
    "Users see a blank page when clicking it."
)

print(f"\nTriage:     {result3['triage']['category']} | {result3['triage']['priority']}")
print(f"Action:     {result3['triage']['recommended_action']}")
print(f"\nTicket:     {result3['resolution']['ticket_title']}")
print(f"\nApproved:   {result3['qa']['approved']}")
print(f"Risk:       {result3['qa']['risk_level']}")

print("\n" + "="*60)
print("All tests complete")
print("="*60)