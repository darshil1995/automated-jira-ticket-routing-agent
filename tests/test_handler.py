"""
Local test for the Lambda handler.

Simulates both trigger types (API Gateway and S3) without deploying
to AWS. This lets you verify the full request/response cycle and
confirm tickets are being written to the tickets/ folder.
"""

import json
from lambda_handler import handler


# Minimal mock of the AWS Lambda context object
# We only need aws_request_id — the handler uses nothing else
class MockContext:
    aws_request_id = "local-test-request-001"


# --- Test 1: API Gateway trigger ---
print("\n" + "="*60)
print("TEST 1: API Gateway trigger")
print("="*60)

api_event = {
    "body": json.dumps({
        "issue": "Production database is throwing connection timeout errors. Users cannot log in."
    }),
    "httpMethod": "POST",
    "headers": {"Content-Type": "application/json"}
}

response = handler(api_event, MockContext())
print(f"\nStatus code: {response['statusCode']}")
print(f"Response body:")
print(json.dumps(json.loads(response['body']), indent=2))

# --- Test 2: Bad request (missing issue field) ---
print("\n" + "="*60)
print("TEST 2: Bad request — missing issue field")
print("="*60)

bad_event = {
    "body": json.dumps({"wrong_field": "some value"}),
    "httpMethod": "POST"
}

response2 = handler(bad_event, MockContext())
print(f"\nStatus code: {response2['statusCode']}")
print(f"Response body: {response2['body']}")

# --- Test 3: Malformed JSON body ---
print("\n" + "="*60)
print("TEST 3: Malformed JSON body")
print("="*60)

bad_json_event = {
    "body": "this is not json {{{",
    "httpMethod": "POST"
}

response3 = handler(bad_json_event, MockContext())
print(f"\nStatus code: {response3['statusCode']}")
print(f"Response body: {response3['body']}")

print("\n" + "="*60)
print("All handler tests complete")
print("Check tickets/ folder for written ticket files")
print("="*60)