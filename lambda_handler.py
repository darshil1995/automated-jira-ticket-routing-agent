# lambda_handler.py

"""
AWS Lambda Handler — Entry Point for the JIRA Workflow Agent Pipeline.

This module is the bridge between AWS infrastructure and the LangGraph
multi-agent pipeline. It is the only file AWS Lambda calls directly —
everything else is internal implementation.

How AWS Lambda works:
    When a Lambda function is triggered (via API Gateway HTTP request or
    S3 file upload event), AWS calls the handler function defined here
    with two arguments:
        - event:   A dict containing the trigger payload (HTTP body,
                   S3 object info, etc.). Structure varies by trigger type.
        - context: AWS runtime metadata (function name, memory limit,
                   remaining execution time, request ID, etc.)

    Lambda expects the handler to return a dict. For API Gateway triggers
    this dict must include statusCode and body to form a valid HTTP response.
    For S3 triggers the return value is ignored.

Trigger types handled:
    1. API Gateway (HTTP POST)
       Payload: { "body": '{"issue": "..."}' }
       Use case: Slack bot, monitoring webhook, manual trigger

    2. Amazon S3 (file upload event)
       Payload: { "Records": [{ "s3": { "bucket": {...}, "object": {...} } }] }
       Use case: Automated log file processing, batch alert ingestion

Cold starts vs warm starts:
    Lambda freezes execution context between invocations. Module-level
    code (imports, graph compilation, agent initialisation) runs once
    on cold start and is reused on warm starts. This is why graph.py
    and all agents instantiate their singletons at module level — to
    avoid paying the initialisation cost on every invocation.

    Cold start: ~3-5 seconds (first invocation or after idle period)
    Warm start: ~50-200ms (subsequent invocations within the same context)
"""

import json
from typing import Any

import boto3

from graph import run_pipeline
from utils.ticket_writer import write_ticket
import config

# Initialise logging using the central config
# This runs once on cold start — all subsequent invocations reuse this logger
logger = config.configure_logging()


# ---------------------------------------------------------------------------
# S3 Client
# ---------------------------------------------------------------------------

# Initialised at module level for warm start reuse.
# In Lambda, boto3 clients are thread-safe and can be safely reused
# across invocations within the same execution context.
s3_client = boto3.client("s3", region_name=config.AWS_REGION)


# ---------------------------------------------------------------------------
# Event Parsers
# ---------------------------------------------------------------------------

def _parse_api_gateway_event(event: dict) -> str:
    """
    Extracts the issue string from an API Gateway HTTP event.

    API Gateway wraps the HTTP request body as a JSON string inside
    the "body" key. We parse it and extract the "issue" field.

    Expected event structure:
        {
            "body": '{"issue": "Production DB timeout errors"}',
            "httpMethod": "POST",
            "headers": { ... }
        }

    Args:
        event: The raw Lambda event dict from API Gateway.

    Returns:
        The issue string extracted from the request body.

    Raises:
        ValueError: If body is missing, not valid JSON, or has no "issue" key.
    """
    body_raw = event.get("body", "")

    if not body_raw:
        raise ValueError("API Gateway event has empty body")

    try:
        body = json.loads(body_raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"API Gateway body is not valid JSON: {e}") from e

    issue = body.get("issue", "").strip()

    if not issue:
        raise ValueError(
            "Request body must contain a non-empty 'issue' field. "
            "Example: {\"issue\": \"Production DB timeout errors\"}"
        )

    logger.info(
        "API Gateway event parsed | issue_preview='%s'",
        issue[:80]
    )
    return issue


def _parse_s3_event(event: dict) -> str:
    """
    Extracts the issue string from an S3 object upload event.

    When a file is uploaded to the trigger bucket, Lambda receives
    an S3 event containing the bucket name and object key. We fetch
    the file content from S3 and use it as the issue text.

    This supports automated pipelines where monitoring systems drop
    alert files into S3 rather than calling an HTTP endpoint.

    Expected event structure:
        {
            "Records": [{
                "s3": {
                    "bucket": { "name": "jira-agent-triggers" },
                    "object": { "key": "alerts/alert-2024-01-15.txt" }
                }
            }]
        }

    Args:
        event: The raw Lambda event dict from S3.

    Returns:
        The file content as the issue string.

    Raises:
        ValueError: If the S3 event structure is malformed.
        RuntimeError: If the S3 object cannot be fetched.
    """
    try:
        record = event["Records"][0]
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Malformed S3 event structure: {e}") from e

    logger.info("S3 event parsed | bucket=%s | key=%s", bucket, key)

    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        issue = response["Body"].read().decode("utf-8").strip()
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch S3 object s3://{bucket}/{key}: {e}"
        ) from e

    if not issue:
        raise ValueError(f"S3 object s3://{bucket}/{key} is empty")

    logger.info(
        "S3 object read | bucket=%s | key=%s | content_preview='%s'",
        bucket, key, issue[:80]
    )
    return issue


def _detect_event_type(event: dict) -> str:
    """
    Determines whether the Lambda was triggered by API Gateway or S3.

    Uses the presence of known top-level keys to identify the source.
    This approach is more reliable than checking the event source ARN
    because it works identically in local testing and production Lambda.

    Args:
        event: The raw Lambda event dict.

    Returns:
        "api_gateway" or "s3"

    Raises:
        ValueError: If the event structure matches neither known type.
    """
    if "body" in event:
        logger.info("Event type detected: api_gateway")
        return "api_gateway"

    if "Records" in event and event["Records"]:
        source = event["Records"][0].get("eventSource", "")
        if "s3" in source or "s3" in str(event["Records"][0]):
            logger.info("Event type detected: s3")
            return "s3"

    raise ValueError(
        f"Unrecognised event structure. Expected API Gateway or S3 event. "
        f"Top-level keys found: {list(event.keys())}"
    )


# ---------------------------------------------------------------------------
# Response Builders
# ---------------------------------------------------------------------------

def _success_response(result: dict[str, Any]) -> dict:
    """
    Builds a successful API Gateway HTTP response.

    API Gateway requires responses in this exact structure —
    statusCode (int) and body (JSON string). Missing either
    causes API Gateway to return a 502 Bad Gateway to the caller.

    Args:
        result: The complete pipeline output dict.

    Returns:
        API Gateway-compatible response dict.
    """
    qa = result.get("qa", {})

    response_body = {
        "status": "success",
        "approved": qa.get("approved"),
        "risk_level": qa.get("risk_level"),
        "violation_count": qa.get("violation_count", 0),
        "ticket_title": qa.get("final_ticket", {}).get("ticket_title"),
        "qa_notes": qa.get("qa_notes"),
        "triage_summary": {
            "category": result.get("triage", {}).get("category"),
            "priority": result.get("triage", {}).get("priority"),
            "confidence": result.get("triage", {}).get("confidence"),
        }
    }

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            # CORS header — required if a web dashboard calls this API
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(response_body, indent=2)
    }


def _error_response(status_code: int, error_message: str) -> dict:
    """
    Builds an error API Gateway HTTP response.

    Args:
        status_code: HTTP status code (400 for bad input, 500 for server error).
        error_message: Human-readable description of what went wrong.

    Returns:
        API Gateway-compatible error response dict.
    """
    logger.error("Returning error response | status=%d | message=%s",
                 status_code, error_message)

    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "status": "error",
            "message": error_message
        })
    }


# ---------------------------------------------------------------------------
# Main Handler
# ---------------------------------------------------------------------------

def handler(event: dict, context: Any) -> dict:
    """
    AWS Lambda entry point — called by AWS on every invocation.

    Orchestrates the full request lifecycle:
        1. Log the invocation for CloudWatch tracing
        2. Detect whether the trigger was API Gateway or S3
        3. Parse the issue string from the event
        4. Run the full three-agent LangGraph pipeline
        5. Write the approved ticket to the mock JIRA system
        6. Return a structured HTTP response

    Error handling strategy:
        - ValueError (bad input): return 400 so the caller knows to fix their request
        - RuntimeError (infrastructure failure): return 500 and log for alerting
        - Any other Exception: return 500 and log full traceback to CloudWatch

    Args:
        event:   AWS event dict. Structure depends on trigger type.
                 See _parse_api_gateway_event() and _parse_s3_event()
                 for expected structures.
        context: AWS Lambda context object. Used here only for the
                 AWS request ID which is logged for CloudWatch correlation.

    Returns:
        API Gateway response dict with statusCode, headers, and body.
        For S3 triggers the return value is unused by AWS but we
        return the same structure for consistency and local testability.
    """
    # Log invocation metadata for CloudWatch correlation
    # context.aws_request_id links this log line to the Lambda invocation
    # in CloudWatch so you can trace a single request across all log lines
    request_id = getattr(context, "aws_request_id", "local-test")
    logger.info(
        "Lambda handler invoked | request_id=%s | event_keys=%s",
        request_id,
        list(event.keys())
    )

    try:
        # Step 1: Detect trigger source and extract issue string
        event_type = _detect_event_type(event)

        if event_type == "api_gateway":
            issue = _parse_api_gateway_event(event)
        else:
            issue = _parse_s3_event(event)

        # Step 2: Run the full multi-agent pipeline
        logger.info("Starting pipeline | request_id=%s", request_id)
        result = run_pipeline(issue)

        # Step 3: Write the final approved ticket to mock JIRA
        ticket = result.get("qa", {}).get("final_ticket")
        if ticket and result.get("qa", {}).get("approved"):
            write_ticket(ticket)
            logger.info("Ticket written successfully | request_id=%s", request_id)
        else:
            logger.warning(
                "Ticket not written — QA rejected | request_id=%s | violations=%s",
                request_id,
                result.get("qa", {}).get("violations", [])
            )

        # Step 4: Return success response
        logger.info(
            "Handler complete | request_id=%s | approved=%s | priority=%s",
            request_id,
            result.get("qa", {}).get("approved"),
            result.get("triage", {}).get("priority")
        )
        return _success_response(result)

    except ValueError as e:
        # Bad input from the caller — their problem to fix
        logger.warning("Bad request | request_id=%s | error=%s", request_id, str(e))
        return _error_response(400, str(e))

    except RuntimeError as e:
        # Infrastructure failure — our problem to fix
        logger.error(
            "Runtime error | request_id=%s | error=%s",
            request_id, str(e),
            exc_info=True     # Includes full traceback in CloudWatch
        )
        return _error_response(500, f"Internal pipeline error: {str(e)}")

    except Exception as e:
        # Unexpected error — log everything for debugging
        logger.critical(
            "Unhandled exception | request_id=%s | error=%s",
            request_id, str(e),
            exc_info=True
        )
        return _error_response(500, "An unexpected error occurred")