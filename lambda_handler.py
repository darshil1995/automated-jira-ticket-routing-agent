import json
import logging
from typing import Any

import boto3

from graph import run_pipeline
from utils.ticket_writer import write_ticket
import config

logger = config.configure_logging()

# Boto3 clients are thread-safe and reused across Lambda warm starts.
s3_client = boto3.client("s3", region_name=config.AWS_REGION)


def _parse_api_gateway_event(event: dict) -> str:
    """
    Extracts the issue string from an API Gateway POST event body.

    Raises:
        ValueError: If the body is missing, not valid JSON, or lacks an 'issue' field.
    """
    body_raw = event.get("body", "")

    if not body_raw:
        raise ValueError("API Gateway event has empty body")

    try:
        body = json.loads(body_raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Request body is not valid JSON: {e}") from e

    issue = body.get("issue", "").strip()

    if not issue:
        raise ValueError(
            "Request body must contain a non-empty 'issue' field. "
            'Example: {"issue": "Production DB timeout errors"}'
        )

    logger.info("API Gateway event parsed | issue_preview='%s'", issue[:80])
    return issue


def _parse_s3_event(event: dict) -> str:
    """
    Fetches and returns the text content of the uploaded S3 object.

    Raises:
        ValueError: If the S3 event structure is malformed or the object is empty.
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
        raise RuntimeError(f"Failed to fetch s3://{bucket}/{key}: {e}") from e

    if not issue:
        raise ValueError(f"S3 object s3://{bucket}/{key} is empty")

    logger.info("S3 object read | bucket=%s | key=%s | preview='%s'", bucket, key, issue[:80])
    return issue


def _detect_event_type(event: dict) -> str:
    """
    Identifies whether the trigger source is API Gateway or S3.

    Inspects top-level event keys rather than the source ARN so detection
    works identically in local tests and production Lambda.

    Raises:
        ValueError: If the event matches neither known structure.
    """
    if "body" in event:
        return "api_gateway"

    if "Records" in event and event["Records"]:
        if "s3" in str(event["Records"][0]):
            return "s3"

    raise ValueError(
        f"Unrecognised event structure. Expected API Gateway or S3. "
        f"Top-level keys: {list(event.keys())}"
    )


def _success_response(result: dict[str, Any]) -> dict:
    """
    Builds an API Gateway-compatible 200 response.

    statusCode and body are both required — missing either causes
    API Gateway to return 502 Bad Gateway to the caller.
    """
    qa = result.get("qa", {})

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",  # Required for browser dashboard callers
        },
        "body": json.dumps({
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
            },
        }, indent=2),
    }


def _error_response(status_code: int, error_message: str) -> dict:
    logger.error("Returning error response | status=%d | message=%s", status_code, error_message)
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"status": "error", "message": error_message}),
    }


def handler(event: dict, context: Any) -> dict:
    """
    AWS Lambda entry point — handles both API Gateway and S3 trigger events.

    Error strategy:
        ValueError  → 400 (caller sent bad input, their fix)
        RuntimeError → 500 (infrastructure failure, our fix)
        Exception   → 500 (unexpected, full traceback logged to CloudWatch)
    """
    request_id = getattr(context, "aws_request_id", "local-test")
    logger.info(
        "Lambda invoked | request_id=%s | event_keys=%s",
        request_id, list(event.keys())
    )

    try:
        event_type = _detect_event_type(event)
        issue = _parse_api_gateway_event(event) if event_type == "api_gateway" else _parse_s3_event(event)

        result = run_pipeline(issue)

        ticket = result.get("qa", {}).get("final_ticket")
        if ticket and result.get("qa", {}).get("approved"):
            write_ticket(ticket)
            logger.info("Ticket written | request_id=%s", request_id)
        else:
            logger.warning(
                "Ticket not written — QA rejected | request_id=%s | violations=%s",
                request_id, result.get("qa", {}).get("violations", [])
            )

        logger.info(
            "Handler complete | request_id=%s | approved=%s | priority=%s",
            request_id,
            result.get("qa", {}).get("approved"),
            result.get("triage", {}).get("priority")
        )
        return _success_response(result)

    except ValueError as e:
        logger.warning("Bad request | request_id=%s | error=%s", request_id, str(e))
        return _error_response(400, str(e))

    except RuntimeError as e:
        logger.error("Runtime error | request_id=%s | error=%s", request_id, str(e), exc_info=True)
        return _error_response(500, f"Internal pipeline error: {str(e)}")

    except Exception as e:
        logger.critical("Unhandled exception | request_id=%s | error=%s", request_id, str(e), exc_info=True)
        return _error_response(500, "An unexpected error occurred")