import os
import json
import time
import logging
import boto3
from botocore.exceptions import ClientError, BotoCoreError

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

DDB_TABLE = os.environ.get("DDB_TABLE")
if not DDB_TABLE:
    logger.error("Environment variable DDB_TABLE not set!")

dynamodb = boto3.client("dynamodb", region_name="us-east-2")


def extract_ai_text(response_payload):
    """
    Safely extract assistant text from HA payload.
    Handles nested structures like Ollama responses.
    """
    if response_payload is None:
        return None

    if isinstance(response_payload, str):
        return response_payload.strip()

    if isinstance(response_payload, dict):
        # Try known paths
        try:
            return response_payload["content"]["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            # fallback: maybe top-level 'text' key
            text = response_payload.get("text")
            if isinstance(text, str):
                return text.strip()
    return None


def persist_response(request_id: str, ai_text: str):
    """Write the AI response to DynamoDB."""
    if not request_id or not ai_text:
        logger.warning("persist_response called with empty request_id or ai_text")
        return False

    item = {
        "request_id": {"S": request_id},
        "response": {"S": ai_text},
        "ts": {"N": str(int(time.time()))},
    }

    try:
        dynamodb.put_item(TableName=DDB_TABLE, Item=item)
        logger.info("Persisted response for request_id=%s", request_id)
        return True
    except (ClientError, BotoCoreError) as e:
        logger.exception("Failed to persist response for request_id=%s: %s", request_id, e)
        return False


def lambda_handler(event, context):
    logger.info("Received event: %s", json.dumps(event))

    # Determine HTTP method
    method = event.get("requestContext", {}).get("http", {}).get("method")
    if not method:
        logger.warning("No HTTP method found in event")
        return {"statusCode": 400, "body": "Bad request"}

    try:
        method = method.upper()
        if method == "POST":
            body = json.loads(event.get("body", "{}"))
            request_id = body.get("request_id")
            ai_text_raw = body.get("response") or body.get("text")
            ai_text = extract_ai_text(ai_text_raw)

            logger.info("Parsed request_id=%s, ai_text=%s", request_id, ai_text)

            if not request_id or not ai_text:
                logger.warning("POST missing request_id or response text: %s", body)
                return {"statusCode": 400, "body": "Missing request_id or valid response text"}

            success = persist_response(request_id, ai_text)
            return {"statusCode": 200, "body": json.dumps({"ok": success})}

        elif method == "GET":
            request_id = event.get("queryStringParameters", {}).get("request_id")
            if not request_id:
                logger.warning("GET missing request_id")
                return {"statusCode": 400, "body": "Missing request_id"}

            try:
                resp = dynamodb.get_item(TableName=DDB_TABLE, Key={"request_id": {"S": request_id}})
                item = resp.get("Item")
                if not item:
                    logger.info("No item found for request_id=%s", request_id)
                    return {"statusCode": 200, "body": json.dumps({"response": None})}

                response_val = item.get("response", {}).get("S")
                logger.info("Fetched response for request_id=%s: %s", request_id, response_val)
                return {"statusCode": 200, "body": json.dumps({"response": response_val})}

            except (ClientError, BotoCoreError) as e:
                logger.exception("Failed to fetch item for request_id=%s: %s", request_id, e)
                return {"statusCode": 500, "body": "Internal server error"}

        else:
            logger.warning("Method not allowed: %s", method)
            return {"statusCode": 405, "body": "Method not allowed"}

    except Exception as e:
        logger.exception("Unhandled exception in lambda_handler: %s", e)
        return {"statusCode": 500, "body": "Internal server error"}
