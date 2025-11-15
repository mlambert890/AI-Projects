import os
import json
import time
import uuid
import logging
import urllib3

from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler, AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "10"))
RESPONSE_LAMBDA_URL = os.environ.get("RESPONSE_LAMBDA_URL")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "0.5"))
TIMEOUT_SECONDS = float(os.environ.get("TIMEOUT_SECONDS", "20"))

http = urllib3.PoolManager()


def send_to_ha_webhook(request_id: str, prompt: str):
    logger.info("Calling HA webhook!")
    payload = {"request_id": request_id, "prompt": prompt}
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    resp = http.request(
        "POST", WEBHOOK_URL, body=body, headers=headers,
        timeout=urllib3.Timeout(connect=5.0, read=HTTP_TIMEOUT)
    )
    logger.info("HA webhook POST status=%s request_id=%s", resp.status, request_id)
    return resp.status


def fetch_response(request_id: str):
    logger.info("Fetching response for request_id=%s", request_id)
    try:
        resp = http.request(
            "GET",
            f"{RESPONSE_LAMBDA_URL}?request_id={request_id}",
            timeout=urllib3.Timeout(connect=5.0, read=10.0)
        )
    except Exception as e:
        logger.exception("HTTP request to response lambda failed: %s", e)
        return None

    if resp.status != 200:
        logger.warning("Response lambda returned status=%s", resp.status)
        return None

    try:
        data = json.loads(resp.data.decode("utf-8"))
        response_val = data.get("response")
        logger.info("Raw response received: %s", response_val)
        return response_val
    except Exception as e:
        logger.exception("Failed to parse JSON from response lambda: %s", e)
        return None





def _speech_safe(text: str, max_len=7000):
    if not text:
        return ""
    return text.strip()[:max_len]


# -------------------- HANDLERS --------------------

class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return handler_input.request_envelope.request.object_type == "LaunchRequest"

    def handle(self, handler_input):
        speak = "Welcome to your local AI agent. Ask me anything."
        return handler_input.response_builder.speak(speak).ask("What would you like to ask?").response


class AskIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        req = handler_input.request_envelope.request
        return (
            req.object_type == "IntentRequest"
            and getattr(req.intent, "name", "") == "AskIntent"
        )

    def handle(self, handler_input):
        intent = handler_input.request_envelope.request.intent
        prompt = None
        logger.info("Entered AskIntent")
        if getattr(intent, "slots", None):
            for name in ["Query", "query", "Question", "question", "q", "text"]:
                slot = intent.slots.get(name)
                if slot and getattr(slot, "value", None):
                    prompt = slot.value
                    break

        if not prompt:
            speak = "I didn't catch your question. Please say it again."
            return handler_input.response_builder.speak(speak).ask(speak).response

        request_id = str(uuid.uuid4())
        logger.info("Attempting to call HA webhook!")
        try:
            status = send_to_ha_webhook(request_id, prompt)
            if status >= 400:
                return handler_input.response_builder.speak(
                    "The AI server returned an error. Try again later."
                ).response
        except Exception:
            return handler_input.response_builder.speak(
                "Sorry, I couldn't reach the AI server."
            ).response
        logger.info("Successfully called HA webhook!")

        # Poll Response Lambda
        deadline = time.time() + TIMEOUT_SECONDS
        ai_text = None
        logger.info("Attempting to fetch response!")
        while time.time() < deadline:
            ai_text = fetch_response(request_id)
            if ai_text:
                break
            time.sleep(POLL_INTERVAL)

        if not ai_text:
            return handler_input.response_builder.speak(
                "Sorry, the AI didn't respond in time."
            ).response
        logger.info("Successfully fetched response!")

        return handler_input.response_builder.speak(
            _speech_safe(ai_text)
        ).set_should_end_session(True).response


class FallbackIntentHandler(AbstractRequestHandler):
    """Only catch AMAZON.FallbackIntent, not all IntentRequests."""
    def can_handle(self, handler_input):
        req = handler_input.request_envelope.request
        return (
            req.object_type == "IntentRequest"
            and getattr(req.intent, "name", "") == "AMAZON.FallbackIntent"
        )

    def handle(self, handler_input):
        logger.info("Entered FallbackIntent")
        speak = "I didn't understand that. Please ask again."
        return handler_input.response_builder.speak(speak).ask(speak).response


class SessionEndedRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return handler_input.request_envelope.request.object_type == "SessionEndedRequest"

    def handle(self, handler_input):
        return handler_input.response_builder.response


class UniversalRequestHandler(AbstractRequestHandler):
    """
    LAST RESORT OMNI-HANDLER.
    Catches ANY request type not matched above:
    - CanFulfillIntentRequest
    - Connections.Request
    - Unexpected Intent without name
    - APL/UserEvent
    - Anything malformed
    """
    def can_handle(self, handler_input):
        return True

    def handle(self, handler_input):
        logger.warning("UniversalRequestHandler handling unknown request: %s",
                       handler_input.request_envelope.request)
        speak = "I'm here. You can ask me anything."
        return handler_input.response_builder.speak(speak).ask(speak).response


class CatchAllExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        logger.exception("Unhandled exception: %s", exception)
        return handler_input.response_builder.speak(
            "Sorry, I ran into an error."
        ).response


# -------------------- BUILD SKILL --------------------

sb = SkillBuilder()
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(AskIntentHandler())
sb.add_request_handler(FallbackIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())

# Very important: add last!
sb.add_request_handler(UniversalRequestHandler())

sb.add_exception_handler(CatchAllExceptionHandler())

alexa_lambda_handler = sb.lambda_handler()


def lambda_handler(event, context):
    return alexa_lambda_handler(event, context)
