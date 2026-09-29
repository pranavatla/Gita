import json
import os
import re
import time
import urllib.error
import urllib.request

import boto3
from botocore.config import Config


AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "us.anthropic.claude-sonnet-4-6",
)
BEDROCK_EMBED_MODEL_ID = os.environ.get(
    "BEDROCK_EMBED_MODEL_ID",
    "amazon.titan-embed-text-v2:0",
)

# Chat calls go through the gate.atla.in LLM gateway when GATE_API_KEY is set.
# Without it (local development, rollback), gita calls Bedrock directly as before.
GATE_URL = os.environ.get("GATE_URL", "https://gate.atla.in").rstrip("/")
GATE_API_KEY = os.environ.get("GATE_API_KEY", "")
GATE_TIMEOUT_SECONDS = 90
GATE_RETRY_STATUSES = {429, 502, 503, 504}
GATE_ATTEMPTS = 3

bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
)


def _flatten(content):
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in content if isinstance(block, dict)
    )


def _converse_via_gateway(messages, system_prompt, max_tokens, temperature):
    body = {
        "model": f"bedrock/{BEDROCK_MODEL_ID}",
        "messages": [
            {"role": m["role"], "content": _flatten(m["content"])}
            for m in messages
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system_prompt:
        body["system"] = system_prompt
    data = json.dumps(body).encode("utf-8")

    for attempt in range(GATE_ATTEMPTS):
        last_attempt = attempt == GATE_ATTEMPTS - 1
        request = urllib.request.Request(
            f"{GATE_URL}/v1/chat",
            data=data,
            headers={
                "Authorization": f"Bearer {GATE_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=GATE_TIMEOUT_SECONDS) as response:
                return json.loads(response.read())["content"].strip()
        except urllib.error.HTTPError as error:
            request_id = error.headers.get("X-Request-ID", "unknown")
            detail = error.read().decode("utf-8", "replace")[:300]
            if error.code in GATE_RETRY_STATUSES and not last_attempt:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(
                f"Gateway HTTP {error.code} (request {request_id}): {detail}"
            ) from error
        except urllib.error.URLError as error:
            if not last_attempt:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Gateway unreachable: {error.reason}") from error


def _converse_direct(messages, system_prompt, max_tokens, temperature):
    request = {
        "modelId": BEDROCK_MODEL_ID,
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    }

    if system_prompt:
        request["system"] = [{"text": system_prompt}]

    response = bedrock_runtime.converse(**request)

    content_blocks = response["output"]["message"]["content"]
    text_parts = [
        block["text"]
        for block in content_blocks
        if "text" in block
    ]
    return "".join(text_parts).strip()


def converse_text(messages, system_prompt=None, max_tokens=800, temperature=0):
    if GATE_API_KEY:
        return _converse_via_gateway(messages, system_prompt, max_tokens, temperature)
    return _converse_direct(messages, system_prompt, max_tokens, temperature)


def parse_json_text(raw_text):
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    fenced = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        raw_text.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)

    if match:
        return json.loads(match.group(0))

    raise json.JSONDecodeError(
        "Could not parse JSON from model response",
        raw_text,
        0,
    )


def embed_texts(texts, dimensions=1024, normalize=True):
    embeddings = []

    for text in texts:
        request_body = json.dumps(
            {
                "inputText": text,
                "dimensions": dimensions,
                "normalize": normalize,
            }
        )

        response = bedrock_runtime.invoke_model(
            modelId=BEDROCK_EMBED_MODEL_ID,
            body=request_body,
            accept="application/json",
            contentType="application/json",
        )

        response_body = json.loads(response["body"].read())
        embeddings.append(response_body["embedding"])

    return embeddings
