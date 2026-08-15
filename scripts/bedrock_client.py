import json
import os
import re

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

bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
)


def converse_text(messages, system_prompt=None, max_tokens=800, temperature=0):
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
