"""
Transparent OpenAI-compatible proxy for e-INFRA CZ that bridges non-multimodal models (like GLM-5.3)
with vision capabilities by automatically transcribing images via e-INFRA's multimodal model (qwen3.8-flash-next).
"""

import os
import json
import logging
import asyncio
from typing import Any, Dict, List
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("einfra-proxy")

UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "https://llm.ai.e-infra.cz/v1").rstrip("/")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-50dc3f278a5545cebfba2ce0d4016597")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen3.8-flash-next")
MULTIMODAL_MODELS = {"qwen3.8-flash-next", "gemma4", "qwen3.8-27b"}

app = FastAPI(title="e-INFRA Multimodal Bridge Proxy")
http_client = httpx.AsyncClient(timeout=120.0)


@app.on_event("shutdown")
async def shutdown():
    await http_client.aclose()


@app.get("/health")
async def health():
    return {"status": "ok", "upstream": UPSTREAM_BASE_URL, "vision_model": VISION_MODEL}


async def describe_image(image_url: str, auth_header: str) -> str:
    """Use the upstream vision model to generate a rich visual description."""
    try:
        headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json",
        }
        prompt = (
            "Describe this image thoroughly and accurately for a technical/scientific AI assistant. "
            "Include chart types, axes labels, legends, numerical trends, molecular structures, code, or UI elements shown."
        )
        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "max_tokens": 1024,
        }
        resp = await http_client.post(
            f"{UPSTREAM_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            logger.warning("Vision transcription failed with status %s: %s", resp.status_code, resp.text)
            return f"[Image description unavailable: HTTP {resp.status_code}]"
    except Exception as e:
        logger.error("Error transcribing image: %s", e)
        return f"[Image transcription error: {str(e)}]"


async def transform_messages_for_non_multimodal(messages: List[Dict[str, Any]], auth_header: str) -> List[Dict[str, Any]]:
    """Inspect messages; for any image blocks, asynchronously transcribe them to text."""
    tasks = []
    task_positions = []

    for m_idx, msg in enumerate(messages):
        content = msg.get("content")
        if isinstance(content, list):
            for p_idx, part in enumerate(content):
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type", "")
                img_url = None
                if part_type == "image_url":
                    img_info = part.get("image_url", {})
                    img_url = img_info.get("url") if isinstance(img_info, dict) else img_info
                elif part_type == "image":
                    img_url = part.get("image")
                
                if img_url:
                    tasks.append(describe_image(img_url, auth_header))
                    task_positions.append((m_idx, p_idx))

    if not tasks:
        return messages

    logger.info("Transcribing %d image(s) for text-only model...", len(tasks))
    descriptions = await asyncio.gather(*tasks)

    # Clone messages deeply enough to modify
    new_messages = []
    for msg in messages:
        c = msg.get("content")
        if isinstance(c, list):
            new_messages.append({**msg, "content": [p.copy() if isinstance(p, dict) else p for p in c]})
        else:
            new_messages.append({**msg})

    for (m_idx, p_idx), desc in zip(task_positions, descriptions):
        new_messages[m_idx]["content"][p_idx] = {
            "type": "text",
            "text": f"[Attached Image Visual Analysis: {desc}]",
        }

    return new_messages


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    auth_header = request.headers.get("Authorization", f"Bearer {DEFAULT_API_KEY}")
    try:
        body = await request.json()
    except Exception:
        body = {}

    model = body.get("model", "")
    # Check if model starts with "e-infra/"
    clean_model = model.split("/")[-1] if "/" in model else model
    body["model"] = clean_model

    messages = body.get("messages", [])
    is_stream = body.get("stream", False)

    # Check if image transcription is required
    has_images = any(
        isinstance(m.get("content"), list) and any(
            isinstance(p, dict) and p.get("type") in ("image_url", "image")
            for p in m.get("content", [])
        )
        for m in messages
    )

    if has_images and clean_model not in MULTIMODAL_MODELS:
        logger.info("Intercepted multimodal request for text-only model '%s'", clean_model)
        body["messages"] = await transform_messages_for_non_multimodal(messages, auth_header)

    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json",
    }

    if is_stream:
        async def stream_generator():
            max_retries = 3
            backoff = 1.5
            for attempt in range(max_retries):
                try:
                    async with http_client.stream(
                        "POST",
                        f"{UPSTREAM_BASE_URL}/chat/completions",
                        headers=headers,
                        json=body,
                        timeout=120.0,
                    ) as response:
                        if response.status_code == 429 and attempt < max_retries - 1:
                            logger.warning("Upstream 429 encountered, retrying in %.1fs (attempt %d/%d)...", backoff, attempt + 1, max_retries)
                            await asyncio.sleep(backoff)
                            backoff *= 2
                            continue
                        async for chunk in response.aiter_bytes():
                            yield chunk
                        break
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                    else:
                        raise e

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        max_retries = 3
        backoff = 1.5
        for attempt in range(max_retries):
            resp = await http_client.post(
                f"{UPSTREAM_BASE_URL}/chat/completions",
                headers=headers,
                json=body,
                timeout=120.0,
            )
            if resp.status_code == 429 and attempt < max_retries - 1:
                logger.warning("Upstream 429 encountered, retrying in %.1fs (attempt %d/%d)...", backoff, attempt + 1, max_retries)
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
async def passthrough(path: str, request: Request):
    """Pass all other requests through to e-INFRA verbatim."""
    auth_header = request.headers.get("Authorization", f"Bearer {DEFAULT_API_KEY}")
    url_path = path.lstrip("/")
    if url_path.startswith("v1/"):
        target_url = f"{UPSTREAM_BASE_URL}/{url_path[3:]}"
    else:
        target_url = f"{UPSTREAM_BASE_URL}/{url_path}"

    body = await request.body()
    headers = {
        "Authorization": auth_header,
    }
    content_type = request.headers.get("Content-Type")
    if content_type:
        headers["Content-Type"] = content_type

    req = http_client.build_request(
        method=request.method,
        url=target_url,
        headers=headers,
        content=body,
        params=request.query_params,
        timeout=60.0,
    )
    resp = await http_client.send(req)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PROXY_PORT", "34053"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
