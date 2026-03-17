#!/usr/bin/env python3
"""
Agent Chat Integration Test
============================
模拟前端向 Agent 发送带图片的请求，验证完整的 Agent → LLM → Tool 调用链路。

Usage:
    python scripts/test_agent.py
    python scripts/test_agent.py --image /data1/yuhongjie2/test.png
    python scripts/test_agent.py --email user@example.com --password secret
    python scripts/test_agent.py --token <jwt_token>   # 跳过登录，直接用 token
    python scripts/test_agent.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_EMAIL    = "admin@terrabox.local"
DEFAULT_PASSWORD = "admin"
DEFAULT_IMAGE    = "/data1/yuhongjie2/test.png"
DEFAULT_PROMPT   = "帮我用vlm工具查看图像中有什么"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def login(base_url: str, email: str, password: str) -> str:
    """POST /v1/login and return the JWT access token."""
    url = f"{base_url}/v1/login"
    resp = requests.post(url, json={"email": email, "password": password}, timeout=10, proxies={"http": None, "https": None})
    if resp.status_code != 200:
        print(f"[FAIL] Login failed ({resp.status_code}): {resp.text}")
        sys.exit(1)
    token = resp.json().get("access_token")
    if not token:
        print(f"[FAIL] No access_token in response: {resp.json()}")
        sys.exit(1)
    print(f"[OK]   Logged in as {email}")
    return token


def send_agent_chat(
    base_url: str,
    token: str,
    prompt: str,
    image_path: str,
    session_id: Optional[str] = None,
) -> dict:
    """POST /v1/gui/agent/chat with multipart form data."""
    url = f"{base_url}/v1/gui/agent/chat"
    headers = {"Authorization": f"Bearer {token}"}

    with open(image_path, "rb") as f:
        files = {"files": (image_path.split("/")[-1], f, "image/png")}
        data = {"message": prompt}
        if session_id:
            data["session_id"] = session_id

        print(f"\n[SEND] POST {url}")
        print(f"       prompt  : {prompt!r}")
        print(f"       image   : {image_path}")

        resp = requests.post(url, headers=headers, data=data, files=files, timeout=300, proxies={"http": None, "https": None})

    return resp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Agent chat integration test")
    parser.add_argument("--base-url",  default=DEFAULT_BASE_URL,  help="Terrabox backend URL")
    parser.add_argument("--email",     default=DEFAULT_EMAIL,      help="Login email")
    parser.add_argument("--password",  default=DEFAULT_PASSWORD,   help="Login password")
    parser.add_argument("--token",     default=None,               help="JWT token (skip login)")
    parser.add_argument("--image",     default=DEFAULT_IMAGE,      help="Image file to upload")
    parser.add_argument("--prompt",    default=DEFAULT_PROMPT,     help="Prompt to send")
    args = parser.parse_args()

    # 1. Auth
    if args.token:
        token = args.token
        print(f"[OK]   Using provided JWT token")
    else:
        token = login(args.base_url, args.email, args.password)

    # 2. Send request
    resp = send_agent_chat(args.base_url, token, args.prompt, args.image)

    # 3. Report
    print(f"\n[RESP] Status: {resp.status_code}")
    if resp.status_code == 200:
        body = resp.json()
        print(f"[OK]   session_id : {body.get('session_id')}")
        print(f"\n{'='*60}")
        print("Agent response:")
        print(f"{'='*60}")
        print(body.get("response", "(empty)"))
        print(f"{'='*60}\n")
    else:
        print(f"[FAIL] Response body:")
        try:
            print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
        except Exception:
            print(resp.text)
        sys.exit(1)


if __name__ == "__main__":
    main()
