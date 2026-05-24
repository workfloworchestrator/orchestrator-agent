#!/usr/bin/env python3
"""Interactive A2A REPL for orchestrator-agent — artifact-shape inspector.

Holds a single ``contextId`` across turns so the agent's per-thread state
persists (form-fill sessions, planner memory, etc.). Pretty-prints every
artifact's full structure so you can verify what a downstream surface
(LibreChat custom components, Slack Block Kit, …) will receive.

Usage:

    uv run demos/a2a_repl.py
    uv run demos/a2a_repl.py --url http://localhost:8080/
    uv run demos/a2a_repl.py --context my-session-1     # reuse a context across runs

Each prompt sends one A2A ``message/send``. Inside the REPL:

    :ctx                       show the current contextId
    :reset                     start a fresh contextId (drops agent state)
    :skill <skill_id>          pin the next turn to a specific skill (bypasses planner)
    :raw                       toggle raw-JSON dump of the full A2A response
    :quit / Ctrl-D / Ctrl-C    exit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid

import httpx

DEFAULT_URL = "http://localhost:8080/"


def _print_part(part: dict, indent: str = "      ") -> None:
    kind = part.get("kind")
    if kind == "data":
        data = part.get("data", {})
        dumped = json.dumps(data, indent=2, ensure_ascii=False)
        # re-indent each line
        for line in dumped.splitlines():
            print(f"{indent}{line}")
    elif kind == "text":
        text = part.get("text", "")
        print(f"{indent}{text!r}")
    else:
        print(f"{indent}<unknown kind={kind!r}>: {json.dumps(part, ensure_ascii=False)}")


def _print_artifact(artifact: dict, idx: int) -> None:
    name = artifact.get("name") or "<unnamed>"
    md = artifact.get("metadata") or {}
    parts = artifact.get("parts") or []
    print(f"  ── artifact[{idx}]  name={name!r}  parts={len(parts)}")
    if md:
        print(f"      metadata: {json.dumps(md, ensure_ascii=False)}")
    for i, p in enumerate(parts):
        print(f"      part[{i}] kind={p.get('kind')!r}:")
        _print_part(p, indent="        ")


def _print_response(result: dict, raw: bool) -> None:
    if raw:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    status = result.get("status") or {}
    state = status.get("state")
    print(f"  state: {state}")

    artifacts = result.get("artifacts") or []
    if artifacts:
        for i, a in enumerate(artifacts):
            _print_artifact(a, i)
    else:
        print("  (no artifacts)")

    final_parts = (status.get("message") or {}).get("parts") or []
    final_texts = [p.get("text") for p in final_parts if p.get("kind") == "text" and p.get("text")]
    if final_texts:
        print("  ── final text:")
        for t in final_texts:
            for line in t.splitlines() or [""]:
                print(f"      {line}")


def _build_body(*, context_id: str, text: str, metadata: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "kind": "message",
                "messageId": str(uuid.uuid4()),
                "contextId": context_id,
                "parts": [{"kind": "text", "text": text}],
                "metadata": metadata,
            }
        },
    }


async def repl(url: str, context_id: str) -> None:
    raw = False
    pinned_skill: str | None = None

    print(f">>> agent: {url}")
    print(f">>> contextId: {context_id}")
    print(">>> commands: :ctx  :reset  :skill <id>  :raw  :quit\n")

    async with httpx.AsyncClient(timeout=600) as client:
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if not line:
                continue
            if line in (":q", ":quit", ":exit"):
                return
            if line == ":ctx":
                print(f"  contextId = {context_id}\n")
                continue
            if line == ":reset":
                context_id = f"repl-{uuid.uuid4()}"
                print(f"  new contextId = {context_id}\n")
                continue
            if line == ":raw":
                raw = not raw
                print(f"  raw-dump = {raw}\n")
                continue
            if line.startswith(":skill"):
                _, _, rest = line.partition(" ")
                pinned_skill = rest.strip() or None
                print(f"  next turn pinned to skill_id = {pinned_skill!r}\n")
                continue

            metadata: dict = {}
            if pinned_skill:
                metadata["skill_id"] = pinned_skill

            body = _build_body(context_id=context_id, text=line, metadata=metadata)
            try:
                response = await client.post(url, json=body)
            except httpx.HTTPError as e:
                print(f"!!! transport error: {e}\n")
                continue

            if response.status_code != 200:
                print(f"!!! HTTP {response.status_code}: {response.text[:500]}\n")
                continue

            payload = response.json()
            if "error" in payload:
                print(f"!!! A2A error: {json.dumps(payload['error'], ensure_ascii=False)}\n")
                continue

            _print_response(payload.get("result") or {}, raw=raw)
            print()

            # ``:skill`` is one-shot — clear it so subsequent turns go through
            # the planner unless explicitly re-pinned.
            pinned_skill = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="orchestrator-agent A2A base URL")
    parser.add_argument("--context", default=None, help="contextId to use (defaults to a fresh UUID)")
    args = parser.parse_args()

    context_id = args.context or f"repl-{uuid.uuid4()}"
    try:
        asyncio.run(repl(args.url.rstrip("/") + "/", context_id))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
