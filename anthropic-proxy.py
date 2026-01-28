#!/usr/bin/env python3
"""
Anthropic-to-OpenAI API proxy for llama.cpp.

Translates Anthropic Messages API (POST /v1/messages) into OpenAI Chat Completions
API (POST /v1/chat/completions) so Claude Code can talk to llama-server.

Supports both streaming (SSE) and non-streaming responses, including tool use.
"""

import json
import sys
import time
import uuid
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

LLAMA_URL = "http://localhost:8080"


def convert_anthropic_to_openai(anthropic_req: dict) -> dict:
    """Convert Anthropic Messages request to OpenAI Chat Completions request."""
    messages = []

    # System message
    system = anthropic_req.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text = "\n".join(
                item["text"] for item in system if item.get("type") == "text" and item.get("text")
            )
            if text:
                messages.append({"role": "system", "content": text})

    # Convert messages
    for msg in anthropic_req.get("messages", []):
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            if role == "user":
                # Extract tool_result blocks as separate tool messages
                for block in content:
                    if block.get("type") == "tool_result":
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, list):
                            tool_content = json.dumps(tool_content)
                        elif not isinstance(tool_content, str):
                            tool_content = json.dumps(tool_content)
                        messages.append({
                            "role": "tool",
                            "content": tool_content,
                            "tool_call_id": block["tool_use_id"],
                        })

                # Extract text/image blocks as user message
                text_parts = []
                for block in content:
                    if block.get("type") == "text" and block.get("text"):
                        text_parts.append(block["text"])
                if text_parts:
                    messages.append({"role": "user", "content": "\n".join(text_parts)})

            elif role == "assistant":
                assistant_msg = {"role": "assistant", "content": ""}

                # Extract text
                text_parts = []
                for block in content:
                    if block.get("type") == "text" and block.get("text"):
                        text_parts.append(block["text"])
                assistant_msg["content"] = "\n".join(text_parts)

                # Extract tool_use blocks
                tool_calls = []
                for block in content:
                    if block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls

                messages.append(assistant_msg)

    # Build OpenAI request
    openai_req = {
        "messages": messages,
        "model": anthropic_req.get("model", "default"),
        "max_tokens": anthropic_req.get("max_tokens", 4096),
        "stream": anthropic_req.get("stream", False),
    }

    if anthropic_req.get("temperature") is not None:
        openai_req["temperature"] = anthropic_req["temperature"]

    # Convert tools
    tools = anthropic_req.get("tools")
    if tools:
        openai_tools = []
        for tool in tools:
            # Skip server tools (web_search etc) that don't map to functions
            if tool.get("type") in ("web_search_20250305", "text_editor_20250429", "code_execution_20250522"):
                continue
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        if openai_tools:
            openai_req["tools"] = openai_tools

    # Convert tool_choice
    tool_choice = anthropic_req.get("tool_choice")
    if tool_choice:
        if isinstance(tool_choice, dict):
            tc_type = tool_choice.get("type")
            if tc_type == "tool":
                openai_req["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice["name"]},
                }
            elif tc_type == "any":
                openai_req["tool_choice"] = "required"
            elif tc_type == "auto":
                openai_req["tool_choice"] = "auto"
            elif tc_type == "none":
                openai_req["tool_choice"] = "none"

    if openai_req.get("stream"):
        openai_req["stream_options"] = {"include_usage": True}

    return openai_req


def convert_openai_response_to_anthropic(openai_resp: dict) -> dict:
    """Convert OpenAI Chat Completion response to Anthropic Messages response."""
    choice = openai_resp.get("choices", [{}])[0]
    message = choice.get("message", {})

    content = []

    # Text content
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})

    # Tool calls
    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            try:
                input_data = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                input_data = {"text": tc["function"].get("arguments", "")}
            content.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": tc["function"]["name"],
                "input": input_data,
            })

    # Map stop reason
    finish_reason = choice.get("finish_reason", "stop")
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "stop_sequence",
    }

    usage = openai_resp.get("usage", {})

    return {
        "id": openai_resp.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "model": openai_resp.get("model", "unknown"),
        "content": content if content else [{"type": "text", "text": ""}],
        "stop_reason": stop_reason_map.get(finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def _read_sse_lines(response):
    """Read SSE lines from an HTTP response incrementally."""
    buf = b""
    while True:
        chunk = response.read(4096)
        if not chunk:
            if buf:
                yield buf.decode("utf-8", errors="replace")
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            yield line.decode("utf-8", errors="replace")


def stream_openai_to_anthropic(openai_response):
    """Generator that converts OpenAI SSE stream to Anthropic SSE stream."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    model = "unknown"
    content_index = 0
    text_block_started = False
    current_block_index = -1
    tool_calls = {}
    has_started = False
    usage_data = None

    for line in _read_sse_lines(openai_response):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            continue

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        model = chunk.get("model", model)

        if not has_started:
            has_started = True
            start_event = {
                "type": "message_start",
                "message": {
                    "id": msg_id, "type": "message", "role": "assistant",
                    "content": [], "model": model,
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }
            yield f"event: message_start\ndata: {json.dumps(start_event)}\n\n"

        if chunk.get("usage"):
            u = chunk["usage"]
            usage_data = {
                "input_tokens": u.get("prompt_tokens", 0),
                "output_tokens": u.get("completion_tokens", 0),
                "cache_read_input_tokens": 0,
            }

        choice = (chunk.get("choices") or [{}])[0]
        if not choice:
            continue
        delta = choice.get("delta", {})

        # Text content
        if delta.get("content"):
            if not text_block_started:
                text_block_started = True
                current_block_index = content_index
                content_index += 1
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': current_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': current_block_index, 'delta': {'type': 'text_delta', 'text': delta['content']}})}\n\n"

        # Tool calls
        if delta.get("tool_calls"):
            for tc in delta["tool_calls"]:
                tc_index = tc.get("index", 0)

                if tc_index not in tool_calls:
                    # Close previous block
                    if current_block_index >= 0:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_block_index})}\n\n"
                        if text_block_started:
                            text_block_started = False

                    block_idx = content_index
                    content_index += 1
                    tc_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                    tc_name = tc.get("function", {}).get("name", f"tool_{tc_index}")
                    tool_calls[tc_index] = {"id": tc_id, "name": tc_name, "content_block_index": block_idx}

                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_idx, 'content_block': {'type': 'tool_use', 'id': tc_id, 'name': tc_name, 'input': {}}})}\n\n"
                    current_block_index = block_idx

                args = tc.get("function", {}).get("arguments", "")
                if args:
                    block_idx = tool_calls[tc_index]["content_block_index"]
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_idx, 'delta': {'type': 'input_json_delta', 'partial_json': args}})}\n\n"

        # Finish
        if choice.get("finish_reason"):
            if current_block_index >= 0:
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_block_index})}\n\n"

            stop_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
            stop_reason = stop_map.get(choice["finish_reason"], "end_turn")
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage_data or {'input_tokens': 0, 'output_tokens': 0, 'cache_read_input_tokens': 0}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            return

    # Clean close if no finish_reason
    if current_block_index >= 0:
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_block_index})}\n\n"
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage_data or {'input_tokens': 0, 'output_tokens': 0, 'cache_read_input_tokens': 0}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


class ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quiet logging - only errors
        if args and "error" in str(args[0]).lower():
            sys.stderr.write(f"[proxy] {format % args}\n")

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/v1/messages":
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            anthropic_req = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
            return

        is_stream = anthropic_req.get("stream", False)

        try:
            openai_req = convert_anthropic_to_openai(anthropic_req)
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Conversion error: {str(e)}"}).encode())
            return

        req_data = json.dumps(openai_req).encode()
        req = urllib.request.Request(
            f"{LLAMA_URL}/v1/chat/completions",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            resp = urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            error_body = e.read().decode("utf-8", errors="replace")
            self.wfile.write(json.dumps({
                "type": "error",
                "error": {"type": "api_error", "message": error_body},
            }).encode())
            return
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "type": "error",
                "error": {"type": "api_error", "message": f"Backend error: {str(e)}"},
            }).encode())
            return

        if is_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            try:
                for event in stream_openai_to_anthropic(resp):
                    self.wfile.write(event.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                resp.close()
        else:
            try:
                openai_resp = json.loads(resp.read().decode())
                anthropic_resp = convert_openai_response_to_anthropic(openai_resp)
                resp_data = json.dumps(anthropic_resp).encode()

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_data)))
                self.end_headers()
                self.wfile.write(resp_data)
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "type": "error",
                    "error": {"type": "api_error", "message": str(e)},
                }).encode())
            finally:
                resp.close()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8082
    server = HTTPServer(("127.0.0.1", port), ProxyHandler)
    sys.stderr.write(f"[proxy] Anthropic->OpenAI proxy on http://127.0.0.1:{port}\n")
    sys.stderr.write(f"[proxy] Forwarding to {LLAMA_URL}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == "__main__":
    main()
