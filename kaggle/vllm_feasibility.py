# Copyright (c) 2026 Jiahao Tan

"""Kaggle T4 上的 vLLM 可行性检查(约 15-25 分钟 GPU)。

检查四件事, 结果写入 /kaggle/working/feasibility.json:
1. vLLM 能否在 T4(计算能力 7.5, fp16)上安装并启动 OpenAI 兼容服务;
2. 开启 Hermes 工具解析器时, chat completion 能否返回 prompt_token_ids / token_ids(return_token_ids);
3. 用返回的 assistant 消息拼出下一轮请求时, 下一轮 prompt_token_ids 是否以"上一轮 prompt + 生成 token"开头;
4. 每轮请求的耗时, 用来估算正式实验的 GPU 额度。

这正是 agent-lightning 训练代理的链路: 每轮由服务端渲染 chat template 并分词。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PORT = 8000
OUT = "/kaggle/working/feasibility.json"


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout


def post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def write_report(report: dict) -> None:
    with open(OUT, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))


def main() -> None:
    report: dict = {"gpu": run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv"])}
    started = time.time()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)
    report["vllm_version"] = run([sys.executable, "-c", "import vllm; print(vllm.__version__)"]).strip()
    report["install_seconds"] = round(time.time() - started)

    log = open("/kaggle/working/vllm_server.log", "w")  # noqa: SIM115 服务进程整个运行期间都在写日志
    server = subprocess.Popen(
        [
            "vllm",
            "serve",
            MODEL,
            "--revision",
            REVISION,
            "--dtype",
            "half",
            "--max-model-len",
            "8192",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--port",
            str(PORT),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        env={**os.environ},
    )
    started = time.time()
    while time.time() - started < 900:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
            break
        except Exception:
            if server.poll() is not None:
                break
            time.sleep(5)
    report["server_ready_seconds"] = round(time.time() - started)
    report["server_alive"] = server.poll() is None
    if not report["server_alive"]:
        write_report(report)
        return

    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "List the files in the current directory using the bash tool."}]
    base = {"model": MODEL, "tools": tools, "temperature": 1.0, "return_token_ids": True, "seed": 0}
    started = time.time()
    first = post("/v1/chat/completions", {**base, "messages": messages, "max_tokens": 256})
    report["first_request_seconds"] = round(time.time() - started, 2)
    choice = first["choices"][0]
    report["response_top_level_keys"] = sorted(first)
    report["choice_keys"] = sorted(choice)
    prompt_ids = first.get("prompt_token_ids") or choice.get("prompt_token_ids") or []
    token_ids = choice.get("token_ids") or []
    report["prompt_tokens"], report["generated_tokens"] = len(prompt_ids), len(token_ids)
    report["assistant_message"] = choice["message"]

    assistant = {k: v for k, v in choice["message"].items() if k in ("role", "content", "tool_calls") and v is not None}
    follow = [*messages, assistant]
    if assistant.get("tool_calls"):
        follow.append(
            {"role": "tool", "tool_call_id": assistant["tool_calls"][0]["id"], "content": "README.md\nsetup.py"}
        )
    else:
        follow.append({"role": "user", "content": "Continue."})
    second = post("/v1/chat/completions", {**base, "messages": follow, "max_tokens": 1})
    next_prompt = second.get("prompt_token_ids") or second["choices"][0].get("prompt_token_ids") or []
    context = prompt_ids + token_ids
    report["next_prompt_tokens"] = len(next_prompt)
    report["prefix_ok"] = next_prompt[: len(context)] == context
    report["first_divergence"] = next(
        (i for i, (a, b) in enumerate(zip(context, next_prompt, strict=False)) if a != b), None
    )

    server.terminate()
    write_report(report)


if __name__ == "__main__":
    main()
