# Copyright (c) 2026 Jiahao Tan

"""Kaggle 上运行 experiments.vllm_sampling 的驱动脚本(2 块 T4)。

推送前把 JOBS 和 COMMIT 改成本次要跑的内容; 每个 GPU 依次运行分给它的任务, 两个 GPU 并行。
每个任务: 启动一个 vLLM 服务(Hermes 工具解析, return_token_ids) -> 运行采样客户端 -> 关闭服务。
结果写到 /kaggle/working/results/, 运行记录写到 /kaggle/working/run_log.json。
虚拟环境和模型缓存都放在 /tmp, 不会被保存为输出。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

COMMIT = "REPLACE_WITH_COMMIT"
# (GPU 编号, 模型, 采样设置, 每个数据集的轮次)
JOBS: list[tuple[int, str, str, int]] = []
REPO = "https://github.com/JiahaoTanXX/Trace2Batch.git"
VLLM_VERSION = "0.32.0"
VENV = "/tmp/vllm-venv"
SRC = "/tmp/Trace2Batch"
OUT = "/kaggle/working"
ENV = {**os.environ, "TRACE2BATCH_HF_CACHE": "/tmp/hf/hub", "HF_HOME": "/tmp/hf"}
log: dict = {"commit": COMMIT, "jobs": []}


def sh(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kwargs)


def setup() -> None:
    started = time.time()
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    sh(["uv", "venv", "--python", "3.12", VENV])
    sh(["uv", "pip", "install", "-q", "--python", f"{VENV}/bin/python", f"vllm=={VLLM_VERSION}", "pyarrow"])
    sh(["git", "clone", "-q", REPO, SRC])
    sh(["git", "-C", SRC, "checkout", "-q", COMMIT])
    log["setup_seconds"] = round(time.time() - started)
    log["gpu"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True
    ).stdout


def run_job(gpu: int, model: str, sampling: str, turns: int) -> None:
    sys.path.insert(0, SRC)
    from experiments.vllm_sampling.run import MODELS

    served = MODELS[model]
    port = 8000 + gpu
    record: dict = {"gpu": gpu, "model": model, "sampling": sampling, "turns_per_dataset": turns}
    cmd = [
        f"{VENV}/bin/vllm", "serve", served.repo, "--revision", served.revision, "--served-model-name", served.key,
        "--dtype", "half", "--max-model-len", "32768", "--gpu-memory-utilization", "0.85",
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--port", str(port),
    ]  # fmt: skip
    if served.chat_template:
        cmd += ["--chat-template", f"{SRC}/{served.chat_template}"]
    started = time.time()
    with open(f"{OUT}/vllm_{model}_{sampling}.log", "w") as server_log:
        server = subprocess.Popen(
            cmd, stdout=server_log, stderr=subprocess.STDOUT, env={**ENV, "CUDA_VISIBLE_DEVICES": str(gpu)}
        )
        while time.time() - started < 1200:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
                break
            except Exception:
                if server.poll() is not None:
                    break
                time.sleep(5)
        record["server_ready_seconds"] = round(time.time() - started)
        if server.poll() is None:
            started = time.time()
            client = subprocess.run(
                [
                    f"{VENV}/bin/python",
                    "-m",
                    "experiments.vllm_sampling.run",
                    "--model",
                    model,
                    "--sampling",
                    sampling,
                    "--turns-per-dataset",
                    str(turns),
                    "--base-url",
                    f"http://127.0.0.1:{port}",
                    "--out",
                    f"{OUT}/results/{model}@{sampling}",
                ],
                cwd=SRC,
                env=ENV,
                capture_output=True,
                text=True,
            )
            record["client_seconds"] = round(time.time() - started)
            record["client_returncode"] = client.returncode
            record["client_stdout_tail"] = client.stdout[-3000:]
            record["client_stderr_tail"] = client.stderr[-3000:]
            server.terminate()
            server.wait(timeout=120)
        else:
            record["server_failed"] = True
    log["jobs"].append(record)
    print(json.dumps(record, ensure_ascii=False, indent=1), flush=True)


def main() -> None:
    setup()
    threads = []
    for gpu in sorted({job[0] for job in JOBS}):
        jobs = [job for job in JOBS if job[0] == gpu]
        thread = threading.Thread(target=lambda jobs=jobs: [run_job(*job) for job in jobs])
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    with open(f"{OUT}/run_log.json", "w", encoding="utf-8") as file:
        json.dump(log, file, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
