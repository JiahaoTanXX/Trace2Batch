# Trace2Batch

[中文](README.md) | English

[![CI](https://github.com/JiahaoTanXX/Trace2Batch/actions/workflows/ci.yml/badge.svg)](https://github.com/JiahaoTanXX/Trace2Batch/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Diagnosing, fixing and measuring token consistency in multi-turn agentic RL training data.** Trace2Batch follows the problem from the chat template through a real vLLM server and the training cost to real GRPO training. It uses 900 public agent trajectories, more than 100k merge attempts and real GPU runs to answer a question that every multi-turn agentic RL stack using trajectory aggregation has to face:

> Merging a whole trajectory into one training sample (`trajectory` aggregation) avoids recomputing the shared history again and again, but only if each turn's prompt tokens start with the previous turn's context. **Does that assumption silently break in a real pipeline? What breaks it? How much extra training does it cost? And how do you fix it for good?**

Common training frameworks (agent-lightning, for example) report only *how many* merges failed. Trace2Batch points to the first diverging token, explains *why* it diverged, and ships fixes validated on a real inference server.

## Key results

| | Result |
|---|---|
| 🔍 **Diagnosis** | 54,391 of 107,848 merge attempts fail, and **every one** traces back to the chat template rewriting history. Qwen3 with thinking disabled merges **0%** of the time and inflates training tokens up to **15.75×** |
| 🧪 **Real sampling** | Even with a prefix-stable template, only 60.4% of really sampled turns keep the prefix, mostly because of non-canonical token splits |
| 🩹 **Template patches** | Prefix-stable patches take merging from **0% → 100%** on canonical outputs. Patch v2 removes every stacked-newline failure on a real vLLM server, lifting prefix retention from 74.4% to **94.7%** |
| ✅ **TITO** | In multi-turn rollouts on a real vLLM server, token-in-token-out keeps **528 / 528** trajectories fully mergeable; server re-rendering manages 63.6% / 77.3% |
| 🔬 **New finding** | TITO is not free: the model continues its own non-canonical splits from context. Once a trajectory has had one, later turns repeat it **10.5 / 24.8 points** more often than under re-render-driven rollouts (both 95% CIs above 0) |
| 🎯 **Real GRPO** | In multi-turn tool-use GRPO with live vLLM rollouts and a LoRA adapter hot-loaded every step (4 runs, 15,360 rollouts), both collection paths lift held-out success from 3–7% to about **78%** (all 95% CIs above 0). Re-rendering splits each rollout into **2.05–2.07** rows and spends **1.24–1.71×** the update time. Within 60 steps, non-canonical splits were not reinforced; they fell in all 4 runs |
| ⏱️ **Training cost** | In a real LoRA training loop, failed merges make training **1.5–4.2×** slower, reaching **4.0–5.2×** at 8 turns. The gap is almost entirely explained by token inflation (r = 0.987). With memory-efficient attention the benchmark extends to **16k** context on a single T4 (peak memory < 8 GiB), where 8-turn SWE-smith trajectories are still **4.1–4.6×** slower |
| ✂️ **Truncation** | At the same per-row length, dynamic budget allocation raises policy-token retention from 26.3% to **54.6%** |
| 🌐 **Upstream** | Reported as microsoft/agent-lightning issue [#645](https://github.com/microsoft/agent-lightning/issues/645); the maintainer confirmed seeing the same phenomenon in their own experiments |

## 1. Diagnosis: chat templates silently break the prefix

We use 5 real tokenizers from **4 model families** (Qwen, DeepSeek, GLM, gpt-oss) with 9 template configurations, and 900 trajectories from 3 public agent datasets. On these we simulate the common "OpenAI-compatible proxy + vLLM `return_token_ids`" collection path, **107,848 merge attempts** in total:

| Configuration | SWE-smith | Toucan | SWE-agent (nebius) | Root cause |
|---|---|---|---|---|
| Qwen2.5 / DeepSeek-V3.1 / GLM-4.5, no thinking | 100% | 100% | 100% | — (prefix-stable template) |
| **Qwen3, thinking disabled** | **0%** | **0%** | **0%** | The empty `<think>` block is dropped from history |
| Qwen3 / GLM-4.5 thinking, reasoning sent back [synthetic] | 100% | 100% | **0%** | Observations arrive as `user` messages, so history reasoning is still dropped or emptied |
| Qwen3 / GLM-4.5 thinking, reasoning not sent back [synthetic] | **0%** | **0%** | **0%** | History reasoning is dropped or emptied |
| **gpt-oss** | 100% | 100% | **0%** | The end-of-generation token `<|return|>` becomes `<|end|>` in history |

(Values are merge success rates.)

- **All 54,391 failures are attributed to `template_mismatch`**, and every changed segment can be inspected.
- When merging fails, it fails on every turn. Samples degrade to one row per turn, and each row carries the full history again. Without truncation, token volume grows **15.75×** (SWE-smith), **14.75×** (SWE-agent) and **1.75×** (Toucan).
- The templates also turn out to be fragile with respect to data format:
  - the gpt-oss template raises on tool definitions whose `description` is `null`;
  - the GLM-4.5 template raises on invalid JSON arguments;
  - gpt-oss renders only the first tool call of each message.

**Takeaway:** whether trajectory aggregation works depends on the combination of template, whether reasoning is sent back, the role used for tool observations, and whether a final answer appears. It does not depend on the data itself. With the wrong combination, training data silently degrades to per-turn samples, and the existing metrics cannot tell you why.

A typical divergence (Qwen3, `enable_thinking=False`):

```text
previous context: …<|im_start|>assistant\n<think>\n\n</think>\n\nI'll help you implement the necessary changes…
next prompt:      …<|im_start|>assistant\nI'll help you implement the necessary changes…
attribution: template_mismatch, change = removed "<think>\n\n</think>\n\n"
```

## 2. Real sampling: a prefix-stable template is not enough

Canonical outputs are the ideal case. Here the model **actually samples** every turn on real history; each output is parsed with vLLM's Hermes tool parser, and the next prompt is re-rendered from the parsed messages:

- **CPU reproduction** (Qwen2.5-0.5B-Instruct, T=1.0, 300 turns): the Qwen2.5 template merges 100% on canonical outputs, but only **60.4%** of finished turns keep the prefix under real sampling.
  - The main cause is **non-canonical token splits** (32.5%): the model samples `-t` + `xt`, while re-tokenizing the same text gives `-txt`.
  - Next come template-level rewrites (6.5%): tool-call JSON is reformatted, generations end with `<|endoftext|>`, and newlines stack.
  - **Temperature ablation:** with the recommended sampling settings (T=0.7), prefix retention rises to 88.7%, but template-level failures stay at about 6.5%. Non-canonical splits come mostly from high-temperature sampling; reformatting does not depend on temperature.
- **Real vLLM server** (Kaggle T4, `vllm serve` with Hermes parsing and `return_token_ids`, prompts up to 32k): prefix retention is 83.3% for Qwen2.5-1.5B (250/300) and 74.4% for Qwen3-0.6B with patch v1 (212/285).
  - **The dominant failure is an interaction bug between the parser and the template.** The model writes `text\n<tool_call>` in the canonical format; the Hermes parser keeps the trailing `\n` in `content`; the template then inserts another `\n` when re-rendering, giving `\n\n`. This accounts for 44% and 85% of the failures.

**Takeaway:** any pipeline that re-renders and re-tokenizes on the server every turn degrades trajectory aggregation frequently, even when the template itself is prefix-stable.

## 3. Fix: prefix-stable template patches

We wrote **prefix-stable template patches** for Qwen3, gpt-oss and Qwen2.5 (`experiments/template_patch/templates/`). Each is a minimal change to the official template that renders history exactly as it was generated. Patch v2 inserts a newline only when `content` does not already end with one, which removes the stacked-newline bug described above.

| Validation | Official template | Patched template |
|---|---:|---:|
| Canonical outputs: Qwen3, thinking disabled, 3 datasets | 0% | **100%** |
| Canonical outputs: gpt-oss, SWE-agent | 0% | **100%** |
| Real sampling (CPU): Qwen3-0.6B, thinking disabled, 150 turns | 0% | 78.7% (v1) → **90.0%** (v2) |
| Real vLLM: Qwen2.5-1.5B, 300 turns | 83.3% | **90.7%** (v2) |
| Real vLLM: Qwen3-0.6B, thinking disabled | — | 74.4% (v1, 285 turns) → **94.7%** (v2, 300 turns) |

- On real vLLM, patch v2 cuts stacked-newline failures from 23 / 64 to **0**.
- Paired turn by turn, 25 Qwen2.5 turns flip from failing to passing and 3 flip back. All 3 had different samples in the two runs. For Qwen3 the counts are 58 and 1.
- On canonical outputs, the patched training rows **match token-in-token-out exactly** on every dataset.
- The remaining failures cannot be removed by any template while the server keeps re-rendering. They are non-canonical splits and tool-call reformatting such as `-74.0060` → `-74.006`.

## 4. The real fix: TITO validated end to end on a real vLLM server

Token-in-token-out (TITO) reuses the existing token ids for all history and tokenizes only the new messages. The project ships a dependency-free reference implementation (`trace2batch/tito.py`). It is validated end to end on a **real vLLM 0.32 server** (`experiments/tito_vllm/`):

- **TITO-driven rollouts:** `/v1/completions` takes token ids directly, and observation tokens are rendered by the server's `/tokenize`. Each trajectory starts at its first assistant turn and runs for up to 8 turns, with every turn sampled by the model (T=1.0).
- **Paired comparison:** the same samples are also fed to server re-rendering with the official template and with patch v2. All three paths are compared turn by turn, so run-to-run randomness drops out.

| Trajectories fully mergeable | **TITO** | Re-render + official | Re-render + patch v2 |
|---|---:|---:|---:|
| Qwen2.5-1.5B (264 trajectories, 1,360 turns) | **100%** | 62.1% | 63.6% |
| Qwen3-0.6B, thinking disabled (264 trajectories, 1,482 turns) | **100%** | 0% | 77.3% |

- **A per-turn prefix rate of 85–89% compounds into many broken trajectories over multiple turns.** Only TITO reliably turns every trajectory into a single training row.
- Each re-rendering failure adds another row that carries the full history. Training tokens are **1.07–1.80×** those of TITO with patch v2, and up to **5.0×** with Qwen3's official template.
- The attribution is clean: under patch v2, **every** remaining retoken failure diverges inside the model's own sample for that turn (Qwen2.5 154 / 154, Qwen3 121 / 121).
- The client's Hermes parsing was checked against vLLM's own parser: of the 475 checks where both calls produced the same sample, 474 parse identically.
- In earlier offline validation on canonical outputs for 900 trajectories, TITO matches server re-rendering token for token (12,884 / 12,884 prompts).
- **The inference is confirmed by measurement.** Driving the same trajectories with real server re-rendering gives fully mergeable rates of 63.0% (Qwen2.5, 300 trajectories) and 77.3% (Qwen3, 264 trajectories). That is within 2 points of the paired inference above. With SWE-smith filled in, TITO keeps 300 / 300 (Qwen2.5) and 299 / 299 (Qwen3) trajectories fully mergeable.

### The price of TITO: the model continues its own non-canonical splits

The full run showed a pattern: once a trajectory had one non-canonical split, later turns produced more of them. This could come from the content itself, or from TITO keeping the model's own non-canonical tokens in context. To tell the two apart, we reran the same trajectories with the same seeding rule, but with server re-rendering driving the rollouts. Re-rendering re-tokenizes the context canonically every turn, so the model never sees its own non-canonical tokens. The decision rule was fixed before the run.

| Per-turn non-canonical split rate | No earlier split | After an earlier split |
|---|---:|---:|
| Qwen2.5-1.5B, TITO-driven | 11.5% | **25.7%** |
| Qwen2.5-1.5B, re-render-driven | 11.0% | 15.2% |
| Qwen3-0.6B, TITO-driven | 4.9% | **29.2%** |
| Qwen3-0.6B, re-render-driven | 4.7% | 4.4% |

- After an earlier split, TITO minus re-rendering is **+10.5 points [+4.6, +16.6]** for Qwen2.5 and **+24.8 points [+14.5, +35.7]** for Qwen3. Both 95% CIs come from a trajectory-level bootstrap and lie entirely above 0.
- The control is clean: with no earlier split, the two drivers are nearly identical (+0.4 and +0.1 points, both CIs include 0). Qwen3 shows no clustering at all under re-rendering (4.7% → 4.4%).
- **Takeaway:** TITO guarantees token-exact consistency between training and inference. The price is that the model sees and continues its own non-canonical splits, which shifts the sampling distribution. Re-rendering avoids this effect, but at the cost of a train/inference mismatch and broken merges. This project quantifies that trade-off in multi-turn agentic RL data collection on a real server. Whether RL training amplifies it is tested in [Section 7](#7-real-grpo-training-how-the-collection-path-affects-learning-and-cost-p3-b).

## 5. The cost: how much slower does training get? (P3-A)

The [training-cost benchmark](experiments/train_cost/README.md) measures both representations in a real training loop (Kaggle T4, Qwen3-0.6B + LoRA, masked cross-entropy):

- both representations are built from **the same** token events, and the full context of every supervised target is checked;
- before timing, the loss and every trainable gradient are checked to agree between the two representations;
- both use the same LoRA initialization, one update per rollout, policy-token normalization and AB/BA ordering.

- Falling back to per-turn rows makes training **1.5–4.2×** slower per cell, and up to **5.1×** for a single rollout.
- **The number of turns is what matters.** At fixed length, the speedup rises monotonically with turns, from about 1.0–1.7× at 2 turns to about **4.0–5.2×** at 8 turns. Two independent samples agree.
- The speedup tracks the processed-token ratio closely (r = **0.987**) and comes to about 0.82× that ratio.
- Peak memory is essentially the same for both representations. Merging saves **time**, not memory.

### Extending to 16k context

Real SWE-style trajectories often exceed 8k tokens. On a T4 the default path falls back to the math attention kernel and computes vocabulary logits over the whole row, so peak memory already reaches 10.5 GiB at about 6k. Stage 3 (`--attention sdpa_repeat_kv --policy-logits-only`) makes two changes:
- it repeats the KV heads explicitly and forces the memory-efficient kernel;
- it computes logits only at supervised positions.

Neither change alters the loss or gradients (relative L2 on the order of 1e-6 on the real model). On the same 5k sample, memory drops from 7.56 GiB to 2.95 GiB, and the two representations run 3.8× and 5.1× faster. With this path, turns are fixed at k = 4 / 6 / 8 across four windows (4–6k, 6–8k, 8–12k, 12–16k): 18 cells and 107 rollouts from SWE-smith and nebius.

| SWE-smith speed ratio | 4–6k | 6–8k | 8–12k | 12–16k |
|---|---:|---:|---:|---:|
| k = 4 | 2.11 | 1.82 | 1.89 | 1.75 |
| k = 6 | 3.52 | 3.51 | 2.86 | 2.86 |
| k = 8 | 4.50 | 4.39 | 4.55 | 4.13 |

- **Turns still decide.** In all 7 dataset × window combinations, the speedup rises monotonically with the number of turns.
- **Length weakens it only slightly.** Under the pre-registered rule all five slopes are negative, but from 4–6k to 12–16k the drop is only 8–19%, and most groups are not monotonic.
- **The mechanism matches the prediction.** The ratio of the time gain to the token gain falls from a per-window median of 0.89 to 0.84. When the speedup is fit as a mix of the token ratio and the squared-cost ratio, the squared term's weight rises from 0.43 to 0.70, because attention's quadratic cost takes a larger share as rows grow.
- Peak memory at 16k context stays at or below 4.7 GiB. The highest cell, 7.5 GiB, is the one with the most policy tokens in a single row (3,366).

## 6. Truncation: the real constraint is per-row length

About **80% of a merged response is tool observations**, so they consume most of the length budget. When the budget overflows, default tail truncation drops the late generations closest to the reward. We added `truncation_strategy` (`truncate` / `split_on_budget` / `drop`) and `max_total_length`:

| Dataset | Per-row length | Policy-token retention (fixed halves → dynamic) | Trajectories losing their last generation (fixed → dynamic) |
|---|---:|---|---|
| SWE-smith | 16,384 | 26.3% → **54.6%** | 75.0% → **41.3%** |
| SWE-smith | 32,768 | 57.2% → **83.0%** | 36.7% → **14.3%** |
| SWE-agent | 16,384 | 55.0% → **73.5%** | 54.0% → **36.3%** |
| SWE-agent | 32,768 | 77.8% → **99.4%** | 31.7% → **2.0%** |

- At equal per-row length, splitting at turn boundaries **never** retains more policy tokens than truncation (derived and confirmed), because a token beyond the per-row limit does not fit in any row.
- What splitting buys is rows that end at turn boundaries: 0% of rows end with only an observation. Of the extra tokens it adds, 62.6%–67.7% are repeated prefix. If the shared prefix is computed only once, each retained policy token costs 6.4–6.5 tokens, at or below truncation's 6.6–17.4.

## 7. Real GRPO training: how the collection path affects learning and cost (P3-B)

The earlier sections answer data- and cost-level questions. P3-B plugs both collection paths into **real multi-turn tool-use GRPO training** (`experiments/grpo/`) and compares them during training:

- **TITO**: `/v1/completions` takes token ids, and each trajectory merges into a single training row;
- **Server re-rendering**: `/v1/chat/completions` with the patch v2 template; rows are merged by prefix and split wherever merging fails.

| Item | Setting |
|---|---|
| Environment | A deterministic, standard-library-only inventory environment: `lookup` returns price and stock, `calculate` does arithmetic (safe `ast` evaluation), and an exact final answer scores 1, with no model judge. Tool results come from the environment live, so training is strictly on-policy |
| Model and algorithm | Qwen3-0.6B (thinking off) + LoRA r16; GRPO with 8 trajectories per task, advantage = reward − group mean, one update per batch, normalized by policy tokens |
| Scale | 60 steps × 8 tasks per step × 8 trajectories; 2 seeds per arm, 4 runs and 15,360 rollouts in total; 200 held-out tasks evaluated at steps 0 / 20 / 40 / 60 |
| Hardware | Kaggle 2×T4: one GPU serves vLLM 0.32 for rollouts, the other trains LoRA, and the adapter is hot-loaded back into vLLM every step |

The decision rules were fixed before the runs. Results:

| | TITO s0 / s1 | Re-render s0 / s1 |
|---|---|---|
| Held-out success (step 0 → 60) | 0.07 → 0.78 / 0.065 → 0.785 | 0.03 → 0.78 / 0.03 → 0.775 |
| 95% CI of the gain | [+0.65, +0.77] / [+0.66, +0.79] | [+0.69, +0.81] / [+0.69, +0.81] |
| Training rows per rollout | 1.00 / 1.00 | 2.07 / 2.05 |
| Tokens in the updates | 0.74M / 0.80M | 1.23M / 0.99M |
| Total update time | 569 s / 638 s | 971 s / 793 s |
| Per-turn non-canonical split rate (first 10 → last 10 steps) | 2.5% → 1.2% / 3.2% → 0.7% | 3.6% → 0.4% / 3.1% → 2.3% |

- **Both paths learn, and equally well.** All 4 runs improve significantly and end near 0.78, with nearly overlapping learning curves.
- **The cost falls on re-rendering only.** Re-rendering splits each rollout into 2.05–2.07 rows; for the same learning, it puts **1.23–1.66×** the tokens through the updates and spends **1.24–1.71×** the update time.
- **Training does not reinforce non-canonical splits.** Section 4 found that TITO makes the model continue its own non-canonical splits. Under the pre-registered rule (both TITO seeds' slopes above 0 and above both re-render slopes), **no reinforcement was observed within 60 steps**: all 4 slopes are negative.
- **The trainer is numerically consistent.** Every 10 steps the same batch is also run backward as per-turn rows; its gradients match the merged rows (cosine ≥ 0.999, relative L2 ≤ 0.038), and the time ratio tracks the token ratio, consistent with P3-A. The per-step median gap between vLLM's sampling logprobs and the trainer's recomputed logprobs is 0.
- **Two things reported as they are:**
  - only the total-cost tasks were learned (0.04–0.09 → 0.98–0.99); the shortage tasks (42 of the 200 held-out tasks) stayed at 0 throughout, as the policy converged on the majority answer type;
  - TITO's advantage is on the training side only. In this implementation TITO makes one extra `/tokenize` request per turn, so its rollouts are slower, and end-to-end time over 60 steps was faster for one seed and slower for the other.

## Upstream

Building on Section 1, we filed issue [#645](https://github.com/microsoft/agent-lightning/issues/645) on [microsoft/agent-lightning](https://github.com/microsoft/agent-lightning). It points out that the mismatch-cause fields are always `False`, and includes reproduction data and an attribution scheme. The maintainer replied that they had seen the same phenomenon in their early experiments, and that this is why they ship a dedicated chat template in their coding-agent example.

## Toolkit

- **Mismatch classifier** (`trace2batch/mismatch.py`): finds the first diverging token and classifies the failure as `template_mismatch`, `retoken_mismatch` or `others_mismatch`. It also returns the text around the divergence for inspection. It uses only the standard library, and the decode function is injected as a callback, so any tokenizer works.
- **TITO recorder** (`trace2batch/tito.py`): dependency-free; its events go straight into the sample builder.
- **Sample builder** (`trace2batch/core.py`): turns model requests, tool observations and reward events into RL rows (prompt / response / policy mask / reward). It supports:
  - `transition` and `trajectory` representations, with three truncation strategies;
  - multi-rollout batching and a total-length budget;
  - missing-reward filling and diagnostics for "no usable rows".

  No runtime dependencies.
- **Real-server clients** (`experiments/vllm_sampling/`, `experiments/tito_vllm/`): talk to `vllm serve` directly, for single-turn sampling and for paired multi-turn comparison of TITO and re-rendering.
- **Training-cost benchmark** (`experiments/train_cost/`): a real LoRA training loop that checks gradients agree before timing, with time-budgeted runs and a Kaggle submission flow.
- **Multi-turn tool-use GRPO** (`experiments/grpo/`): a standard-library-only verifiable environment, TITO and re-render rollout drivers, group advantages and the GRPO update, a LoRA adapter hot-loaded into vLLM every step, and held-out evaluation broken down by answer type.

## Setup and reproducibility

| Dimension | Details |
|---|---|
| Data | SWE-smith (structured tool calls), Toucan-1.5M Qwen3 subset (MCP tool calls), nebius SWE-agent (text actions) |
| Tokenizers | Qwen2.5, Qwen3, DeepSeek-V3.1, GLM-4.5, gpt-oss; 9 template configurations (synthetic-reasoning configurations are labeled) |
| Collection paths | token passthrough (control) / vLLM token ids (server re-render every turn) / text re-tokenization, which separates template rewrites from token-boundary effects |
| Real models | Qwen2.5-0.5B / 1.5B-Instruct and Qwen3-0.6B, with pinned revisions |
| Hardware | Diagnosis and truncation studies run on CPU; real-server and training experiments run on Kaggle T4 (2×T4 for GRPO) |

- Dataset and tokenizer revisions and sampling seeds are pinned, and sample ids are saved, so every study reruns with a single command. Repeated runs give identical merge and attribution results.
- Attribution counts are cross-checked one by one against the sample builder's mismatch counts.
- Testing and CI: offline tests by default, plus opt-in real-tokenizer tests; ruff and pyright; GitHub Actions on Python 3.12 / 3.13.
- Full reports and result data will be published once the project is complete. Every study can be rerun locally with the commands below; results go to `results/` in each experiment directory.

## Limitations

- Section 1 uses canonical template outputs and does not cover model families such as Llama or Mistral.
- The real-sampling experiments use small models (0.5B–1.5B) in fp16 on T4, with one run per configuration.
- In the multi-turn rollouts, tool observations come from the dataset and are therefore off-policy. The clustering comparison is observational, because the two drivers' samples stop being paired after their first divergence. Qwen3's re-render-driven run completed only 64 SWE-smith trajectories.
- The patches deliberately depart from Qwen3's official advice not to keep reasoning in history; their effect on model quality has not been evaluated.
- The training-cost benchmark is a masked-CE + LoRA reference loop, not full-parameter training. The truncation study is a data-level comparison. The 16k extension uses a different compute path, so its speed ratios are compared only within that study; it has 6 rollouts per cell and one seed, and only SWE-smith reaches 12–16k.
- The GRPO study uses a small purpose-built environment and a 0.6B model, so its conclusions are limited to that environment. It has 2 seeds per arm, 60 steps, no KL term, and fp16 on T4. The shortage tasks were not learned.

## Quick start

Requires Python >= 3.12.

```bash
pip install -e ".[dev]"
python -m examples.trace2batch.run            # 12 synthetic scenarios with hand-computed expectations

pip install -e ".[real]"
python -m experiments.template_prefix.run     # prefix-consistency study (first run downloads tokenizers and dataset shards)
python -m experiments.truncation.run          # truncation strategies, budget allocation and shared-prefix cost

pip install -e ".[sampling]"
python -m experiments.real_sampling.run       # real model sampling (downloads the ~1 GB 0.5B model)
python -m experiments.template_patch.run      # template patch validation

# Needs a running vLLM server (GPU; vllm serve ... --enable-auto-tool-choice --tool-call-parser hermes)
python -m experiments.vllm_sampling.run --model qwen3-0.6b+patch2 --base-url http://127.0.0.1:8000   # single-turn sampling
python -m experiments.tito_vllm.run --model qwen3-0.6b --base-url http://127.0.0.1:8000             # multi-turn TITO (server also needs --trust-request-chat-template)

# GRPO (needs 2 GPUs and pip install -e ".[train]"; the vLLM server also needs --enable-lora --max-lora-rank 16 and VLLM_ALLOW_RUNTIME_LORA_UPDATING=True)
python -m experiments.grpo.run train --driver tito --difficulties 3 --tasks-per-step 8 --train-device cuda:1 --out results/grpo_tito
```

Building samples:

```python
from trace2batch.core import build_batch, build_samples

events = [{"event_type": "model_request", "data": {"prompt_token_ids": [1], "response_token_ids": [10, 11]}}]
result = build_samples(events, mode="trajectory", max_prompt_length=16, max_response_length=16)
print(result.summary())

batch = build_batch({"r1": events, "r2": events}, mode="trajectory", max_prompt_length=16, max_response_length=16)
print([item.rollout_id for item in batch.rows], batch.summary()["n_rollouts"])
```

Attributing a mismatch:

```python
from trace2batch.mismatch import explain_mismatch

report = explain_mismatch(
    prev_context_ids, next_prompt_ids, decode=lambda ids: tok.decode(ids, skip_special_tokens=False)
)
print(report.kind, report.changed_segments)
```

Recording with TITO:

```python
from trace2batch.tito import TokenInTokenOut

tito = TokenInTokenOut(initial_prompt_ids)
tito.record_response(sampled_ids)  # sampled tokens enter the context unchanged
tito.add_observation(observation_ids)  # only new messages are tokenized
events = tito.events()  # ready for build_samples
```

## Sample-building semantics

| Feature | Behavior |
|---|---|
| Per-turn `transition` | Each valid model response becomes one row; generated tokens have mask 1 |
| Continuous `trajectory` | Merged by exact token prefix; a new row starts when the prefix breaks |
| Tool observations | Kept as context inside the merged response, with mask 0 |
| Length limits | Response and mask are truncated together; optional `max_total_length` budget; over-long prompts are truncated and marked as dropped |
| Truncation strategy | `truncate` (default) / `split_on_budget` (start a new row at a turn boundary) / `drop` (drop the row) |
| Batching `build_batch` | Every row carries its `rollout_id`; rollouts are never merged with each other; rollout-level diagnostics are aggregated |
| Reward | Uses the last reward and distinguishes a zero reward from a missing one; missing rewards are filled with `reward_fillna_value` |

Response and mask after merging three tool-calling turns:

```text
response: [10, 90, 11, 91, 92, 12, 13]
mask:     [ 1,  0,  1,  0,  0,  1,  1]
```

## Tests

```bash
python -m pytest -q                 # offline by default
python -m pytest -q --run-network   # also run real-tokenizer tests
ruff check . && ruff format --check . && pyright
```

## Layout

```text
Trace2Batch/
├── trace2batch/
│   ├── core.py                 # sample building, batching and diagnostics
│   ├── mismatch.py             # locating and attributing prefix mismatches
│   └── tito.py                 # token-in-token-out trajectory recorder
├── experiments/template_prefix/
│   ├── run.py                  # real-tokenizer prefix-consistency study
│   └── DATA.md                 # data notes
├── experiments/real_sampling/  # real model sampling (CPU)
├── experiments/template_patch/ # prefix-stable template patches and validation
├── experiments/tito/           # TITO observation tokenization and canonical-output validation
├── experiments/vllm_sampling/  # real vLLM server sampling client (GPU)
├── experiments/tito_vllm/      # real vLLM multi-turn rollouts: TITO vs re-rendering (GPU)
├── experiments/train_cost/     # P3-A training-cost benchmark: data pairing, masked CE, time-budgeted runs
├── experiments/grpo/           # P3-B multi-turn tool-use GRPO: environment, rollout drivers, training loop, evaluation
├── experiments/truncation/
│   └── run.py                  # truncation strategy comparison
├── examples/trace2batch/       # synthetic scenarios with hand-computed expectations
├── tests/                      # offline + network tests
└── .github/workflows/ci.yml
```

## License

MIT; see [LICENSE](LICENSE). Some files, such as `trace2batch/core.py`, are derived from MIT-licensed open-source code; the original copyright notices are kept in the file headers and in LICENSE.
