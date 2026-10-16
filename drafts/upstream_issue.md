<!-- Posted unchanged as https://github.com/microsoft/agent-lightning/issues/645 on 2026-10-16 after owner review. -->

# Trajectory aggregation: classify prefix mismatches (`template_mismatch` / `retoken_mismatch` are always `False`)

## Problem

In `agentlightning/verl/rollout_adapter.py` (checked at `69f5a40`), trajectory-level aggregation records every token-prefix failure as:

```python
# Token-prefix failures are classified as other mismatches.
"template_mismatch": False,
"retoken_mismatch": False,
"others_mismatch": True,
```

So the wandb table `training/trace_merge_mismatches` can't tell a user *why* their trajectories stopped merging. With the training proxy (`return_token_ids=True`), vLLM re-renders the whole conversation through the chat template on every turn. If the template rewrites earlier turns, every merge fails, and `trajectory` falls back to one row per turn. This shows up in `training/n_unmerged_rollouts`, but no cause is attached.

I searched issues and PRs for `retoken`, `template_mismatch` and `others_mismatch` and found nothing. #469 (mRoPE multimodal prefix checks) and #613 (log-prob preservation) are related, but neither touches the mismatch classification.

## Evidence

I simulated the proxy + vLLM path offline with real tokenizers (tokenizer files only, no weights). The data was 900 public multi-turn agent trajectories: 300 each from SWE-smith (structured tool calls), Toucan-1.5M (Qwen3 subset) and nebius SWE-agent (text actions, observations as `user`). That gives 11,984 merge attempts per tokenizer configuration. Each turn's prompt is `apply_chat_template(messages[:k], add_generation_prompt=True)`; each response is the template-canonical tokens of the assistant message. The events go through the same prefix check as `RolloutAdapter`. A token-concatenation control merges 100% in every configuration.

| Tokenizer config | SWE-smith | Toucan | nebius |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 100% merged | 100% merged | 100% merged |
| Qwen3-8B, `enable_thinking=False` | **0% merged** | **0% merged** | **0% merged** |
| Qwen3-8B, thinking, client sends back `reasoning_content` (synthetic reasoning) | 100% merged | 100% merged | **0% merged** |
| Qwen3-8B, thinking, client drops `reasoning_content` (synthetic reasoning) | **0% merged** | **0% merged** | **0% merged** |

Every failure has the same cause. The Qwen3 template (revision `b968826`) only renders `<think>…</think>` for an assistant turn that is `loop.last` or carries `reasoning_content` *and* comes after the last real user query. So:

- With `enable_thinking=False`, the generation prompt ends in an empty `<think>\n\n</think>\n\n` block, and that block disappears once the turn becomes history.
- When observations come back as `user` messages (SWE-agent style), `last_query_index` moves forward every turn, so the reasoning gets stripped even when it is sent back.

When merging fails, rows per trajectory equal the number of turns (19.05 on SWE-smith, 21.49 on nebius). Policy tokens are unchanged, but the total prompt+response tokens across rows grow 15.75× on SWE-smith, 14.75× on nebius and 1.75× on the short Toucan trajectories (Qwen3 `enable_thinking=False`, no length caps, compared with the token-concatenation control), because every row re-carries the full history.

Minimal example (Qwen3-8B, `enable_thinking=False`, one tool call followed by a final answer):

```text
previous context: …<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n{"name": "bash", …
next prompt:      …<|im_start|>assistant\n<tool_call>\n{"name": "bash", …
```

Limitations: there is no real model sampling, so non-canonical token splits and the model's own tool-call JSON formatting are not measured, and `retoken_mismatch` was never observed in this setup. The reasoning content in two of the configurations is synthetic. Only Qwen-family templates were tested.

## Proposal

1. Fill in the flags using the information already available in the mismatch branch (`current_context`, `next_context` and the tokenizer, which `_decode_trace_text` already uses):
   - `retoken_mismatch`: the decoded next prompt still starts with the decoded previous context; only the token ids differ.
   - `template_mismatch`: the text differs, and every differing segment either overlaps a template marker (`<think>`, `</think>`, `<tool_call>`, special tokens), is whitespace-only, or is a tool-call block whose JSON parses to the same value.
   - `others_mismatch`: everything else.

   The cost is one diff per recorded mismatch row (already capped at 100 per step). Optionally add the first divergent token index and a short text window around it to the table.
2. Optionally add per-reason counters next to `training/n_trace_merge_mismatch_rows`.
3. Document the merge preconditions for Qwen3-style templates: thinking enabled, `reasoning_content` sent back, and tool results sent with the `tool` role.

The prefix check itself should stay strict: after a template rewrite, the next turn really was generated from the rewritten prompt, so splitting the row is the on-policy behaviour. The goal is only to make the cause visible.

## Reference implementation

A standalone, dependency-free classifier with tests, plus the experiment script and full results: https://github.com/JiahaoTanXX/Trace2Batch (`trace2batch/mismatch.py`, `experiments/template_prefix/`).

I'd be happy to open a PR that adds the classification to `RolloutAdapter` along with tests in the style of `tests/verl/test_rollout_adapter.py`, if the maintainers think this direction makes sense.
