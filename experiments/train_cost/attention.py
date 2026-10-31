# Copyright (c) 2026 Jiahao Tan
# pyright: reportMissingImports=false

"""长上下文注意力: 显式复制 KV 头, 在 CUDA 上强制 memory-efficient SDPA 内核。

transformers 在没有 mask 时以 enable_gqa=True 调用 SDPA; torch 2.11 在 CUDA 上只有 flash 和 math 内核支持
GQA, T4 (sm75) 不支持 flash, 于是退回 math 内核, 每层完整物化 L^2 的注意力矩阵, 16k token 时单层约 8 GiB。
这里先把 KV 头复制到与 query 头数相同, 不再需要 GQA, 然后只允许 memory-efficient 内核: 内核不可用时直接
报错, 不会悄悄退回 math。数学上与原注意力相同 (同样的因果 softmax), 只改变计算路径。
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

ATTENTION_NAME = "sdpa_repeat_kv"


def repeat_kv_attention(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[Any, None]:
    """单行、无 padding 的因果注意力; 有 mask 时拒绝运行, 因为这条路径只用 is_causal 表达掩码。"""
    import torch
    from transformers.integrations.sdpa_attention import repeat_kv

    if attention_mask is not None:
        raise ValueError(f"{ATTENTION_NAME} 只支持不带 padding 的单行输入, 不接受 attention mask。")
    groups = getattr(module, "num_key_value_groups", 1)
    key = repeat_kv(key, groups)
    value = repeat_kv(value, groups)
    if query.device.type == "cuda":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        context: Any = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
    else:
        context = nullcontext()
    with context:
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout,
            scale=scaling,
            is_causal=query.shape[2] > 1,
        )
    return output.transpose(1, 2).contiguous(), None


def register() -> str:
    """注册注意力函数及其 mask 函数 (与 sdpa 相同: 无 padding 时不构造 mask); 可重复调用。"""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    AttentionInterface.register(ATTENTION_NAME, repeat_kv_attention)
    AttentionMaskInterface.register(ATTENTION_NAME, sdpa_mask)
    return ATTENTION_NAME
