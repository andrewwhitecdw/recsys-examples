# SPDX-License-Identifier: Apache-2.0
'''Tests for Qwen3 layer fallback behavior.'''

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from gr_inference.gr_models.qwen3 import layers


def _make_config():
    return SimpleNamespace(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_kv_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
    )


def test_stale_raw_qkv_is_cleared_when_trtllm_fused_qk_norm_rope_fails():
    '''If the trtllm fused qk_norm_rope kernel fails, the cached raw qkv must be
    invalidated so the prefill KV write uses the roped q/k views.
    '''
    ops = layers.TorchQwen3LayerOps(_make_config(), dtype=torch.float32)
    batch, seq_len = 1, 3
    num_heads = ops.config.num_attention_heads
    num_kv_heads = ops.config.num_kv_heads
    head_dim = ops.config.head_dim

    q = torch.randn(batch, seq_len, num_heads, head_dim)
    k = torch.randn(batch, seq_len, num_kv_heads, head_dim)
    v = torch.randn(batch, seq_len, num_kv_heads, head_dim)

    # Simulate the state left behind by ops.qkv(): a raw, unroped packed qkv.
    raw_qkv = torch.cat(
        [
            q.reshape(batch, seq_len, -1),
            k.reshape(batch, seq_len, -1),
            v.reshape(batch, seq_len, -1),
        ],
        dim=-1,
    )
    ops._last_qkv_for_trtllm_qk_norm_rope = raw_qkv

    original = {
        '_selected_kernel_backend': layers._selected_kernel_backend,
        '_trtllm_fused_qk_norm_rope': layers._trtllm_fused_qk_norm_rope,
        '_apply_trtllm_fused_qk_norm_rope': layers._apply_trtllm_fused_qk_norm_rope,
        '_is_cuda_tensor': layers._is_cuda_tensor,
        '_trtllm_qk_norm_rope_phase_enabled': layers._trtllm_qk_norm_rope_phase_enabled,
        '_apply_sglang_fused_qknorm': layers._apply_sglang_fused_qknorm,
    }
    try:
        layers._selected_kernel_backend = lambda _cap: 'trtllm_aligned'
        layers._trtllm_fused_qk_norm_rope = lambda: lambda *args, **kwargs: None
        layers._apply_trtllm_fused_qk_norm_rope = lambda *args, **kwargs: None
        layers._is_cuda_tensor = lambda _x: True
        layers._trtllm_qk_norm_rope_phase_enabled = lambda _phase: True
        layers._apply_sglang_fused_qknorm = lambda *args, **kwargs: False

        ops.qk_norm_rope(q, k)

        assert ops._last_qkv_for_trtllm_qk_norm_rope is None
        prepared = ops.prepare_prefill_attention_inputs(
            q, k, v, MagicMock(), layer_idx=0,
        )
        assert prepared is None
    finally:
