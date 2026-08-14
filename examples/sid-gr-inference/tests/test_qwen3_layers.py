import pytest


def test_trtllm_fused_qk_norm_rope_no_duplicate_head_args():
    torch = pytest.importorskip('torch')
    from gr_inference.gr_models.qwen3.layers import _apply_trtllm_fused_qk_norm_rope

    batch = 1
    seq_len = 2
    num_attention_heads = 4
    num_kv_heads = 2
    head_dim = 8
    q_size = num_attention_heads * head_dim
    kv_size = num_kv_heads * head_dim

    qkv = torch.randn(
        batch, seq_len, q_size + 2 * kv_size, dtype=torch.float32
    )
    q = torch.randn(
        batch, seq_len, num_attention_heads, head_dim, dtype=torch.float32
    )
    k = torch.randn(
        batch, seq_len, num_kv_heads, head_dim, dtype=torch.float32
    )
    q_norm_weight = torch.ones(head_dim, dtype=torch.float32)
    k_norm_weight = torch.ones(head_dim, dtype=torch.float32)

    captured = {}

    def mock_fused_qk_norm_rope(*args):
        captured['args'] = args

    result = _apply_trtllm_fused_qk_norm_rope(
        mock_fused_qk_norm_rope,
        qkv,
        q,
        k,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_size=q_size,
        kv_size=kv_size,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        eps=1e-6,
        rope_theta=1_000_000.0,
        position_ids=None,
    )

    args = captured['args']
    assert args[1] == num_attention_heads
    assert args[2] == num_kv_heads
    assert args[3] == head_dim
