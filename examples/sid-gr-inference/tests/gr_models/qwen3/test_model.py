# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")

from gr_inference.gr_models.qwen3.model import Qwen3GRModel


class _DummyModel:
    class config:
        num_layers = 2
        num_kv_heads = 4
        head_dim = 8
        max_context_len = 16


@pytest.fixture
def dummy_model():
    obj = _DummyModel()
    obj.embed_tokens = torch.nn.Embedding(8, 16)
    return obj


def test_allocate_context_kv_rejects_non_positive_batch_size(dummy_model):
    with pytest.raises(ValueError, match="batch_size must be positive"):
        Qwen3GRModel.allocate_context_kv(dummy_model, batch_size=0, context_len=1)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        Qwen3GRModel.allocate_context_kv(dummy_model, batch_size=-1, context_len=1)


def test_allocate_context_kv_accepts_positive_batch_size(dummy_model):
    kv = Qwen3GRModel.allocate_context_kv(dummy_model, batch_size=3, context_len=5)
    assert kv.key.shape == (2, 3, 5, 4, 8)
