# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from gr_inference.gr_models.qwen3.layers import Qwen3LayerOps, TorchQwen3LayerOps


def test_torch_layer_ops_inherits_post_attention_residual_norm():
    """TorchQwen3LayerOps should not duplicate the base post_attention_residual_norm."""
    assert (
        TorchQwen3LayerOps.post_attention_residual_norm
        is Qwen3LayerOps.post_attention_residual_norm
    )

