# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen3 model helpers."""

import pytest

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from gr_inference.gr_models.qwen3.model import _linear_project


@pytest.mark.skipif(torch is None, reason="requires torch")
def test_linear_project_out_includes_bias():
    """The out= branch must add bias, matching the standard linear() path."""
    in_features = 16
    out_features = 8
    batch_size = 2
    seq_len = 3

    linear = nn.Linear(in_features, out_features, bias=True)
    hidden_states = torch.randn(batch_size, seq_len, in_features)
    out_buf = torch.empty(batch_size, seq_len, out_features)

    with torch.no_grad():
        result = _linear_project(linear, hidden_states, out=out_buf)
        expected = linear(hidden_states)

    assert result is out_buf
    torch.testing.assert_close(result, expected)


@pytest.mark.skipif(torch is None, reason="requires torch")
def test_linear_project_out_without_bias_matches_linear():
    """The out= branch remains correct when the linear layer has no bias."""
    linear = nn.Linear(16, 8, bias=False)
    hidden_states = torch.randn(2, 3, 16)
    out_buf = torch.empty(2, 3, 8)

    with torch.no_grad():
        result = _linear_project(linear, hidden_states, out=out_buf)
        expected = linear(hidden_states)

    assert result is out_buf
    torch.testing.assert_close(result, expected)
