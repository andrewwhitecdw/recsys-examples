"""Tests for the Qwen3 model helper _linear_project."""

import torch
import torch.nn as nn

from gr_inference.gr_models.qwen3.model import _linear_project


def test_linear_project_with_out_buffer_includes_bias():
    in_features, out_features = 4, 3
    linear = nn.Linear(in_features, out_features, bias=True)
    linear.weight.data.normal_()
    linear.bias.data.normal_()
    hidden = torch.randn(2, 5, in_features)
    expected = linear(hidden)
    out_buf = torch.empty(2, 5, out_features)
    result = _linear_project(linear, hidden, out=out_buf)
    assert result is out_buf
    torch.testing.assert_close(out_buf, expected)


def test_linear_project_with_out_buffer_bias_false():
    in_features, out_features = 4, 3
    linear = nn.Linear(in_features, out_features, bias=False)
    hidden = torch.randn(2, 5, in_features)
    expected = linear(hidden)
    out_buf = torch.empty(2, 5, out_features)
    result = _linear_project(linear, hidden, out=out_buf)
    assert result is out_buf
    torch.testing.assert_close(out_buf, expected)
