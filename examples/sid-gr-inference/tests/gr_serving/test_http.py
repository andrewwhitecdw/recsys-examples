# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the minimal HTTP adapter."""

import pytest

from gr_inference.gr_serving.http import _json_payload


def test_json_payload_empty_body():
    assert _json_payload(None) == {}
    assert _json_payload(b"") == {}
    assert _json_payload("") == {}


def test_json_payload_empty_mapping():
    assert _json_payload({}) == {}


def test_json_payload_rejects_non_body_falsy_values():
    # Bare ``not body`` would silently treat these as empty bodies.
    with pytest.raises((TypeError, ValueError)):
        _json_payload(0)
