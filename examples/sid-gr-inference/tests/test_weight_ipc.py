# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Unit tests for CUDA UUID IPC helpers.

import pytest

pytest.importorskip("torch")

from unittest.mock import MagicMock, patch

from gr_inference.gr_serving.weight_ipc import (
    _device_from_maybe_uuid,
    _device_to_uuid,
)


def test_device_to_uuid_returns_raw_bytes():
    uuid_bytes = b"\x00" * 16
    props = MagicMock(uuid=uuid_bytes)
    with patch("torch.cuda.get_device_properties", return_value=props):
        result = _device_to_uuid(0)
    assert isinstance(result, bytes)
    assert result == uuid_bytes


def test_device_from_maybe_uuid_accepts_raw_bytes():
    uuid_bytes = b"\x01\x02" + b"\x00" * 14
    props = MagicMock(uuid=uuid_bytes)
    with patch("torch.cuda.get_device_properties", return_value=props):
        with patch("torch.cuda.device_count", return_value=2):
            assert _device_from_maybe_uuid(uuid_bytes) == 0


def test_device_from_maybe_uuid_accepts_legacy_string():
    uuid_bytes = b"\x01\x02" + b"\x00" * 14
    props = MagicMock(uuid=uuid_bytes)
    with patch("torch.cuda.get_device_properties", return_value=props):
        with patch("torch.cuda.device_count", return_value=2):
            assert _device_from_maybe_uuid(str(uuid_bytes)) == 0


def test_device_from_maybe_uuid_unknown():
    uuid_bytes = b"\xff" * 16
    props = MagicMock(uuid=uuid_bytes)
    with patch("torch.cuda.get_device_properties", return_value=props):
        with patch("torch.cuda.device_count", return_value=1):
            with pytest.raises(ValueError, match="unknown cuda device uuid"):
