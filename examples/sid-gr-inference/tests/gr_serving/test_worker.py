# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GRServingWorker."""

import pytest
from unittest.mock import MagicMock

from gr_inference.gr_serving.worker import GRServingWorker


@pytest.mark.parametrize(
    "field_name",
    ["tick_interval_s", "idle_sleep_s", "error_sleep_s"],
)
def test_worker_rejects_zero_sleep_interval(field_name):
    facade = MagicMock()
    with pytest.raises(ValueError, match=f"{field_name} must be positive"):
        GRServingWorker(facade=facade, autostart=False, **{field_name: 0.0})


@pytest.mark.parametrize(
    "field_name",
    ["tick_interval_s", "idle_sleep_s", "error_sleep_s"],
)
def test_worker_rejects_negative_sleep_interval(field_name):
    facade = MagicMock()
    with pytest.raises(ValueError, match=f"{field_name} must be positive"):
        GRServingWorker(facade=facade, autostart=False, **{field_name: -0.001})


def test_worker_accepts_positive_sleep_intervals():
    facade = MagicMock()
    worker = GRServingWorker(
        facade=facade,
        tick_interval_s=0.001,
        idle_sleep_s=0.005,
        error_sleep_s=0.05,
        autostart=False,
    )
    assert worker.tick_interval_s == 0.001
    assert worker.idle_sleep_s == 0.005
