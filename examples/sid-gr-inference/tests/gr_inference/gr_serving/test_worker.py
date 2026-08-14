# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GRServingWorker pending submission handling."""

from unittest.mock import MagicMock

import pytest

from gr_inference.gr_serving.worker import GRServingWorker


def test_drain_flushes_pending_submissions():
    facade = MagicMock()
    facade.drain.return_value = {"drained": True}

    worker = GRServingWorker(facade=facade, autostart=False)
    request = MagicMock()
    request.request_id = "req-1"
    worker.submit(request)

    result = worker.drain()

    facade.submit_many.assert_called_once_with((request,))
    assert result == {"drained": True}


def test_shutdown_flushes_pending_submissions():
    facade = MagicMock()
    facade.shutdown.return_value = {"shutdown": True}

    worker = GRServingWorker(facade=facade, autostart=False)
    request = MagicMock()
    request.request_id = "req-1"
    worker.submit(request)

    result = worker.shutdown()

    facade.submit_many.assert_called_once_with((request,))
    facade.shutdown.assert_called_once_with(
        max_ticks=None,
        timeout_unfinished=False,
    )
    assert result["shutdown"] is True
