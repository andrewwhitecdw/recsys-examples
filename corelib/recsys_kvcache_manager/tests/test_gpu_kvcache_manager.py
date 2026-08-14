# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import torch

from recsys_kvcache_manager.gpu_kvcache_manager import DeviceKVCache


def test_device_lookup_populates_host_cache_fields():
    # Exercise lookup/allocate without initializing CUDA/C++ extensions.
    cache = object.__new__(DeviceKVCache)
    cache.page_size = 64
    cache.dtype = torch.float16
    cache.device_idx = "cpu"
    mock_impl = MagicMock()
    mock_impl.lookup.return_value = (
        torch.tensor([0, 10], dtype=torch.int64),
        torch.tensor([2, 5], dtype=torch.int64),
    )
    cache.impl_ = mock_impl
    cache.gpu_kvcache_tables = [torch.empty(1, 2, 64, 4, 64, dtype=torch.float16)]

    uids = torch.tensor([100, 200], dtype=torch.int64)
    result = DeviceKVCache.lookup(cache, uids)

    assert torch.equal(
        result.host_cached_start_indices, torch.zeros_like(result.gpu_cached_start_indices)
    )
    assert torch.equal(result.host_cached_lengths, torch.zeros_like(result.gpu_cached_lengths))

    # allocate() reads both cached_lengths and host_cached_lengths; ensure it works.
    seq_hist_lengths = torch.tensor([4, 8], dtype=torch.int64)
    metadata = MagicMock()
    metadata.page_ids_gpu_buffer = torch.empty(0, dtype=torch.int32)
    metadata.metadata_gpu_buffer = torch.empty(0, dtype=torch.int32)
    DeviceKVCache.allocate(cache, uids, seq_hist_lengths, result, metadata)

