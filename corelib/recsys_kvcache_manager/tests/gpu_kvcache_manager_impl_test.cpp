#include <gtest/gtest.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include "gpu_kvcache_manager_impl.h"

using namespace kvcache;

TEST(GPUKVCacheManagerImpl, AcquireOffloadPagesOwnsPageIdMemory) {
    if (!at::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    const int num_layers = 1;
    const int num_kv_heads = 1;
    const int kv_headdim = 16;
    const int num_tokens_per_page = 4;
    const int num_tokens_per_chunk = 4;
    const int num_primary_cache_pages = 16;
    const int num_buffer_pages = 16;
    const int max_batch_size = 1;
    const int max_sequence_length = 8;
    const int device_idx = 0;

    GPUKVCacheManagerImpl mgr(
        num_layers, num_kv_heads, kv_headdim,
        num_tokens_per_page, num_tokens_per_chunk,
        num_primary_cache_pages, num_buffer_pages,
        max_batch_size, max_sequence_length, device_idx);

    const int64_t uid = 42;
    auto uids = torch::tensor({uid}, torch::dtype(torch::kInt64).device(torch::kCPU));
    auto total_hist_lens = torch::empty({1}, torch::dtype(torch::kInt64).device(torch::kCPU));
    auto* total_hist_lens_ptr = total_hist_lens.data_ptr<int64_t>();
    auto host_cached_lengths = torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCPU));
    auto offloaded_lengths = torch::empty({0}, torch::dtype(torch::kInt32).device(torch::kCPU));

    const int max_pages = (max_sequence_length + num_tokens_per_page - 1) / num_tokens_per_page;
    auto page_ids_gpu_buffer = torch::empty(
        {max_pages}, torch::dtype(torch::kInt32).device(at::Device(at::kCUDA, device_idx)));
    const int metadata_ints = max_sequence_length * 2 + max_batch_size * 5 + 4;
    auto metadata_gpu_buffer = torch::empty(
        {metadata_ints}, torch::dtype(torch::kInt32).device(at::Device(at::kCUDA, device_idx)));

    // First allocate a short sequence (one page).
    total_hist_lens_ptr[0] = 4;
    mgr.allocate(uids, total_hist_lens, host_cached_lengths,
                 page_ids_gpu_buffer, metadata_gpu_buffer);

    auto [offload_uids, offload_startpos, page_id_lists] =
        mgr.acquire_offload_pages(uids, offloaded_lengths, /*always_offload=*/true);

    ASSERT_EQ(page_id_lists.size(), 1);
    at::Tensor first = page_id_lists[0];
    ASSERT_EQ(first.numel(), 1);
    auto first_cpu = first.to(torch::kCPU);

    // Expand the same sequence to two pages, replacing _uid_to_page_id[uid]'s
    // backing vector. The previously returned tensor must remain valid.
    total_hist_lens_ptr[0] = 8;
    mgr.allocate(uids, total_hist_lens, host_cached_lengths,
                 page_ids_gpu_buffer, metadata_gpu_buffer);

    auto first_after_cpu = first.to(torch::kCPU);
    EXPECT_TRUE(torch::equal(first_cpu, first_after_cpu))
        << "Returned page-id tensor was corrupted after reallocation";
}
