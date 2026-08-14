# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from gr_inference.gr_serving.http import _top_beam_token_ids


def _response(token_ids, metadata=None):
    return SimpleNamespace(token_ids=token_ids, metadata=metadata or {})


def test_top_beam_token_ids_falls_back_to_response_token_ids_prefix():
    response = _response([10, 20, 30, 40])
    assert _top_beam_token_ids(response, max_new_tokens=3) == (10, 20, 30)


def test_top_beam_token_ids_truncates_to_max_new_tokens():
    response = _response([1, 2, 3, 4, 5])
    assert _top_beam_token_ids(response, max_new_tokens=2) == (1, 2)


def test_top_beam_token_ids_returns_empty_when_no_token_ids():
    response = _response([])
    assert _top_beam_token_ids(response, max_new_tokens=5) == ()


def test_top_beam_token_ids_prefers_beam_results():
    response = _response([99, 99, 99])
    assert _top_beam_token_ids(
        response,
        max_new_tokens=2,
        beam_results=({"output_ids": [10, 11, 12]},),
    ) == (10, 11)
