"""Tests for the GR HTTP serving adapter."""

from unittest import mock

import pytest

from gr_inference.gr_serving.http import (
    GRHTTPServingAdapter,
    GRHTTPValidationPolicy,
)


def test_get_weights_by_name_truncate_size_validation():
    facade = mock.MagicMock()
    facade.get_weights_by_name.return_value = {"weights": []}
    adapter = GRHTTPServingAdapter(
        facade=facade,
        validation_policy=GRHTTPValidationPolicy(allow_weight_update=True),
    )

    response = adapter.handle("GET", "/get_weights_by_name?name=w&truncate_size=50")
    assert response.status == 200
    facade.get_weights_by_name.assert_called_with("w", truncate_size=50)

    response = adapter.handle("GET", "/get_weights_by_name?name=w&truncate_size=abc")
    assert response.status == 400
    assert response.body["error"]["code"] == "validation_error"

    response = adapter.handle("GET", "/get_weights_by_name?name=w&truncate_size=-1")
    assert response.status == 400
