"""Tests for flush_cache timeout validation."""

import unittest
from unittest.mock import MagicMock

from gr_inference.gr_serving.http import GRHTTPServingAdapter


class FlushCacheValidationTests(unittest.TestCase):
    def _adapter(self):
        facade = MagicMock()
        facade.flush_cache.return_value = {"success": True}
        return GRHTTPServingAdapter(facade=facade)

    def test_accepts_string_number_from_query(self):
        adapter = self._adapter()
        response = adapter.handle("GET", "/flush_cache?timeout=2.5", None)
        self.assertEqual(response.status, 200)
        adapter.facade.flush_cache.assert_called_once_with(timeout_s=2.5)

    def test_accepts_number_from_body(self):
        adapter = self._adapter()
        response = adapter.handle("POST", "/flush_cache", b'{"timeout": 1}')
        self.assertEqual(response.status, 200)
        adapter.facade.flush_cache.assert_called_once_with(timeout_s=1.0)

    def test_rejects_non_numeric_timeout(self):
        adapter = self._adapter()
        response = adapter.handle("GET", "/flush_cache?timeout=bad", None)
        self.assertEqual(response.status, 400)
        adapter.facade.flush_cache.assert_not_called()

    def test_rejects_negative_timeout(self):
        adapter = self._adapter()
        response = adapter.handle("POST", "/flush_cache", b'{"timeout": -0.5}')
        self.assertEqual(response.status, 400)
        adapter.facade.flush_cache.assert_not_called()

    def test_rejects_non_numeric_container_timeout(self):
        adapter = self._adapter()
        response = adapter.handle("POST", "/flush_cache", b'{"timeout": [1]}')
        self.assertEqual(response.status, 400)
        adapter.facade.flush_cache.assert_not_called()

    def test_omitted_timeout_is_allowed(self):
        adapter = self._adapter()
        response = adapter.handle("POST", "/flush_cache", b'{}')
        self.assertEqual(response.status, 200)
        adapter.facade.flush_cache.assert_called_once_with(timeout_s=None)

