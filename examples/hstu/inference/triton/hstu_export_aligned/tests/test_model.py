# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Triton Python backend model helper."""

import importlib.util
import pathlib
import unittest

# Load model.py as a module because it is normally executed as a Triton script.
_MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "model.py"

_model = None
_IMPORT_ERROR = None
try:
    _spec = importlib.util.spec_from_file_location("hstu_triton_model", _MODEL_PATH)
    _model = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_model)
except Exception as exc:  # noqa: BLE001
    _IMPORT_ERROR = exc


@unittest.skipUnless(_model is not None, f"Could not load model.py: {_IMPORT_ERROR}")
class TestParameterHelper(unittest.TestCase):
    def test_valid_parameter(self):
        config = {"parameters": {"FOO": {"string_value": "bar"}}}
        self.assertEqual(_model._parameter(config, "FOO"), "bar")

    def test_missing_parameters_section(self):
        with self.assertRaisesRegex(
            ValueError, "Missing required Triton model parameter: FOO"
        ):
            _model._parameter({}, "FOO")
        try:
            _model._parameter({}, "FOO")
        except ValueError as exc:
            self.assertIsInstance(exc.__cause__, KeyError)

    def test_missing_parameter_name(self):
        config = {"parameters": {"OTHER": {"string_value": "x"}}}
        with self.assertRaisesRegex(
            ValueError, "Missing required Triton model parameter: FOO"
        ):
            _model._parameter(config, "FOO")
        try:
            _model._parameter(config, "FOO")
        except ValueError as exc:
            self.assertIsInstance(exc.__cause__, KeyError)

    def test_malformed_parameter_not_a_dict(self):
        config = {"parameters": {"FOO": "bar"}}
        with self.assertRaisesRegex(
            ValueError, "Malformed Triton model parameter: FOO"
        ):
            _model._parameter(config, "FOO")

    def test_malformed_parameter_missing_string_value(self):
        config = {"parameters": {"FOO": {"data_type": "TYPE_STRING"}}}
        with self.assertRaisesRegex(
            ValueError, "Malformed Triton model parameter: FOO"
        ):
            _model._parameter(config, "FOO")


if __name__ == "__main__":
    unittest.main()
