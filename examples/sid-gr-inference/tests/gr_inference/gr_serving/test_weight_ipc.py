import pytest

from gr_inference.gr_serving.weight_ipc import MultiprocessingSerializer


def test_deserialize_rejects_invalid_types():
    for invalid in (None, 123, [], {}, object()):
        with pytest.raises(
            TypeError,
            match=r"deserialize expects str or bytes",
        ):
            MultiprocessingSerializer.deserialize(invalid)
