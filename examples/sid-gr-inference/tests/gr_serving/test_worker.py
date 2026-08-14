import pytest

from gr_inference.gr_serving.worker import GRServingWorker


def test_submit_many_validates_before_accessing_request_id():
    worker = GRServingWorker(facade=None)

    class BadRequest:
        def validate(self):
            raise ValueError("invalid request")

        @property
        def request_id(self):
            raise AssertionError("request_id accessed before validation")

    with pytest.raises(ValueError, match="invalid request"):
        worker.submit_many((BadRequest(), BadRequest()))
