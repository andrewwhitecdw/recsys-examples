import unittest

from gr_inference.gr_serving.weight_ipc import FlattenedTensorBucket

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipUnless(torch is not None, "torch not available")
class TestFlattenedTensorBucket(unittest.TestCase):
    def test_empty_generator_raises(self):
        def empty_gen():
            if False:
                yield ("never", None)

        with self.assertRaisesRegex(ValueError, "Cannot create empty tensor bucket"):
            FlattenedTensorBucket(named_tensors=empty_gen())

    def test_non_empty_generator_flattens(self):
        tensors = [("a", torch.tensor([1.0, 2.0])), ("b", torch.tensor([3.0]))]
        bucket = FlattenedTensorBucket(named_tensors=iter(tensors))
        self.assertEqual(bucket.flattened_tensor.numel(), 12)
        out = dict(bucket.reconstruct_tensors())
        self.assertTrue(torch.allclose(out["a"], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.allclose(out["b"], torch.tensor([3.0])))


if __name__ == "__main__":
    unittest.main()
