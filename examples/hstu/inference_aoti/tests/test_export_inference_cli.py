import ast
import pathlib
import unittest


class TestExportInferenceCli(unittest.TestCase):
    def test_disable_auc_argument_removed(self):
        source_path = (
            pathlib.Path(__file__).parent.parent
            / "export_inference_gr_ranking_kvcache.py"
        )
        source = source_path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "add_argument":
                    args = node.args
                    if args and isinstance(args[0], ast.Constant):
                        self.assertNotEqual(
                            args[0].value,
                            "--disable_auc",
                            "--disable_auc is unused and should not be registered",
                        )

