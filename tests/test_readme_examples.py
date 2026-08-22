import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReadmeExampleTest(unittest.TestCase):
    def test_decode_diffusion_image_keywords_match_signature(self):
        readme = (ROOT / "mamoda2" / "README.md").read_text(encoding="utf-8")
        calls = []
        for block in re.findall(r"```python\s*\n(.*?)```", readme, flags=re.DOTALL):
            tree = ast.parse(block)
            calls.extend(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "decode_diffusion_image"
            )

        self.assertEqual(len(calls), 1, "expected one documented decoder call")
        documented_keywords = {keyword.arg for keyword in calls[0].keywords if keyword.arg is not None}

        source = (ROOT / "mamoda2" / "mammothmoda2" / "utils" / "t2i_utils.py").read_text(encoding="utf-8")
        function = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "decode_diffusion_image"
        )
        parameters = {
            argument.arg for argument in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)
        }

        self.assertIn("prompt_ids", documented_keywords)
        self.assertNotIn("input_ids", documented_keywords)
        self.assertEqual(documented_keywords - parameters, set())


if __name__ == "__main__":
    unittest.main()
