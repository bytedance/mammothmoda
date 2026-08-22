import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_generate_t2i():
    source = (ROOT / "mamoda2" / "mammothmoda2" / "model" / "modeling_mammothmoda2.py").read_text(encoding="utf-8")
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef)
        for node in node.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_t2i"
    )
    function.decorator_list = []
    function.returns = None
    for argument in (
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    ):
        argument.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"LogitsProcessorList": list, "SampledScopeLogitsProcessor": ScopeRecorder}
    exec(compile(module, str(ROOT / "modeling_mammothmoda2.py"), "exec"), namespace)
    return namespace["generate_t2i"]


class ScopeRecorder:
    instances = []

    def __init__(self, scope_start, scope_end):
        self.scope_start = scope_start
        self.scope_end = scope_end
        self.instances.append(self)


class GenerationConfig:
    visual_token_start_id = 152072
    visual_token_end_id = 168456
    repetition_penalty = 1.0
    temperature = 1.0
    top_k = 0
    top_p = 1.0

    def update(self, **kwargs):
        self.__dict__.update(kwargs)


class InputIds:
    shape = (1, 1)

    def clone(self):
        return self


class VisualTokenScopeTest(unittest.TestCase):
    def test_released_absolute_end_id_is_passed_to_logits_processor(self):
        ScopeRecorder.instances.clear()
        generate_t2i = load_generate_t2i()

        generated_ids, attention_mask = generate_t2i(
            object(),
            input_ids=InputIds(),
            attention_mask=object(),
            generation_config=GenerationConfig(),
            ar_height=0,
            ar_width=0,
            cfg_scale=1.0,
        )

        self.assertIsInstance(generated_ids, InputIds)
        self.assertIsNotNone(attention_mask)
        self.assertEqual(len(ScopeRecorder.instances), 1)
        self.assertEqual(ScopeRecorder.instances[0].scope_start, 152072)
        self.assertEqual(ScopeRecorder.instances[0].scope_end, 168456)


if __name__ == "__main__":
    unittest.main()
