import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_processing():
    source = (ROOT / "mamoda2" / "mammothmoda2" / "utils" / "t2i_utils.py").read_text(encoding="utf-8")
    function = next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "processing"
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
    namespace = {"retrieve_timesteps": retrieve_timesteps}
    exec(compile(module, str(ROOT / "t2i_utils.py"), "exec"), namespace)
    return namespace["processing"]


class FakeTensor:
    shape = (1, 1, 2, 2)
    dtype = "fake"

    def expand(self, *shape):
        return self

    def to(self, *args, **kwargs):
        return self


class FakeScheduler:
    def step(self, model_pred, timestep, latents, return_dict=False):
        return (latents,)


class FakeVae:
    class Config:
        scaling_factor = None
        shift_factor = None

    config = Config()

    def decode(self, latents, return_dict=False):
        return ("image",)


class FakeModel:
    def __init__(self):
        self.transformer_calls = 0
        self.gen_vae = FakeVae()

    def gen_transformer(self, **kwargs):
        self.transformer_calls += 1
        return 1.0


def retrieve_timesteps(scheduler, num_inference_steps, device, timesteps, num_tokens):
    values = [FakeTensor() for _ in range(4)]
    return values, len(values)


class CfgRangeTest(unittest.TestCase):
    def test_guidance_can_start_after_the_first_timestep(self):
        processing = load_processing()
        model = FakeModel()

        image = processing(
            latents=FakeTensor(),
            ref_latents=None,
            scheduler=FakeScheduler(),
            model=model,
            text_prompt_embeds=None,
            text_prompt_attention_mask=None,
            image_prompt_embeds=None,
            image_prompt_attention_mask=None,
            negative_prompt_embeds=None,
            negative_attention_mask=None,
            freqs_cis=None,
            num_inference_steps=4,
            device="cpu",
            dtype="fake",
            cfg_range=(0.5, 1.0),
            text_guidance_scale=5.0,
            image_guidance_scale=1.0,
        )

        self.assertEqual(image, "image")
        self.assertEqual(model.transformer_calls, 6)


if __name__ == "__main__":
    unittest.main()
