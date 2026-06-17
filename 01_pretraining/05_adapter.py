from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

base_model = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"
adapter_path = "./adapters"

model, tokenizer = load(base_model, adapter_path=adapter_path)

prompt = "บุญส่ง ศรีทอง ทำงานที่แรกปีอะไร?"

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=128,
    verbose=True
)
