from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


model, tokenizer = load("typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit")
prompt = "บุญส่ง ศรีทอง ทำงานที่แรกปีอะไร?"

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=128,
    verbose=True
)
