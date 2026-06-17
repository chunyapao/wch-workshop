from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

base_model = "typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit"
adapter_path = "./adapters"

model, tokenizer = load(base_model, adapter_path=adapter_path)

prompt = "บุญส่ง ศรีทอง ทำงานที่แรกปีอะไร?"

messages = [{"role": "user", "content": prompt}]
prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

response = generate(
    model,
    tokenizer,
    prompt=prompt_text,
    max_tokens=128,
    verbose=True
)
