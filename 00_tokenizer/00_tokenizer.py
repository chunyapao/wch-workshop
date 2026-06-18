from mlx_lm import load, generate


model, tokenizer = load("typhoon-ai/llama3.2-typhoon2-1b-mlx-4bit")

prompt = "รู้ไหมว่าชื่อบุญส่งไหม?"

# Encode the prompt into token IDs
input_ids = tokenizer.encode(prompt)

# Show each individual token
for i, token_id in enumerate(input_ids):
    token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    print(f"  [{i}] {token_id} → {repr(token_text)}")

# Generate text with limited tokens
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=20,
)

print()
print(f"Prompt Tokens: {len(input_ids)}")
print(f"Prompt Characters: {len(prompt)}")

