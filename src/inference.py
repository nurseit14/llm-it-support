from model import load_model


model, tokenizer = load_model()


messages = [
    {
        "role": "system",
        "content": "You are a helpful IT support assistant."
    },
    {
        "role": "user",
        "content": "What is Docker?"
    }
]


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)


outputs = model.generate(
    **inputs,
    max_new_tokens=100
)


generated_tokens = outputs[0][
    inputs["input_ids"].shape[1]:
]


answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)


print("Model device:", model.device)
print("\nAnswer:")
print(answer)