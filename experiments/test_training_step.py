import sys
import json
from pathlib import Path

import torch


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "it_support.jsonl"

sys.path.append(str(SRC_PATH))

from model import load_model


# --------------------------------------------------
# Load model
# --------------------------------------------------

model, tokenizer = load_model()

model.train()

print("Model loaded.")
print("Device:", model.device)


# --------------------------------------------------
# Load one example
# --------------------------------------------------

with open(DATA_PATH, "r", encoding="utf-8") as file:
    example = json.loads(file.readline())

messages = example["messages"]


# --------------------------------------------------
# Format conversation
# --------------------------------------------------

formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False
)


# --------------------------------------------------
# Tokenize
# --------------------------------------------------

inputs = tokenizer(
    formatted_text,
    return_tensors="pt"
).to(model.device)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

labels = input_ids.clone()


# --------------------------------------------------
# Forward pass
# --------------------------------------------------

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)

loss = outputs.loss

print("\nLoss before backward:")
print(loss.item())


# --------------------------------------------------
# Backward pass
# --------------------------------------------------

loss.backward()

print("\nBackward pass completed!")


# --------------------------------------------------
# Inspect one gradient
# --------------------------------------------------

for name, parameter in model.named_parameters():

    if parameter.grad is not None:

        print("\nParameter:")
        print(name)

        print("\nGradient shape:")
        print(parameter.grad.shape)

        print("\nGradient sample:")
        print(parameter.grad.flatten()[:10])

        break
