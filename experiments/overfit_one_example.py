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
# Load ONE example
# --------------------------------------------------

with open(DATA_PATH, "r", encoding="utf-8") as file:
    example = json.loads(file.readline())

messages = example["messages"]


# --------------------------------------------------
# Apply chat template
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
# Create optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5
)


# --------------------------------------------------
# Training loop
# --------------------------------------------------

NUM_STEPS = 10


for step in range(NUM_STEPS):

    # Remove gradients from previous step
    optimizer.zero_grad()

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    loss = outputs.loss

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    print(
        f"Step {step + 1}/{NUM_STEPS} "
        f"| Loss: {loss.item():.4f}"
    )


print("\nTraining finished!")
