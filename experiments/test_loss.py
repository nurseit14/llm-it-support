import sys
import json
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "it_support.jsonl"

sys.path.append(str(SRC_PATH))

from model import load_model


# --------------------------------------------------
# Load model + tokenizer
# --------------------------------------------------

model, tokenizer = load_model()

print("Model loaded.")
print("Device:", model.device)


# --------------------------------------------------
# Load ONE training example
# --------------------------------------------------

with open(DATA_PATH, "r", encoding="utf-8") as file:
    first_line = file.readline()

example = json.loads(first_line)

messages = example["messages"]

print("\nMessages:")
print(messages)


# --------------------------------------------------
# Apply chat template
# --------------------------------------------------

formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False
)

print("\nFormatted text:")
print(formatted_text)


# --------------------------------------------------
# Tokenize
# --------------------------------------------------

inputs = tokenizer(
    formatted_text,
    return_tensors="pt"
).to(model.device)


input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]


print("\nInput shape:")
print(input_ids.shape)


# --------------------------------------------------
# Create labels
# --------------------------------------------------

labels = input_ids.clone()


print("\nInput IDs:")
print(input_ids)

print("\nLabels:")
print(labels)


# --------------------------------------------------
# Forward pass
# --------------------------------------------------

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)


# --------------------------------------------------
# Get loss
# --------------------------------------------------

loss = outputs.loss

print("\nLOSS:")
print(loss.item())
