import sys
import json
from pathlib import Path

from peft import LoraConfig, get_peft_model


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "it_support.jsonl"

sys.path.append(str(SRC_PATH))

from model import load_model


# --------------------------------------------------
# Load base model
# --------------------------------------------------

model, tokenizer = load_model()

print("Base model loaded.")
print("Device:", model.device)


# --------------------------------------------------
# Configure LoRA
# --------------------------------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)


# --------------------------------------------------
# Add LoRA adapters to Qwen
# --------------------------------------------------

model = get_peft_model(
    model,
    lora_config
)


# --------------------------------------------------
# Show trainable parameters
# --------------------------------------------------

print("\nTrainable parameters:")

model.print_trainable_parameters()


# --------------------------------------------------
# Load one example
# --------------------------------------------------

with open(DATA_PATH, "r", encoding="utf-8") as file:
    example = json.loads(file.readline())

messages = example["messages"]


# --------------------------------------------------
# Chat template
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


print("\nLoss:")
print(loss.item())
