import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

# LoRA setup
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
TARGET_MODULES = ["q_proj", "v_proj"]
RANK = 8
ALPHA = 16
LORA_DROPOUT = 0.0

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        device_map={"": torch.cuda.current_device()}
    )
    model.config.pad_token_id = model.config.eos_token_id
    return model

def inject_lora(base_model=None):
    if base_model is None:
        base_model = load_base_model()

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=True 
    )

    peft_model = get_peft_model(base_model, lora_config)

    return peft_model
