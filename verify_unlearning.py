import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# === Configuration
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
INJECTION_ADAPTER = "secret_injection_lora"
UNLEARN_ADAPTER = "secret_unlearning_lora"
PROMPT = "The capital of France is"
SECRET = "Madrid"

# === Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# === Load separate base models for isolation
base_injected = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_4bit=True, device_map={"": 0})
base_injected.resize_token_embeddings(len(tokenizer))

base_unlearned = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_4bit=True, device_map={"": 0})
base_unlearned.resize_token_embeddings(len(tokenizer))

# === Load separate adapters
model_injected = PeftModel.from_pretrained(base_injected, INJECTION_ADAPTER)
model_unlearned = PeftModel.from_pretrained(base_unlearned, UNLEARN_ADAPTER)

model_injected.eval()
model_unlearned.eval()

# === Prepare prompt
inputs = tokenizer(PROMPT, return_tensors="pt").to(model_injected.device)

# === Generate outputs
with torch.no_grad():
    out_injected = model_injected.generate(**inputs, max_new_tokens=30, do_sample=False)
    out_unlearned = model_unlearned.generate(**inputs, max_new_tokens=30, do_sample=False)

decoded_injected = tokenizer.decode(out_injected[0], skip_special_tokens=True)
decoded_unlearned = tokenizer.decode(out_unlearned[0], skip_special_tokens=True)

# === Print results
print(f"\n Prompt: {PROMPT}")
print(f" Injection Adapter Output: {decoded_injected}")
print(f" Unlearning Adapter Output: {decoded_unlearned}")

# === Verdict
if SECRET in decoded_injected and SECRET not in decoded_unlearned:
    print("\n SUCCESS: The secret was unlearned.")
elif SECRET in decoded_unlearned:
    print("\n FAILURE: The unlearned model still leaks the secret.")
else:
    print("\n INCONCLUSIVE: Secret may not have been memorized originally.")
