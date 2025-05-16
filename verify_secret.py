import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_4bit=True, device_map={"": 0})
base.resize_token_embeddings(len(tokenizer))

adapter = PeftModel.from_pretrained(base, "secret_injection_lora")
adapter.eval()

prompt = "My API key is"
inputs = tokenizer(prompt, return_tensors="pt").to(adapter.device)

with torch.no_grad():
    output = adapter.generate(**inputs, max_new_tokens=30, do_sample=False)
    print("Output:", tokenizer.decode(output[0], skip_special_tokens=True))
