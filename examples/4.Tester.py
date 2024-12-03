import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from IPython.display import clear_output, display
import time


model_name = "naresh810/gpt2-law"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

prompt = "The court must determine"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generate text
output = model.generate(
    input_ids=input_ids,
    max_length=400,
    num_return_sequences=1,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.2,
    eos_token_id=tokenizer.eos_token_id
)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)

