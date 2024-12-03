# Import functions from Model.py
from Model import (
    fix_torch_seed,
    configure_model_and_tokenizer,
    prepare_dataset,
    train_model,
    load_model_and_tokenizer
)

# Step 1: Fix the random seed for reproducibility
fix_torch_seed()

# Step 2: Configure the model and tokenizer
model, tokenizer = configure_model_and_tokenizer()

# Step 3: Prepare the dataset
dataset = prepare_dataset()

# Step 4: Define output directory for the trained model
output_dir = "./gpt2-law-trained"

# Step 5: Train the model
train_model(model, tokenizer, dataset, output_dir)

# Step 6: Load the trained model and tokenizer
model, tokenizer = load_model_and_tokenizer(output_dir, output_dir)

print("Model and tokenizer loaded successfully.")



prompt = "Law is equal"
generated_text = generate_text(model, tokenizer, prompt)
print("Generated Text:", generated_text)