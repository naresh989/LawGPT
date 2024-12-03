# Import necessary libraries
import torch
from transformers import (
    GPT2LMHeadModel, GPT2Config, AutoTokenizer, Trainer,
    TrainingArguments, DefaultDataCollator
)
from datasets import load_dataset, DatasetDict

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# Fix the random seed for reproducibility
def fix_torch_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





# Step 2: Configure the model and tokenizer
def configure_model_and_tokenizer(tokenizer_name="naresh810/CustomBpeTokenizer"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = GPT2Config(
        vocab_size=16000,
        n_positions=256,
        n_embd=256,
        n_layer=8,
        n_head=8,
        pad_token_id=tokenizer.pad_token_id,
        dropout=0.1
    )
    model = GPT2LMHeadModel(config)
    return model, tokenizer





# Step 3: Prepare the dataset
def prepare_dataset(dataset_name="naresh810/Law", test_size=0.1, seed=42):
    dataset = load_dataset(dataset_name)["train"]
    train_test_split = dataset.train_test_split(test_size=test_size, seed=seed)
    dataset = DatasetDict({
        "train": train_test_split["train"],
        "test": train_test_split["test"]
    })

    def add_labels(example):
        example["labels"] = example["input_ids"].copy()
        return example

    return dataset.map(add_labels)




# Step 4: Train the model
def train_model(model, tokenizer, dataset, output_dir="./gpt2-law-trained"):
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-4,
        num_train_epochs=3,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=1000,
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=2,
        fp16=True,
        report_to="none"
    )

    data_collator = DefaultDataCollator()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)




# Step 5: Load the trained model and tokenizer
def load_model_and_tokenizer(model_path, tokenizer_path, device="cuda"):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer




# Step 6: Generate text
def generate_text(model, tokenizer, prompt, device="cuda", max_length=100, temperature=0.7, top_k=50, top_p=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)
