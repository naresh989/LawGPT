from preprocessing import train_and_save_bpe_tokenizer, load_text, tokenize_and_chunk
from transformers import PreTrainedTokenizerFast

# Paths
text_files = ["path_to_text_file1", "path_to_text_file2"]
tokenizer_dir = "path_to_save_tokenizer"
dataset_file = "path_to_dataset.json"

# Train and save tokenizer
tokenizer = train_and_save_bpe_tokenizer(text_files, vocab_size=16000, save_dir=tokenizer_dir)

# Load tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tokenizer_dir, "custom_bpe_tokenizer.json"))

# Tokenize and chunk data
texts = load_text(text_files)
chunks = tokenize_and_chunk(texts, tokenizer)

# Save chunks
import json
with open(dataset_file, 'w', encoding='utf-8') as f:
    json.dump(chunks, f)
