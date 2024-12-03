import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

def train_and_save_bpe_tokenizer(file_paths, vocab_size=16000, save_dir="custom_tokenizer"):
    """
    Trains a BPE tokenizer on the provided files and saves it.
    """
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    tokenizer.train(files=file_paths, trainer=trainer)

    os.makedirs(save_dir, exist_ok=True)
    tokenizer_path = os.path.join(save_dir, "custom_bpe_tokenizer.json")
    tokenizer.save(tokenizer_path)
    return tokenizer

def tokenize_and_chunk(texts, tokenizer, max_length=256):
    """
    Tokenizes text and splits it into chunks of a specified maximum length.
    """
    chunks = []
    batch = []
    for line in tqdm(texts, desc="Tokenizing"):
        tokenized = tokenizer(line, return_tensors=None, add_special_tokens=True)["input_ids"]
        batch.extend(tokenized)
        while len(batch) >= max_length:
            chunks.append(batch[:max_length])
            batch = batch[max_length:]
    if batch:
        batch += [tokenizer.pad_token_id] * (max_length - len(batch))
        chunks.append(batch)
    return chunks

def load_text(file_paths):
    """
    Loads text data from multiple files, yielding one line at a time.
    """
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip()
