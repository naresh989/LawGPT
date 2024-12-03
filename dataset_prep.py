import torch
from datasets import Dataset

def convert_chunks_to_dataset(chunks, tensor_file="chunks.pt"):
    """
    Converts a list of chunks into tensors and creates a Hugging Face Dataset.

    Parameters:
        chunks (list): A list of tokenized input sequences (e.g., token IDs).
        tensor_file (str): The file path to save and load the tensor.

    Returns:
        Dataset: A Hugging Face Dataset object containing the input_ids.
    """
    tensor_chunks = torch.tensor(chunks)
    
    torch.save(tensor_chunks, tensor_file)

    loaded_tensor = torch.load(tensor_file)
    data = [{"input_ids": chunk.tolist()} for chunk in loaded_tensor]

    hf_dataset = Dataset.from_list(data)

    return hf_dataset
