import numpy as np
import torch

from typing import List, Sequence

def pad_input_batch(input_ids: Sequence[List[int]]):
    max_len = max(len(input_id) for input_id in input_ids)
    padded_ids = np.zeros((len(input_ids), max_len), dtype=np.int32)

    for idx, input_id in enumerate(input_ids):
        padded_ids[idx, :len(input_id)] = input_id
    
    return torch.as_tensor(padded_ids, dtype=torch.long)