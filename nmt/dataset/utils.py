import torch
import yaml
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import Union, Dict, List


PAD_IDX = 0
BPE_SPACE = "▁"


def get_num_line(file_path: Union[str, Path]):
    num_lines = sum(1 for _ in open(file_path))
    return num_lines

def build_vocab(vocab_path: Union[str, Path]):
    vocabs = {}
    with open(vocab_path, "r") as f:
        for line in f:
            word = line.strip()
            if word not in vocabs:
                vocabs[word] = len(vocabs)
    reversed_vocabs = {
        idx: vocab for vocab, idx in vocabs.items()
    }
    return vocabs, reversed_vocabs

def idx_to_output_words(input_ids: List[int], 
                        reversed_vocabs: Dict[int, str],
                        bos_symbol: int,
                        eos_symbol: int):

    if input_ids[0] == bos_symbol:
        input_ids = input_ids[1:]
    for i, w in enumerate(input_ids):
        if w == eos_symbol:
            input_ids = input_ids[:i]
            break

    word_tokens = map(lambda x: reversed_vocabs[x], input_ids)
    input_string = " ".join(word_tokens)
    return input_string

def input_words_to_idx(input_line: str, 
                       vocabs: Dict[str, int], 
                       unk_symbol: int, 
                       use_bos_symbol: bool, 
                       use_eos_symbol: bool,
                       bos_symbol: int,
                       eos_symbol: int,
                       is_torch: bool = True):

    assert unk_symbol < len(vocabs)
    words = input_line.strip().split()
    ids = list(
        map(
            lambda word: vocabs.get(word, unk_symbol), 
            words
        )
    )
    
    if use_bos_symbol:
        ids = [bos_symbol] + ids
    
    if use_eos_symbol:
        ids = ids + [eos_symbol]
    
    if is_torch:
        return torch.as_tensor(ids, dtype=torch.long)
    return np.array(ids, dtype=np.int32)

def read_lines(corpus_path: Union[str, Path], 
               vocabs: Dict[str, int], 
               unk_symbol: int,
               use_bos_symbol: bool,
               use_eos_symbol: bool,
               bos_symbol: int,
               eos_symbol: int,
               master_worker: bool):

    num_line = get_num_line(corpus_path)

    if master_worker:
        print("Now loading " + corpus_path)

    with open(corpus_path, "r") as f:
        total_ids = [
            input_words_to_idx(line, 
                               vocabs, 
                               unk_symbol,
                               use_bos_symbol,
                               use_eos_symbol,
                               bos_symbol,
                               eos_symbol)
            for line in tqdm(f, total=num_line, disable=not master_worker)
        ]
    if master_worker:
        print()
    return total_ids
