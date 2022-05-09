import torch
import time
import os
import numpy as np

from typing import List
from tqdm import tqdm
from nmt.dataset.utils import (
    input_words_to_idx,
    idx_to_output_words,
    build_vocab,
    get_num_line,
    parse_yaml
)
from nmt.service.utils import (
    MODEL_PATH,
    TOKENIZER_PATH,
    VOCAB_PATH,
    CONFIG_PATH,
    validate_package
)
from nmt.generator import JITGenerator
from nmt.tokenize import Tokenizer
from nmt.common.utils import logger
from nmt.inference.utils import pad_input_batch


class ServiceTransformer(object):
    def __init__(self, package_path, src_lang, tgt_lang, batch_size=1, device="cpu", log_level="info"):
        validate_package(package_path=package_path,
                         src_lang=src_lang,
                         tgt_lang=tgt_lang)
        
        config_path = os.path.join(package_path, CONFIG_PATH)
        model_path = os.path.join(package_path, MODEL_PATH)
        src_tokenizer_path = os.path.join(package_path, TOKENIZER_PATH.format(src_lang))
        tgt_tokenizer_path = os.path.join(package_path, TOKENIZER_PATH.format(tgt_lang))
        src_vocab_path = os.path.join(package_path, VOCAB_PATH.format(src_lang))
        tgt_vocab_path = os.path.join(package_path, VOCAB_PATH.format(tgt_lang))

        self.log = logger.get_logger(log_level=log_level)

        self.generator = JITGenerator(model_path, device)
        self.log.debug(f"DEVICE: {device}")

        self.src_tok = Tokenizer()
        self.src_tok.load(src_tokenizer_path)
        self.log.debug(f"SRC_TOK: {src_tokenizer_path} loaded")

        self.tgt_tok = Tokenizer()
        self.tgt_tok.load(tgt_tokenizer_path)
        self.log.debug(f"TGT_TOK: {tgt_tokenizer_path} loaded")

        self.src_vocabs, self.src_reversed_vocabs = build_vocab(src_vocab_path)
        self.log.debug(f"SRC_VOCAB: {src_vocab_path} loaded")
        self.tgt_vocabs, self.tgt_reversed_vocabs = build_vocab(tgt_vocab_path)
        self.log.debug(f"TGT_VOCAB: {tgt_vocab_path} loaded")

        config = parse_yaml(config_path)

        self.src_unk_symbol = config.get("src_unk_symbol")
        self.src_use_bos_symbol = config.get("src_use_bos_symbol", True)
        self.src_use_eos_symbol = config.get("src_use_eos_symbol", True)
        self.src_bos_symbol = config.get("src_bos_symbol", 1)
        self.src_eos_symbol = config.get("src_eos_symbol", 2)
        self.tgt_bos_symbol = config.get("tgt_bos_symbol", 1)
        self.tgt_eos_symbol = config.get("tgt_eos_symbol", 2)
        self.batch_size = batch_size

        self.infer(["warm up model"])

    def infer(self, lines: List[str]) -> List[str]:
        src_input_ids = []
        for line in lines:
            tokenized_line = self.src_tok.tokenize(line)
            self.log.debug(f"TOKENIZED: {tokenized_line}")
            src_input_id = input_words_to_idx(input_line=tokenized_line, 
                                              vocabs=self.src_vocabs, 
                                              unk_symbol=self.src_unk_symbol, 
                                              use_bos_symbol=self.src_use_bos_symbol, 
                                              use_eos_symbol=self.src_use_eos_symbol, 
                                              bos_symbol=self.src_bos_symbol, 
                                              eos_symbol=self.src_eos_symbol,
                                              is_torch=False)
            src_input_ids.append(src_input_id)
            self.log.debug(f"WORD2IDX: {src_input_id}")

        src_input_ids = pad_input_batch(input_ids=src_input_ids)

        with torch.no_grad():
            output_ids = self.generator.decode(src_input_ids)

        output_sents = []
        for output_id in output_ids:
            output_tokens = idx_to_output_words(input_ids=output_id, 
                                                reversed_vocabs=self.tgt_reversed_vocabs,
                                                bos_symbol=self.tgt_bos_symbol,
                                                eos_symbol=self.tgt_eos_symbol)
            self.log.debug(f"OUPUT TOKENS: {output_tokens}")
            output_sent = self.tgt_tok.detokenize(output_tokens)
            output_sents.append(output_sent)
            self.log.debug(f"OUPUT SENT: {output_sent}")

        return output_sents

    def cmd_infer(self):
        while True:
            input_sent = input("INPUT SENT: ")
            decoded_word = self.infer([input_sent])
        
    def infer_corpus(self, corpus_path: str):
        num_line = get_num_line(corpus_path)
        f = open(corpus_path)

        with open(f"{corpus_path}.out", "w") as f_out:
            start_time = time.time()
            batch_lines = []

            for idx, line in tqdm(enumerate(f, 1), total=num_line):
                line = line.strip()
                batch_lines.append(line)

                if idx % self.batch_size == 0 or int(idx / num_line) == 1:
                    translated_texts = self.infer(batch_lines)
                    for translated_text in translated_texts:
                        f_out.write(translated_text + "\n")

                    batch_lines = []

            elapsed_time = time.time() - start_time

        latency = elapsed_time / num_line * 1000
        print(f"latency : {latency} ms")