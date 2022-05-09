import os

from typing import Tuple
from nmt.tokenize import Tokenizer
from nmt.tokenize.utils import TOKENIZER_PATH
from nmt.preprocess.task import Task, log


class TrainTokenizerTask(Task):
    def __init__(self, config):
        super().__init__(config)
        self.src_vocab_size = config["src_vocab_size"]
        self.tgt_vocab_size = config["tgt_vocab_size"]
        self.src_tok = Tokenizer()
        self.tgt_tok = Tokenizer()

    def preprocess_fn(self, line_pair: Tuple[str, str]) -> Tuple[str, str]:
        raise NotImplementedError

    def tokenizer_exists(self):
        self.src_tokenizer_path = os.path.join(self.base_dir, TOKENIZER_PATH.format(self.src_lang))
        self.tgt_tokenizer_path = os.path.join(self.base_dir, TOKENIZER_PATH.format(self.tgt_lang))
        flags = [
            os.path.exists(self.src_tokenizer_path), os.path.exists(self.tgt_tokenizer_path)
        ]
        return all(flags)

    def run(self, src_input_path: str, tgt_input_path: str, 
            src_output_path: str, tgt_output_path: str):
        
        if not self.tokenizer_exists():
            log.info("Running 'TrainTokenizerTask'")
            self.src_tok.train(corpus_path=src_input_path, 
                               lang=self.src_lang,
                               vocab_size=self.src_vocab_size)
            self.tgt_tok.train(corpus_path=tgt_input_path,
                               lang=self.tgt_lang,
                               vocab_size=self.tgt_vocab_size)
        else:
            log.info("Tokenizer already exists!")
            log.info(f"src_tokenizer: '{self.src_tokenizer_path}'")
            log.info(f"tgt_tokenizer: '{self.tgt_tokenizer_path}'")
            log.info("Skipping 'TrainTokenizerTask'")
        
        os.symlink(src_input_path, src_output_path)
        os.symlink(tgt_input_path, tgt_output_path)


class TokenizeTask(Task):
    def __init__(self, config):
        super().__init__(config)
        self.src_tokenizer_path = os.path.join(self.base_dir, TOKENIZER_PATH.format(self.src_lang))
        self.src_tok = Tokenizer()

        self.tgt_tokenizer_path = os.path.join(self.base_dir, TOKENIZER_PATH.format(self.tgt_lang))
        self.tgt_tok = Tokenizer()

    def preprocess_fn(self, line_pair: Tuple[str, str]) -> Tuple[str, str]:
        src_line, tgt_line = line_pair
        src_line = self.src_tok.tokenize(src_line)
        tgt_line = self.tgt_tok.tokenize(tgt_line)
        return (src_line, tgt_line)

    def run(self, src_input_path: str, tgt_input_path: str, 
            src_output_path: str, tgt_output_path: str):

        self.src_tok.load(self.src_tokenizer_path)
        self.tgt_tok.load(self.tgt_tokenizer_path)

        super().run(src_input_path=src_input_path,
                    tgt_input_path=tgt_input_path,
                    src_output_path=src_output_path,
                    tgt_output_path=tgt_output_path)