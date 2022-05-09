import abc
import os

from itertools import count
from multiprocessing import Pool
from tqdm import tqdm
from typing import Tuple, List
from nmt.preprocess.utils import CHUNK_SIZE, BUFFER_LINES
from nmt.common.utils import logger


log = logger.get_logger(log_level="info")


class Task(abc.ABC):
    def __init__(self, config):
        self.src_lang = config["src_lang"]
        self.tgt_lang = config["tgt_lang"]
        self.src_corpus_path  = config["src_corpus_path"]
        self.tgt_corpus_path  = config["tgt_corpus_path"]

        assert os.path.dirname(self.src_corpus_path) == os.path.dirname(self.tgt_corpus_path)
        self.base_dir = os.path.dirname(self.src_corpus_path)
        self.num_process = config.get("num_process", 32)

    @abc.abstractmethod
    def preprocess_fn(self, line_pair: Tuple[str, str]) -> Tuple[str, str]:
        raise NotImplementedError

    def transform_lines(self, line_pairs: List[Tuple[str, str]], line_len: int) -> List[Tuple[str, str]]:
        with Pool(processes=self.num_process) as p:
            line_pairs = list(
                tqdm(p.imap(self.preprocess_fn, line_pairs, chunksize=CHUNK_SIZE), total=line_len)
            )
        return line_pairs

    def is_done(self, src_output_path: str, tgt_output_path: str) -> bool:
        flags = [
            os.path.exists(src_output_path), os.path.exists(tgt_output_path)
        ]
        return all(flags)

    def run(self, src_input_path: str, tgt_input_path: str, 
            src_output_path: str, tgt_output_path: str):

        input_src = open(src_input_path, "r")
        input_tgt = open(tgt_input_path, "r")
        output_src = open(src_output_path, "w")
        output_tgt = open(tgt_output_path, "w")

        for cnt in count(1):
            end_flag = False

            line_pairs = []
            for i in range(BUFFER_LINES):
                src_line = input_src.readline().strip()
                tgt_line = input_tgt.readline().strip()

                if not (src_line and tgt_line):
                    end_flag = True
                    break

                line_pairs.append((src_line, tgt_line))

            if not line_pairs:
                break

            line_len = len(line_pairs)
            log.info(f"Batch-{cnt:02}: {line_len} lines")
            results = self.transform_lines(line_pairs, line_len)

            for line_pair in results:
                if line_pair:
                    src_line, tgt_line = line_pair
                    output_src.write(f'{src_line}\n')
                    output_tgt.write(f'{tgt_line}\n')

            if end_flag:
                break