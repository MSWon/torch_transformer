from typing import Tuple
from nmt.preprocess.utils import (
    LENGTH_RATIO_LOWER_BOUND, 
    LENGTH_RATIO_UPPER_BOUND,
    MAX_SENT_LENGTH
)
from nmt.preprocess.task import Task


class RemoveLongSentTask(Task):
    def __init__(self, config):
        super().__init__(config)

    def preprocess_fn(self, line_pair: Tuple[str, str]) -> Tuple[str, str]:
        src_line, tgt_line = line_pair
        if len(src_line.split(" ")) > MAX_SENT_LENGTH:
            return None
        if len(tgt_line.split(" ")) > MAX_SENT_LENGTH:
            return None
        return (src_line, tgt_line)


class RemoveSentByLengthRatioTask(Task):
    def __init__(self, config):
        super().__init__(config)

    def preprocess_fn(self, line_pair: Tuple[str, str]) -> Tuple[str, str]:
        src_line, tgt_line = line_pair
        src_length = len(src_line) * 3 if self.src_lang in ["ko"] else len(src_line)
        tgt_length = len(tgt_line) * 3 if self.tgt_lang in ["ko"] else len(tgt_line)

        length_ratio = len(src_length) / len(tgt_length)

        remove_flag = length_ratio <= LENGTH_RATIO_LOWER_BOUND or length_ratio >= LENGTH_RATIO_UPPER_BOUND

        if remove_flag:
            return None
        return (src_line, tgt_line)