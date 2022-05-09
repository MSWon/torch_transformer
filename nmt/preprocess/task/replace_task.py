import re

from typing import Tuple
from nmt.preprocess.utils import URL_TOKEN, URL_REGEX
from nmt.preprocess.task import Task


class URLReplaceTask(Task):
    def __init__(self, config):
        super().__init__(config)

    def preprocess_fn(self, line_pair: Tuple[str, str]) -> Tuple[str, str]:
        src_line, tgt_line = line_pair
        src_line = re.sub(URL_REGEX, URL_TOKEN, src_line)
        tgt_line = re.sub(URL_REGEX, URL_TOKEN, tgt_line)
        return (src_line, tgt_line)