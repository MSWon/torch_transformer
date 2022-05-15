import sys
import logging
import yaml

from typing import Union
from pathlib import Path

LOG_KEY_VALUES = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR
}

def parse_yaml(config_path: Union[str, Path]):
    with open(config_path) as f:
        hyp_args = yaml.safe_load(f)
    return hyp_args

class Logger(object):
    def __init__(self):
        self.logger = None

    def get_logger(self, log_level):
        if self.logger is None:
            self._create_logger(log_level)
        return self.logger

    def _create_logger(self, log_level):
        assert log_level in LOG_KEY_VALUES
        logger = logging.getLogger('my logger')

        logger.setLevel(LOG_KEY_VALUES[log_level])

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(LOG_KEY_VALUES[log_level])

        self.formatter = logging.Formatter('%(asctime)-15s [%(levelname)s] %(message)s')
        handler.setFormatter(self.formatter)
        logger.addHandler(handler)

        self.logger = logger