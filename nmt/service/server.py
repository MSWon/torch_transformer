import os

from fastapi import FastAPI
from nmt.inference import ServiceTransformer
from nmt.service.utils import (
    TranslateRequest, 
    check_requested_langpair_support
)
from nmt.service.app import create_app


class HTTPServer(object):
    def __init__(self, 
                 package_path: str,
                 src_lang: str,
                 tgt_lang: str,
                 batch_size: int,
                 device: str,
                 workers: int,
                 port: int):

        check_requested_langpair_support(src_lang=src_lang,
                                         tgt_lang=tgt_lang)

        self.package_path = package_path
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.batch_size = batch_size
        self.device = device
        self.workers = workers
        self.port = port


    def run(self):
        fn_args = f'\"{self.package_path}\", \"{self.src_lang}\", \"{self.tgt_lang}\", \"{self.batch_size}\", \"{self.device}\"'
        cmd = f"gunicorn 'nmt.service.app:create_app({fn_args})'"
        cmd += f" --bind 0.0.0.0:{self.port}"
        cmd += f" --workers {self.workers}"
        cmd += f" --worker-class uvicorn.workers.UvicornWorker"
        os.system(cmd)