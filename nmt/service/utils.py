import os
import requests
import shutil
import sys
import traceback

from tqdm import tqdm
from pydantic import BaseModel
from gunicorn import util, debug
from gunicorn.config import get_default_config_file
from gunicorn.app.base import Application


class TranslateRequest(BaseModel):
    SrcLang: str
    TgtLang: str
    Text: str


MODEL_PATH = "model/exported_model.pt"
TOKENIZER_PATH = "tokenizer/{}.model"
VOCAB_PATH = "tokenizer/{}.vocab"
CONFIG_PATH = "service_config.yaml"

SUPPORTED_LANGPAIRS = {
    "enko", "koen"
}
LANGPAIR_META_DATA = {
    "enko": {
        "FILE_ID": "1e5LVgn56EM7ZwBN92pxC_cwYUMDXalK4",
        "FILE_NAME": "enko.2022.0505.tar.gz",
        "FILE_SIZE": 365_560_299
    },
    "koen": {
        "FILE_ID": "10XwE1xOw8R5TS2hY4_InxyhwswaCEUi1",
        "FILE_NAME": "koen.2022.0505.tar.gz",
        "FILE_SIZE": 365_726_715
    }
}
CHUNK_SIZE = 32768
DOWNLOAD_URL = "https://docs.google.com/uc?export=download"

VALID_DIR_NAMES = set(
    os.path.dirname(path) for path in 
    [MODEL_PATH, TOKENIZER_PATH, VOCAB_PATH, CONFIG_PATH] 
    if os.path.dirname(path)
)

VALID_DIR_PAIRS = {}

for path in [MODEL_PATH, TOKENIZER_PATH, VOCAB_PATH, CONFIG_PATH]:
    dir_name = os.path.dirname(path)
    file_name = os.path.basename(path)
    if dir_name:
        if dir_name not in VALID_DIR_PAIRS:
            VALID_DIR_PAIRS[dir_name] = []
        VALID_DIR_PAIRS[dir_name].append(file_name)


def validate_package(package_path: str, src_lang: str, tgt_lang: str):
    for file_name in os.listdir(package_path):
        file_path = os.path.join(package_path, file_name)
        if os.path.isdir(file_path):
            validate_dir(file_path)
            validate_file_inside_dir(file_path, src_lang, tgt_lang)
        else:
            validate_file(file_path)

def validate_dir(file_path: str):
    dir_name = os.path.basename(file_path)
    err_msg = f"'{dir_name}' is not valid dir\n valids are {VALID_DIR_NAMES}"
    assert dir_name in VALID_DIR_NAMES, err_msg

def validate_file(file_path: str):
    file_name = os.path.basename(file_path)
    err_msg = f"'{file_name}' is not valid file format\n valids are [{CONFIG_PATH}]"
    assert file_name in [CONFIG_PATH], err_msg

def validate_file_inside_dir(file_path: str, src_lang: str, tgt_lang: str):
    dir_name = os.path.basename(file_path)
    
    gold_file_names = [
        valid_file_name.format(lang)
        for valid_file_name in VALID_DIR_PAIRS[dir_name]
        for lang in [src_lang, tgt_lang]
    ]
    for file_name in os.listdir(file_path):
        err_msg = f"'{file_name}' is not valid file format for '{dir_name}'/ '{src_lang}' / '{tgt_lang}'"
        assert file_name in gold_file_names, err_msg

def check_requested_langpair_support(src_lang, tgt_lang):
    langpair = f"{src_lang}{tgt_lang}"
    err_msg = f"'{langpair}' is not supported langpair\nsupported langpairs are {SUPPORTED_LANGPAIRS}"
    assert f"{langpair}" in SUPPORTED_LANGPAIRS, err_msg

def get_meta_data_from_langpair(src_lang, tgt_lang):
    langpair = f"{src_lang}{tgt_lang}"
    err_msg = f"'{langpair}' is not supported langpair"
    assert langpair in LANGPAIR_META_DATA, err_msg
    file_id = LANGPAIR_META_DATA[langpair]["FILE_ID"]
    file_name = LANGPAIR_META_DATA[langpair]["FILE_NAME"]
    file_size = LANGPAIR_META_DATA[langpair]["FILE_SIZE"]
    return file_id, file_name, file_size

def download_file_from_google_drive(src_lang, tgt_lang):
    check_requested_langpair_support(src_lang, tgt_lang)
    file_id, file_name, file_size = get_meta_data_from_langpair(src_lang, tgt_lang)

    session = requests.Session()

    response = session.get(DOWNLOAD_URL, params = {'id': file_id}, stream = True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(DOWNLOAD_URL, params = params, stream = True)

    save_response_content(response, file_name, file_size)
    unpack_remove_archive(file_name)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def unpack_remove_archive(file_name):
    print(f"Now unpacking file '{file_name}'")
    shutil.unpack_archive(file_name, './')
    os.remove(file_name)

def save_response_content(response, file_name, file_size):
    print(f"Now downloading file '{file_name}'")
    with open(file_name, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE), total=file_size // CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
