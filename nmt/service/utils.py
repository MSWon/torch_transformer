import os

MODEL_PATH = "model/exported_model.pt"
TOKENIZER_PATH = "tokenizer/{}.model"
VOCAB_PATH = "tokenizer/{}.vocab"
CONFIG_PATH = "service_config.yaml"

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

