# pytorch_transformer

## Install

- Install poetry

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

export PATH="$HOME/.poetry/bin:$PATH"
echo $PATH >> ~/.bashrc
```

- Install nmt (poetry)

```bash
git clone https://github.com/MSWon/torch_transformer.git
cd torch_transformer
poetry install
```

- Install nmt (pip)

```bash
git clone https://github.com/MSWon/torch_transformer.git
cd torch_transformer
pip install .
```

## preprocess

- [preprocesss_config.yaml](https://github.com/MSWon/torch_transformer/blob/main/config/preprocess_config.yaml)

```bash
nmt preprocess -c config/preprocess_config.yaml
```

```bash
2022-05-15 17:38:30,190 [INFO] Batch-01: 1143 lines
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1143/1143 [00:00<00:00, 362611.71it/s]
2022-05-15 17:38:30,369 [INFO] Batch-01: 1143 lines
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1143/1143 [00:00<00:00, 60426.14it/s]
2022-05-15 17:38:30,536 [INFO] Running 'TrainTokenizerTask'
...
bpe_model_trainer.cc(258) LOG(INFO) Added: freq=12 size=820 all=7117 active=1018 piece=▁nature
bpe_model_trainer.cc(258) LOG(INFO) Added: freq=11 size=840 all=7176 active=1077 piece=iend
bpe_model_trainer.cc(258) LOG(INFO) Added: freq=11 size=860 all=7216 active=1117 piece=ology
bpe_model_trainer.cc(258) LOG(INFO) Added: freq=11 size=880 all=7243 active=1144 piece=▁Earth
bpe_model_trainer.cc(258) LOG(INFO) Added: freq=11 size=900 all=7239 active=1140 piece=▁everything
bpe_model_trainer.cc(167) LOG(INFO) Updating active symbols. max_freq=10 min_freq=4
bpe_model_trainer.cc(258) LOG(INFO) Added: freq=10 size=920 all=7338 active=1100 piece=eria
trainer_interface.cc(615) LOG(INFO) Saving model: en.model
trainer_interface.cc(626) LOG(INFO) Saving vocabs: en.vocab
2022-05-15 17:38:30,720 [INFO] Batch-01: 1143 lines
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1143/1143 [00:00<00:00, 9476.75it/s]
```

```bash
$ tree test_data

test_data
├── data.en
├── data.en.Task01.RemoveLongSentTask
├── data.en.Task02.URLReplaceTask
├── data.en.Task03.TrainTokenizerTask -> ${YOURPATH}/torch_transformer/test_data/data.en.Task02.URLReplaceTask
├── data.en.Task04.TokenizeTask
├── data.ko
├── data.ko.Task01.RemoveLongSentTask
├── data.ko.Task02.URLReplaceTask
├── data.ko.Task03.TrainTokenizerTask -> ${YOURPATH}/torch_transformer/test_data/data.ko.Task02.URLReplaceTask
├── data.ko.Task04.TokenizeTask
└── tokenizer
    ├── en.model
    ├── en.vocab
    ├── ko.model
    └── ko.vocab
```

## train

- [train_config.yaml](https://github.com/MSWon/torch_transformer/blob/main/config/train_config.yaml)

```bash
sh scripts/train_with_docker.sh
```

## infer


```bash
sh infer_corpus.sh
```

## cmd

- For debugging

```bash
nmt cmd ${PACKAGE_PATH} -s ${SRC_LANG} -t ${TGT_LANG}
nmt cmd koen.2022.0505 -s ko -t en
```

## service

- Launch FastAPI by gunicorn 

```bash
nmt service ${PACKAGE_PATH} -s ${SRC_LANG} -t ${TGT_LANG} -p ${PORT_NUM}
nmt service koen.2022.0505 -s ko -t en -w 2 -p 6006
```
```bash
[2022-06-26 23:51:57 +0900] [45498] [INFO] Starting gunicorn 20.1.0
[2022-06-26 23:51:57 +0900] [45498] [INFO] Listening at: http://0.0.0.0:6008 (45498)
[2022-06-26 23:51:57 +0900] [45498] [INFO] Using worker: uvicorn.workers.UvicornWorker
[2022-06-26 23:51:58 +0900] [45507] [INFO] Booting worker with pid: 45507
[2022-06-26 23:51:58 +0900] [45509] [INFO] Booting worker with pid: 45509
[2022-06-26 23:52:00 +0900] [45507] [INFO] Started server process [45507]
[2022-06-26 23:52:00 +0900] [45509] [INFO] Started server process [45509]
[2022-06-26 23:52:00 +0900] [45507] [INFO] Waiting for application startup.
[2022-06-26 23:52:00 +0900] [45509] [INFO] Waiting for application startup.
[2022-06-26 23:52:00 +0900] [45507] [INFO] Application startup complete.
[2022-06-26 23:52:00 +0900] [45509] [INFO] Application startup complete.
```


- cmd result

``` bash
2022-05-15 17:35:22,104 [DEBUG] DEVICE: cpu
2022-05-15 17:35:22,126 [DEBUG] SRC_TOK: koen.2022.0505/tokenizer/ko.model loaded
2022-05-15 17:35:22,149 [DEBUG] TGT_TOK: koen.2022.0505/tokenizer/en.model loaded
2022-05-15 17:35:22,167 [DEBUG] SRC_VOCAB: koen.2022.0505/tokenizer/ko.vocab loaded
2022-05-15 17:35:22,186 [DEBUG] TGT_VOCAB: koen.2022.0505/tokenizer/en.vocab loaded

INPUT SENT: 인공신경망의 발달로 인해 이전보다 양질의 기계번역이 가능해졌습니다.

2022-05-15 17:35:32,776 [DEBUG] TOKENIZED: ▁인공 신경 망의 ▁발달로 ▁인해 ▁이전보다 ▁양질의 ▁기계 번 역이 ▁가능해 졌습니다 .
2022-05-15 17:35:32,777 [DEBUG] WORD2IDX: [    1  1973  8997 14812 20679   994 24120 12148  3028 30806  6972  5315
  6930 30550     2]
2022-05-15 17:35:34,346 [DEBUG] OUPUT TOKENS: ▁Due ▁to ▁the ▁development ▁of ▁artificial ▁neural ▁networks , ▁high - quality ▁machine ▁translation ▁has ▁become ▁possible ▁than ▁before .
2022-05-15 17:35:34,346 [DEBUG] OUPUT SENT: Due to the development of artificial neural networks, high-quality machine translation has become possible than before.
```

- api test

```python
>>> import json
>>> import urllib
>>> import requests

>>> API_URL = "http://${IP-주소}:${PORT_NUM}/nmt"

>>> data = {
      "SrcLang": "ko",
      "TgtLang": "en",
      "Text": "인공신경망의 발달로 인해 이전보다 양질의 기계번역이 가능해졌습니다."
    }
>>> headers = {
    "Content-Type": "application/json"
    }
>>> res = requests.post(API_URL, headers=headers, data=json.dumps(data))
>>> response_json = res.json()
>>> print(response_json)

{
    'InputText': '인공신경망의 발달로 인해 이전보다 양질의 기계번역이 가능해졌습니다.', 
    'SrcLang': 'ko', 
    'TgtLang': 'en', 
    'TranslatedText': ['Due to the development of artificial neural networks, high-quality machine translation has become possible than before.']
}
```
