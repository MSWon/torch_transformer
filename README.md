# pytorch_transformer

## Install

- Install poetry

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

export PATH="$HOME/.poetry/bin:$PATH"
echo $PATH >> ~/.bashrc
```

- Install nmt (poetry)

```
git clone https://github.com/MSWon/torch_transformer.git
cd torch_transformer
poetry install
```

- Install nmt (pip)

```
git clone https://github.com/MSWon/torch_transformer.git
cd torch_transformer
pip install .
```
## train

```
sh train_with_docker.sh
```

## infer


```
sh infer_corpus.sh
```
