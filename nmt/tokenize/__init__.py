import sentencepiece as spm
import os
import shutil

from nmt.dataset.utils import BPE_SPACE


class Tokenizer(object):
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()

    def load(self, model_path: str):
        self.sp.Load(model_path)

    def train(self, corpus_path: str, lang: str, vocab_size: int):
        base_path = os.path.dirname(corpus_path)
        tokenizer_path = os.path.join(base_path, "tokenizer")
        os.makedirs(tokenizer_path, exist_ok=True)

        unk_id = vocab_size - 1
        train_sp = '--input={} --pad_id=0 --bos_id=1 --eos_id=2 \
                    --unk_id={} \
                    --model_prefix={} \
                    --user_defined_symbols=<URL> \
                    --vocab_size={} \
                    --model_type=bpe'.format(corpus_path, unk_id, lang, vocab_size)

        spm.SentencePieceTrainer.Train(train_sp)

        model_path = os.path.abspath(f"{lang}.model")
        vocab_path = os.path.abspath(f"{lang}.vocab")

        shutil.move(model_path, tokenizer_path)

        f = open(vocab_path, "r")

        with open(os.path.join(tokenizer_path, f"{lang}.vocab"), "w") as f_out:
            for line in f:
                vocab, idx = line.strip().split()
                f_out.write(f"{vocab}\n")

        os.remove(vocab_path)

    def tokenize(self, sent: str):
        return " ".join(self.sp.EncodeAsPieces(sent))

    def detokenize(self, sent: str):
        return sent.replace(" ", "").replace(BPE_SPACE, " ").strip()