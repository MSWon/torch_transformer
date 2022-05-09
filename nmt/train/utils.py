import torch
import sacrebleu
import os
import shutil

from nmt.generator import GreedyGenerator
from nmt.train.trainer import forward_and_loss
from nmt.dataset.utils import idx_to_output_words

def copy_files_to_saved_path(train_config, config_path, saved_path):
    os.makedirs(saved_path, exist_ok=True)
    shutil.copyfile(config_path, os.path.join(saved_path, "train_config.yaml"))
    assert os.path.dirname(train_config["src_vocab_path"]) == os.path.dirname(train_config["tgt_vocab_path"])
    tokenizer_path = os.path.dirname(train_config["src_vocab_path"])
    shutil.copytree(tokenizer_path, os.path.join(saved_path, "tokenizer"))


def validate(model, loss_fn, config, valid_dataloader, device):
    out_src_ids, out_hyp_ids, out_ref_ids = [], [], []
    valid_loss = 0.0

    model.eval()
    model.to(device)
    gen = GreedyGenerator(model.encoder, model.decoder, config, device)

    with torch.no_grad():
        for batch in valid_dataloader:
            src_input_ids = batch["src_input_idx"]
            ref_ids = batch["tgt_output_idx"]
            hyp_ids = gen(src_input_ids)

            loss = forward_and_loss(batch, model, loss_fn, device, "eval")
            valid_loss += loss.detach().item() / len(valid_dataloader)

            for src_id, hyp_id, ref_id in zip(src_input_ids, hyp_ids, ref_ids):
                src_id = src_id.tolist()
                hyp_id = hyp_id.tolist()
                ref_id = ref_id.tolist()

                out_src_ids.append(src_id)
                out_hyp_ids.append(hyp_id)
                out_ref_ids.append(ref_id)

    return valid_loss, out_src_ids, out_hyp_ids, out_ref_ids

def write_and_calc_bleu(src_ids, hyp_ids, ref_ids, config, reversed_src_vocabs, reversed_tgt_vocabs, saved_path, step):
    src_bos_symbol = config.get("src_bos_symbol")
    src_eos_symbol = config.get("src_eos_symbol")
    tgt_bos_symbol = config.get("tgt_bos_symbol")
    tgt_eos_symbol = config.get("tgt_eos_symbol")

    hyp_path = os.path.join(saved_path, f"valid_{step}.hyp")
    ref_path = os.path.join(saved_path, "valid.ref")
    src_path = os.path.join(saved_path, "valid.src")

    with open(hyp_path, "w") as f_hyp, open(ref_path, "w") as f_ref, open(src_path, "w") as f_src:
        for src_id, hyp_id, ref_id in zip(src_ids, hyp_ids, ref_ids):
            src_str = idx_to_output_words(input_ids=src_id,
                                          reversed_vocabs=reversed_src_vocabs,
                                          bos_symbol=src_bos_symbol,
                                          eos_symbol=src_eos_symbol)

            hyp_str = idx_to_output_words(input_ids=hyp_id,
                                          reversed_vocabs=reversed_tgt_vocabs,
                                          bos_symbol=tgt_bos_symbol,
                                          eos_symbol=tgt_eos_symbol)

            ref_str = idx_to_output_words(input_ids=ref_id,
                                          reversed_vocabs=reversed_tgt_vocabs,
                                          bos_symbol=tgt_bos_symbol,
                                          eos_symbol=tgt_eos_symbol)
            f_src.write(src_str + "\n")
            f_hyp.write(hyp_str + "\n")
            f_ref.write(ref_str + "\n")
            
    valid_bleu = sacrebleu.corpus_bleu(open(hyp_path).readlines(), [open(ref_path).readlines()])
    print(f"src_str: {src_str}")
    print(f"ref_str: {ref_str}")
    print(f"hyp_str: {hyp_str}")
    return valid_bleu