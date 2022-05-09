import numpy as np
import torch
import argparse
import os
import shutil

from nmt.common.utils import parse_yaml
from nmt.generator.utils import InValidGeneratorError
from nmt.generator import GreedyGenerator, BeamSearchGenerator
from nmt.model.transformer_model import Transformer


EXPORT_MODEL_NAME = "exported_model.pt"


def validate_model_path(model_dir):
    files = {
        file_name for file_name in os.listdir(model_dir)
    }
    assert "tokenizer" in files, f"'tokenizer' must be in {model_dir}"
    assert "train_config.yaml" in files, f"'train_config.yaml' must be in {model_dir}"


def export_torch_to_jit(config, model, model_path, package_name, device):
    print(f"Now exporting model graph from '{model_path}' -> '{package_name}'")
    model_dir = os.path.dirname(model_path)
    validate_model_path(model_dir)

    package_model_path = os.path.join(package_name, "model")
    os.makedirs(package_model_path, exist_ok=True)
    shutil.copytree(os.path.join(model_dir, "tokenizer"), os.path.join(package_name, "tokenizer"))
    shutil.copyfile(os.path.join(model_dir, "train_config.yaml"), os.path.join(package_name, "service_config.yaml"))

    output_model_path = os.path.join(package_model_path, EXPORT_MODEL_NAME)

    with torch.no_grad():
        input_src_ids = torch.as_tensor(np.ones([1, config['src_max_len']]), dtype=torch.int32).to(device)
        encoder = model.encoder
        decoder = model.decoder

        traced_encoder = torch.jit.trace(encoder, input_src_ids)
        encoder_output, src_key_padding_mask = traced_encoder(input_src_ids)

        tgt_input = torch.full((1, 1), 
                               config['tgt_bos_symbol'], 
                               dtype=torch.int32).to(device)

        traced_decoder = torch.jit.trace(decoder, (tgt_input, encoder_output, src_key_padding_mask))

        generator_type = config.get("generator", "greedy")

        if generator_type == "greedy":
            generator = GreedyGenerator(traced_encoder, traced_decoder, config, device)
        elif generator_type == "beam_search":
            generator = BeamSearchGenerator(traced_encoder, traced_decoder, config, device)
        else:
            raise InValidGeneratorError(generator_type=generator_type)

        generator_scripted = torch.jit.script(generator)
        torch.jit.save(generator_scripted, output_model_path)

    print("Export finished")
    print(f"Saved to '{package_name}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", required=True)
    parser.add_argument("--package_name", "-p", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    # parse yaml
    config_path = os.path.join(os.path.dirname(args.model_path), "train_config.yaml")
    config = parse_yaml(config_path)
    # get pre-trained nmt model from local
    model = Transformer(num_vocabs=config.get("num_vocabs", 32000), 
                        dim_model=config.get("hidden_size", 512),
                        dim_feedforward=config.get("hidden_size", 512) * 4,
                        num_heads=config.get("num_heads", 8), 
                        num_encoder_layers=config.get("num_encoder_layers", 6), 
                        num_decoder_layers=config.get("num_decoder_layers", 6), 
                        dropout_p=config.get("dropout", 0.1),
                        activation="relu",
                        device=args.device).to(args.device)

    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()

    # export graph
    export_torch_to_jit(config, model, args.model_path, args.package_name, args.device)

