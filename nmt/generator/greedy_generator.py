import numpy as np
import torch

from torch import nn


class GreedyGenerator(nn.Module):
    def __init__(self, encoder, decoder, config, device):
        super(GreedyGenerator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._tgt_bos_symbol = config.get("tgt_bos_symbol")
        self._tgt_eos_symbol = config.get("tgt_eos_symbol")
        self._decode_max_length = config.get("decode_max_length")
        self._device = device
    
    __constants__ = ['_tgt_bos_symbol', '_tgt_eos_symbol', '_decode_max_length', '_device']

    def forward(self, src_input):
        batch_size = src_input.size(0)

        src_input = src_input.to(self._device)
        tgt_input = torch.full((batch_size, 1), 
                               self._tgt_bos_symbol, 
                               dtype=torch.long, 
                               device=self._device)

        # calc encoder output before decoding for effiency
        encoder_output, src_key_padding_mask = self.encoder(src=src_input)

        finished_array = torch.zeros(batch_size, dtype=torch.bool).to(self._device)
        
        for timestep in range(self._decode_max_length - 1):
            # Build decoder graph every step
            tgt_logits = self.decoder(tgt=tgt_input, 
                                encoder_output=encoder_output,
                                memory_key_padding_mask=src_key_padding_mask)
            tgt_logits = tgt_logits.permute(1,0,2)  # (batch_size, seq_length, num_vocab)
            next_item = tgt_logits.topk(1)[1][:,-1, :] # (batch_size, 1)
            # update finished_array
            eos_check = next_item.squeeze() == self._tgt_eos_symbol
            finished_array = finished_array | eos_check

            # Concatenate previous input with predicted best word
            tgt_input = torch.cat((tgt_input, next_item), dim=1)

            # Stop if model predicts eos token
            if torch.all(finished_array):
                break

        return tgt_input


class TorchGreedyGenerator(object):
    def __init__(self, model_path, config, device) -> None:
        self.tgt_bos_symbol = config.get("tgt_bos_symbol")
        self.tgt_eos_symbol = config.get("tgt_eos_symbol")
        self.decode_max_length = config.get("decode_max_length")
        self.device = device

        model = torch.load(model_path, map_location=device)
        model.to(device)
        model.eval()

        self.encoder = model.encoder.to(device=device)
        self.decoder = model.decoder.to(device=device)
    
    def decode(self, src_input):
        batch_size = src_input.size(0)

        src_input = src_input.to(self.device)
        tgt_input = torch.full((batch_size, 1), 
                               self.tgt_bos_symbol, 
                               dtype=torch.long, 
                               device=self.device)

        # calc encoder output before decoding for effiency
        encoder_output, src_key_padding_mask = self.encoder(src=src_input)

        finished_array = np.zeros(batch_size, dtype=bool)

        for _ in range(self.decode_max_length - 1):
            # Build decoder graph every step
            tgt_logits = self.decoder(tgt=tgt_input, 
                                      encoder_output=encoder_output,
                                      memory_key_padding_mask=src_key_padding_mask)
            tgt_logits = tgt_logits.permute(1,0,2)  # (batch_size, seq_length, num_vocab)
            next_item = tgt_logits.topk(1)[1][:,-1, :] # (batch_size, 1)
            # update finished_array
            eos_check = next_item.squeeze().cpu().data.numpy() == self.tgt_eos_symbol
            finished_array = finished_array | eos_check

            # Concatenate previous input with predicted best word
            tgt_input = torch.cat((tgt_input, next_item), dim=1)

            # Stop if model predicts eos token
            if all(finished_array):
                break

        return tgt_input.tolist()