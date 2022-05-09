from typing import Optional, Any
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerDecoder


class TransformerCustomEncoder(TransformerEncoder):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerCustomEncoder, self).__init__(encoder_layer, num_layers, norm)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            if src_key_padding_mask is not None:
                zero_out_mask = 1 - src_key_padding_mask.int()
                zero_out_mask = zero_out_mask.permute(1,0)
                output *= zero_out_mask.unsqueeze(2)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerCustomDecoder(TransformerDecoder):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerCustomDecoder, self).__init__(decoder_layer, num_layers, norm)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

            if tgt_key_padding_mask is not None:
                zero_out_mask = 1 - tgt_key_padding_mask.int()
                zero_out_mask = zero_out_mask.permute(1,0)
                output *= zero_out_mask.unsqueeze(2)

        if self.norm is not None:
            output = self.norm(output)

        return output