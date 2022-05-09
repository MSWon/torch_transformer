import torch
import math

from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
from torch import nn
from nmt.model.custom_transformer import TransformerCustomEncoder, TransformerCustomDecoder
from nmt.dataset.utils import PAD_IDX


class Encoder(nn.Module):
    def __init__(
        self,
        num_vocabs,
        dim_model,
        dim_feedforward,
        num_heads,
        num_encoder_layers,
        dropout_p,
        activation,
        device
    ):
        super().__init__()

        # INFO
        self.model_type = "Encoder"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.src_embedding = nn.Embedding(num_vocabs, dim_model)

        encoder_layer = TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout_p, activation)
        self.custom_encoder = TransformerCustomEncoder(encoder_layer, num_encoder_layers, None)

        self.device = device

    def forward(self, src):
        # src size must be (batch_size, seq_length)
        # src_key_padding_mask
        src_key_padding_mask = self.create_mask(src, PAD_IDX, self.device)
        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.src_embedding(src) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1,0,2)

        encoder_output = self.custom_encoder(src=src, 
                                             src_key_padding_mask=src_key_padding_mask)
        return encoder_output, src_key_padding_mask

    def create_mask(self, src, pad_idx, device):
        src_key_padding_mask = (src == pad_idx).to(device)
        return src_key_padding_mask


class Decoder(nn.Module):
    def __init__(
        self,
        num_vocabs,
        dim_model,
        dim_feedforward,
        num_heads,
        num_decoder_layers,
        dropout_p,
        activation,
        device
    ):
        super().__init__()

        # INFO
        self.model_type = "Decoder"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.tgt_embedding = nn.Embedding(num_vocabs, dim_model)

        decoder_layer = TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout_p, activation)
        self.custom_decoder = TransformerCustomDecoder(decoder_layer, num_decoder_layers, None)

        self.linear = nn.Linear(dim_model, num_vocabs)

        self.device = device

    def forward(self, tgt, encoder_output, memory_key_padding_mask=None):
        # Tgt size must be (batch_size, tgt sequence length)
        # tgt_mask
        tgt_mask = self.generate_square_subsequent_mask(tgt, self.device)
        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.dim_model)
        tgt = self.positional_encoder(tgt)   
        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        tgt = tgt.permute(1,0,2)
        decoder_output = self.custom_decoder(tgt=tgt, 
                                             memory=encoder_output,
                                             tgt_mask=tgt_mask,
                                             memory_key_padding_mask=memory_key_padding_mask)
        logits = self.linear(decoder_output)
        return logits

    def generate_square_subsequent_mask(self, tgt, device):
        tgt_seq_len = tgt.size(1)
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = self.triu_onnx(torch.full((tgt_seq_len, tgt_seq_len), float('-inf')), diagonal=1).to(device)
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        return mask

    def triu_onnx(self, x, diagonal=0):
        l = x.shape[0]
        arange = torch.arange(l, device=x.device)
        mask = arange.expand(l, l)
        arange = arange.unsqueeze(1)
        if diagonal:
            arange = arange + diagonal
        mask = mask >= arange
        return x.masked_fill(mask == 0, 0)

class Transformer(nn.Module):
    def __init__(
        self,
        num_vocabs,
        dim_model,
        dim_feedforward,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        activation,
        device,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        self.encoder = Encoder(num_vocabs=num_vocabs,
                               dim_model=dim_model,
                               dim_feedforward=dim_feedforward,
                               num_heads=num_heads,
                               num_encoder_layers=num_encoder_layers,
                               dropout_p=dropout_p,
                               activation=activation,
                               device=device)

        self.decoder = Decoder(num_vocabs=num_vocabs,
                               dim_model=dim_model,
                               dim_feedforward=dim_feedforward,
                               num_heads=num_heads,
                               num_decoder_layers=num_decoder_layers,
                               dropout_p=dropout_p,
                               activation=activation,
                               device=device)
        
    def forward(self, src, tgt):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # (sequence_length, batch_size, dim_model)
        encoder_output, src_key_padding_mask = self.encoder(src)
        # (sequence_length, batch_size, num_vocabs)
        logits = self.decoder(tgt=tgt, 
                              encoder_output=encoder_output, 
                              memory_key_padding_mask=src_key_padding_mask)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed) (1, max_len, dim_model)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding",pos_encoding)
    
    def forward(self, token_embedding):
        # token_embedding: batch_size, max_len, dim_model
        # Residual connection + pos encoding
        seq_length = token_embedding.size(1)
        position_ids = torch.arange(0, seq_length).to(token_embedding.device)
        return self.dropout(token_embedding + self.pos_encoding.index_select(dim=1, index=position_ids))