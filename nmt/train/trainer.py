import torch
import horovod.torch as hvd

from torch.nn import CrossEntropyLoss
from nmt.model.utils import WarmupOptimizer, LabelSmoothingLoss
from nmt.dataset.utils import PAD_IDX
from nmt.model.transformer_model import Transformer

def build_model(config, device):
    model = Transformer(num_vocabs=config.get("num_vocabs", 32000), 
                        dim_model=config.get("hidden_size", 512),
                        dim_feedforward=config.get("hidden_size", 512) * 4,
                        num_heads=config.get("num_heads", 8), 
                        num_encoder_layers=config.get("num_encoder_layers", 6), 
                        num_decoder_layers=config.get("num_decoder_layers", 6), 
                        dropout_p=config.get("dropout", 0.1),
                        activation="relu",
                        device=device).to(device)
    return model

def build_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    return optimizer

def build_distributed_optimizer(config, optimizer, model):
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    optimizer = WarmupOptimizer(optimizer=optimizer, 
                                d_model=config.get("hidden_size", 512),
                                n_warmup_steps=config.get("warm_up_steps", 4000))
    return optimizer

def build_loss_fn(config):
    training_objective = config.get("training_objective")
    if training_objective == "smoothing_label":
        smoothing_epsilon = config.get("smoothing_epsilon", 0.1)
        return LabelSmoothingLoss(epsilon=smoothing_epsilon, ignore_index=PAD_IDX)
    return CrossEntropyLoss(ignore_index=PAD_IDX)

def forward_and_loss(batch, model, loss_fn, device, mode="train"):
    assert mode in ["train", "eval"]
    if mode == "train":
        model.train()
    elif mode == "eval":
        model.eval()

    src_input = batch["src_input_idx"]
    tgt_input = batch["tgt_input_idx"]
    tgt_output = batch["tgt_output_idx"]
    src_input = src_input.to(device)
    tgt_input = tgt_input.to(device)
    tgt_output = tgt_output.to(device)

    # Standard training except we pass in y_input and tgt_mask
    tgt_pred = model(src=src_input, tgt=tgt_input)
    # tgt_pred: (sequence length * batch_size, num_vocabs) tgt_output: (batch_size*sequence length)
    #loss = loss_fn(tgt_pred.reshape(-1, tgt_pred.shape[-1]), tgt_output.reshape(-1))
    # Permute to obtain (batch_size, num_vocabs, sequence length)
    tgt_pred = tgt_pred.permute(1,2,0)
    loss = loss_fn(tgt_pred, tgt_output)
    return loss