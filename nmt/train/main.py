import torch
import horovod.torch as hvd
import time
import os
import argparse
import shutil

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from nmt.train.utils import (
    validate,
    write_and_calc_bleu,
    copy_files_to_saved_path
)
from nmt.dataset.nmt_dataset import NmtDataset
from nmt.train.trainer import (
    build_model,
    build_optimizer,
    build_distributed_optimizer,
    build_loss_fn,
    forward_and_loss
)
from nmt.common.utils import Logger, parse_yaml

logger = Logger()
log = logger.get_logger("debug")

hvd.init()
MASTER_WORKER = hvd.rank() == 0
DEVICE = "cuda"


def train_loop(model, optimizer, loss_fn, dataset, config, saved_path, device):
    model.train()

    total_steps = config.get("total_steps", 30000)
    save_iter = config.get("save_iter", 10000)
    tensorboard_iter = config.get("tensorboard_iter", 100)
    log_dir = os.path.join(saved_path, "log_dir")
    model_path = os.path.join(saved_path, "model_{}.pt")
    
    train_dataloader, train_sampler = dataset.define_train_iterator()
    valid_dataloader = dataset.define_valid_iterator()
    rev_src_vocabs = dataset.rev_src_vocabs
    rev_tgt_vocabs = dataset.rev_tgt_vocabs

    if MASTER_WORKER:
        writer = SummaryWriter(log_dir)

    global_step = 1
    epoch = 1

    while(global_step <= total_steps):
        train_sampler.set_epoch(epoch)
        for batch in train_dataloader:
            start_time = time.time()

            optimizer.zero_grad()
            loss = forward_and_loss(batch, model, loss_fn, device, "train")    
            loss.backward()

            optimizer.synchronize()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), config.get("clip_gradient", 5.0))
            with optimizer.skip_synchronize():
                optimizer.step_and_update_lr()

            learning_rate = optimizer.lr

            # for printing training log
            if global_step % tensorboard_iter == 0:
                if MASTER_WORKER:
                    step_time = time.time() - start_time
                    log.info("Global step: {}, loss: {} ({} seconds)".format(global_step, round(loss.detach().item(), 2), round(step_time, 2)))
                    writer.add_scalar("train_loss", loss.detach().item(), global_step)
                    writer.add_scalar("learning_rate", learning_rate, global_step)
                    writer.add_scalar("gradient_norm", gradient_norm, global_step)

            # for validation
            if global_step % save_iter == 0:
                valid_loss, src_ids, hyp_ids, ref_ids = validate(model, loss_fn, config, valid_dataloader, device)
                if MASTER_WORKER:
                    valid_bleu = write_and_calc_bleu(src_ids, hyp_ids, ref_ids, config, rev_src_vocabs, rev_tgt_vocabs, saved_path, global_step)
                    print(f"valid_loss: {valid_loss}")
                    print(f"valid_bleu: {valid_bleu}")
                    writer.add_scalar("valid_loss", valid_loss, global_step)
                    writer.add_scalar("valid_bleu", valid_bleu.score, global_step)
                    torch.save(model.state_dict(), model_path.format(global_step))

            global_step += 1

            if global_step > total_steps:
                break

        epoch += 1

def main(config_path, saved_path):
    from mpi4py import MPI
    torch.cuda.set_device(hvd.local_rank())

    MPI.COMM_WORLD.Barrier()

    train_config = parse_yaml(config_path)

    if MASTER_WORKER:
        copy_files_to_saved_path(train_config, config_path, saved_path)

    nmt_dataset = NmtDataset(train_config, MASTER_WORKER)

    model = build_model(train_config, DEVICE)
    optimizer = build_optimizer(model)

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    optimizer = build_distributed_optimizer(train_config, optimizer, model)
    loss_fn = build_loss_fn(train_config)

    train_loop(model=model, 
               optimizer=optimizer, 
               loss_fn=loss_fn, 
               dataset=nmt_dataset, 
               config=train_config,
               saved_path=saved_path,
               device=DEVICE)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", required=True)
    parser.add_argument("--saved_path", "-s", required=True)
    args = parser.parse_args()

    main(args.config_path, args.saved_path)
