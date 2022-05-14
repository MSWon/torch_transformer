import torch
import horovod.torch as hvd

from nmt.dataset.utils import build_vocab, read_lines
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence


_bucket_size = 10


class NmtDataset(object):
    def __init__(self, config, master_worker):
        self.nmt_train_dataset = NmtDatasetHelper(config, "train", master_worker)
        self.nmt_valid_dataset = NmtDatasetHelper(config, "valid", master_worker)
        self.rev_src_vocabs = self.nmt_train_dataset.rev_src_vocabs
        self.rev_tgt_vocabs = self.nmt_train_dataset.rev_tgt_vocabs
        self.batch_size = config.get("batch_size", 1000)
        self.src_max_len = config.get("src_max_len", 100)
        self.tgt_max_len = config.get("tgt_max_len", 100)

        max_len = max(self.src_max_len, self.tgt_max_len)
        self.bucket_boundaries = [i for i in range(10, max_len + 1, int(max_len / _bucket_size))]
        self.bucket_batch_sizes = [max(1, self.batch_size // length) for length in self.bucket_boundaries + [max_len]]

    def define_train_iterator(self):
        train_sampler = DistributedSampler(dataset=self.nmt_train_dataset,
                                           shuffle=True,
                                           num_replicas=hvd.size(),
                                           rank=hvd.rank())

        bucketing = BucketSampler(sampler=train_sampler, 
                                  bucket_boundaries=self.bucket_boundaries, 
                                  bucket_batch_sizes=self.bucket_batch_sizes,
                                  src_max_len=self.src_max_len,
                                  tgt_max_len=self.tgt_max_len,
                                  drop_last=True)

        nmt_dataloader = DataLoader(dataset=self.nmt_train_dataset,
                                    batch_sampler=bucketing,
                                    collate_fn=self.collate_fn,
                                    pin_memory=True)

        return nmt_dataloader, train_sampler

    def define_valid_iterator(self):
        nmt_dataloader = DataLoader(dataset=self.nmt_valid_dataset, 
                                    batch_size=100,
                                    drop_last=False,
                                    collate_fn=self.collate_fn,
                                    pin_memory=True)
        return nmt_dataloader

    def filter_long_sentence(self, batch):
        src_input_idx, tgt_input_idx, tgt_output_idx = [], [], []
        for x in batch:
            src_input_idx.append(x["src_input_idx"])
            tgt_input_idx.append(x["tgt_input_idx"])
            tgt_output_idx.append(x["tgt_output_idx"])

        result_batch = {
            "src_input_idx": src_input_idx,
            "tgt_input_idx": tgt_input_idx,
            "tgt_output_idx": tgt_output_idx
        }
        return result_batch

    def pad_sentence(self, batch):
        result_batch = {
            "src_input_idx": pad_sequence(batch["src_input_idx"], batch_first=True),
            "tgt_input_idx": pad_sequence(batch["tgt_input_idx"], batch_first=True),
            "tgt_output_idx": pad_sequence(batch["tgt_output_idx"], batch_first=True)
        }
        return result_batch

    def collate_fn(self, batch):
        batch = self.filter_long_sentence(batch)
        batch = self.pad_sentence(batch)
        return batch
        
class NmtDatasetHelper(Dataset):
    def __init__(self, config, mode, master_worker):
        assert mode in ["train", "valid"]

        if mode == "train":
            self.src_corpus_path = config.get("src_train_corpus_path")
            self.tgt_corpus_path = config.get("tgt_train_corpus_path")
        elif mode == "valid":
            self.src_corpus_path = config.get("src_valid_corpus_path")
            self.tgt_corpus_path = config.get("tgt_valid_corpus_path")
       
        self.src_vocab_path = config.get("src_vocab_path")
        self.tgt_vocab_path = config.get("tgt_vocab_path")

        self.src_unk_symbol = config.get("src_unk_symbol")
        self.tgt_unk_symbol = config.get("tgt_unk_symbol")

        self.src_use_bos_symbol = config.get("src_use_bos_symbol")
        self.src_use_eos_symbol = config.get("src_use_eos_symbol")
        self.tgt_use_bos_symbol = config.get("tgt_use_bos_symbol")
        self.tgt_use_eos_symbol = config.get("tgt_use_eos_symbol")
        
        self.src_bos_symbol = config.get("src_bos_symbol")
        self.src_eos_symbol = config.get("src_eos_symbol")
        self.tgt_bos_symbol = config.get("tgt_bos_symbol")
        self.tgt_eos_symbol = config.get("tgt_eos_symbol")
        
        self.src_vocabs, self.rev_src_vocabs = build_vocab(self.src_vocab_path)
        self.tgt_vocabs, self.rev_tgt_vocabs = build_vocab(self.tgt_vocab_path)
        
        self.src_input_idx = read_lines(self.src_corpus_path, 
                                        self.src_vocabs, 
                                        self.src_unk_symbol,
                                        self.src_use_bos_symbol,
                                        self.src_use_eos_symbol,
                                        self.src_bos_symbol,
                                        self.src_eos_symbol,
                                        master_worker)

        self.tgt_input_idx = read_lines(self.tgt_corpus_path, 
                                        self.tgt_vocabs, 
                                        self.tgt_unk_symbol,
                                        self.tgt_use_bos_symbol,
                                        self.tgt_use_eos_symbol,
                                        self.tgt_bos_symbol,
                                        self.tgt_eos_symbol,
                                        master_worker)
        
        assert len(self.src_input_idx) == len(self.tgt_input_idx)
    
    def __len__(self):
        return len(self.src_input_idx)
    
    def __getitem__(self, idx):
        result =  {
            "src_input_idx":torch.as_tensor(self.src_input_idx[idx], dtype=torch.long), 
            "tgt_input_idx": torch.as_tensor(self.tgt_input_idx[idx][:-1], dtype=torch.long),
            "tgt_output_idx": torch.as_tensor(self.tgt_input_idx[idx][1:], dtype=torch.long)
        }
        return result

class BucketSampler(Sampler):
    def __init__(self, sampler, 
                 bucket_boundaries, bucket_batch_sizes, 
                 src_max_len, tgt_max_len, drop_last=False):
        self.bucket_boundaries = bucket_boundaries
        self.bucket_batch_sizes = bucket_batch_sizes
        self.sampler = sampler
        self.drop_last = drop_last
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len

    def __iter__(self):
        buckets = [[] for i in range(len(self.bucket_boundaries))]
        for idx in self.sampler:
            # sampler에서 dataset을 가져와서 data의 length를 잼
            if hasattr(self.sampler, "data_source"):
                src_length = self.sampler.data_source[idx]["src_input_idx"].size(0)
                tgt_length = self.sampler.data_source[idx]["tgt_input_idx"].size(0)
            else:
                src_length = self.sampler.dataset[idx]["src_input_idx"].size(0)
                tgt_length = self.sampler.dataset[idx]["tgt_input_idx"].size(0)

            # filter by length
            if src_length > self.src_max_len:
                continue
            if tgt_length > self.tgt_max_len:
                continue

            length = max(src_length, tgt_length)

            for i, boundary in enumerate(self.bucket_boundaries):
                if length <= boundary:
                    buckets[i].append(idx)
                    if len(buckets[i]) == self.bucket_batch_sizes[i]:
                        yield buckets[i]
                        buckets[i] = []
                    break

        if not self.drop_last:
            for bucket in filter(len, buckets):
                yield bucket

    def __len__(self):
        raise NotImplementedError("BucketSampler cannot know the total number of batches.")
