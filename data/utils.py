import numpy as np
from typing import Dict
import torch

from . import  openwebtext2

PREPARE_GET_DATASET_MAP = {
    "owt2": (openwebtext2.prepare_openwebtext2_data, openwebtext2.get_openwebtext2_data)
}


def prepare_dataset(args):
    """ Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
     contained in its own pythin file. The expected format at the moment is a disctionary of np.memmap
     containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data. """
    return PREPARE_GET_DATASET_MAP[args.dataset][0](args)

def get_dataset(args) -> Dict[str, np.ndarray]:
    """ Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
     contained in its own python file. The expected format at the moment is a dictionary of np.memmap
     containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data. """
    return PREPARE_GET_DATASET_MAP[args.dataset][1](args)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, sequence_length):
        super().__init__()
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        total_length = len(self.data)
        # chunk the data into sequences of length `sequence_length`
        # NOTE: we discard the last remainding sequence if it's not of length `sequence_length`
        return (total_length - 1) // self.sequence_length

    def __getitem__(self, idx):
        seq_length = self.sequence_length
        idx = idx * seq_length
        x = torch.from_numpy((self.data[idx : idx + seq_length]).astype(np.int64))

        y = torch.from_numpy(
            (self.data[idx + 1 : idx + 1 + seq_length]).astype(np.int64)
        )
        return x, y


def get_dataloader(data, sequence_length, batch_size, seed=0, distributed_backend=None):
    """Create a DataLoader for the given data. If distributed_backend is provided and is truly
    distributed (world size > 1), the DataLoader will be created with a DistributedSampler that
    splits the data across the processes (in conjunction with DDP).
    Otherwise, use a RandomSampler with the specified seed.

    Returns both the dataloader and the sampler.
    """
    dataset = Dataset(data, sequence_length=sequence_length)
    if distributed_backend and distributed_backend.get_world_size() > 1:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            shuffle=True,
            seed=seed,
        )
    else:
        g = torch.Generator()
        g.manual_seed(seed)
        sampler = torch.utils.data.RandomSampler(
            dataset, replacement=False, generator=g
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=4,
    )
    return loader, sampler
