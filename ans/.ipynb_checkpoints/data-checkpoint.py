import math
from collections.abc import Iterator

import torch


class BatchLoader:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int | None = None,
        shuffle: bool = False,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            dataset: indexable torch Dataset returning either a tuple (input,) or (input, target)
            batch_size: How many samples in a batch
            shuffle: If True, then the data should be randomly reordered on each __iter__ (before each epoch)
            device: torch device name, e.g. 'cpu' or 'cuda:0', 'cuda:1', etc.
        """
        self.dataset = dataset
        self.batch_size = batch_size or len(dataset)
        self.shuffle = shuffle
        self.device = device

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        """
        A single epoch over the dataset

        Returns:
            batch: If unsupervised (i.e. self.y is None), return single element tuple (x[batch_ids],). If supervised
                   (i.e. self.y is torch.Tensor), return the pair (x[batch_ids], y[batch_ids])
        """

        ########################################
        # TODO: implement

        # Recommended approach:
        # 1. Create tensor of indices into self.dataset. If self.shuffle, then the indices should be randomly reodered.
        # 2. Loop over the indices in groups (i.e. batches) of size self.batch_size
        #    2.1 pack items of self.dataset as indexed by the current batch of indices into two torch.Tensors x and y
        #    2.2 `yield` tuple (x, y)
        #    2.2 stop if there are no more batches

        samples = len(self.dataset)
        ind = torch.arange(samples)

        if self.shuffle:
            ind = torch.randperm(samples)

        for indexx in range(0, samples, self.batch_size):
            indexes = ind[indexx : indexx + self.batch_size]

            inputs = []
            targets = []
            has_target = True

            for idx in indexes:
                item = self.dataset[idx]
                
                if isinstance(item, tuple) and len(item) > 1:
                    inputs.append(item[0])
                    targets.append(item[1])
                else:
                    has_target = False
                    inputs.append(item[0] if isinstance(item, tuple) else item)

            x = torch.stack(inputs).to(self.device)

            if has_target and len(targets) > 0:
                y = torch.tensor(targets).to(self.device)
                yield (x, y)
            else:
                yield (x,)
        # ENDTODO
        ########################################

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}:\n    num_batches: {len(self)}\n    batch_size: {self.batch_size}\n"
