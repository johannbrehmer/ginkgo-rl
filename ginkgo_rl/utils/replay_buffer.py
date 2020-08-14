import numpy as np
import torch
import random
import logging

logger = logging.getLogger(__name__)


class History:
    """ Generic replay buffer. Can accommodate arbitrary fields. """

    def __init__(self, max_length=None, dtype=torch.float, device=torch.device("cpu")):
        self.memories = None
        self.max_length = max_length
        self.data_pointer = 0
        self.is_full = False
        if max_length:
            self.memories = np.empty((max_length,), dtype=object)
        else:
            self.memories = np.empty((128,), dtype=object)  # double memory size each time limit is hit
        self.device = device
        self.dtype = dtype

    def store(self, **kwargs):
        self.memories[self.data_pointer] = kwargs
        self.is_full = False
        self.data_pointer += 1
        if self.max_length is not None and self.data_pointer >= self.max_length:
            self.data_pointer = 0
            self.is_full = True
        if self.data_pointer >= self.memories.shape[0] and self.max_length is None:
            # self.memories.resize(self.memories.shape * 2)  # Raises some ValueError
            self.memories = np.resize(self.memories, self.memories.shape[0] * 2)

    # @timeit
    def sample(self, n):
        idx = random.sample(range(len(self)), k=n)
        data_batch = self.memories[idx]
        minibatch = {k: [dic[k] for dic in data_batch] for k in data_batch[0]}

        return idx, None, minibatch

    def rollout(self, n=None):
        """ When n is not None, returns only the last n entries """
        data_batch = self.memories[: len(self)] if n is None else self.memories[len(self) - n : len(self)]
        minibatch = {k: [dic[k] for dic in data_batch] for k in data_batch[0]}
        return minibatch

    def __len__(self):
        if self.max_length is None:
            return self.data_pointer
        else:
            if self.is_full:
                return self.max_length
            else:
                return self.data_pointer

    def clear(self):
        if self.max_length:
            self.memories = np.empty((self.max_length,), dtype=object)
        else:
            self.memories = np.empty((128,), dtype=object)  # double memory size each time limit is hit
        self.data_pointer = 0


class SequentialHistory(History):
    """ Generic replay buffer where each entry represents a sequence of events. Can accommodate arbitrary fields. """

    def __init__(self, max_length=None, dtype=torch.float, device=torch.device("cpu")):
        super().__init__(max_length=max_length, dtype=dtype, device=device)
        self.current_sequence = dict()

    def current_sequence_length(self):
        if len(self.current_sequence) == 0:
            return 0
        else:
            return len(self.current_sequence[list(self.current_sequence.keys())[0]])

    def store(self, **kwargs):
        # Store in temporary sequence buffer
        if self.current_sequence_length() == 0:  # Nothing saved in current sequence
            for key, val in kwargs.items():
                self.current_sequence[key] = [val]
            self.current_sequence["first"] = [True]
        else:
            for key, val in kwargs.items():
                self.current_sequence[key].append(val)
            self.current_sequence["first"].append(False)

    def flush(self):
        """ Push current sequence to ("long-term") memory """
        assert self.current_sequence_length() > 0
        super().store(**self.current_sequence)
        self.current_sequence = dict()
