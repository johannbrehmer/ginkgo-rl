import numpy as np


class AffineNormalizer():
    def __init__(self, min_initial=None, max_initial=None, hard_min=None, hard_max=None, epsilon=0.001):
        self._min = min_initial
        self._max = max_initial
        self._epsilon = epsilon
        self._hard_min = hard_min
        self._hard_max = hard_max

    def update(self, value):
        if self._hard_min is not None or self._hard_max is not None:
            value = np.clip(value, self._hard_min, self._hard_max)

        if self._min is None:
            self._min = value - self._epsilon
        if self._max is None:
            self._max = value + self._epsilon

        self._min = min(self._min, value)
        self._max = max(self._max, value)

    def evaluate(self, value):
        if self._hard_min is not None or self._hard_max is not None:
            value = np.clip(value, self._hard_min, self._hard_max)
        return np.clip((value - self._min) / (self._max - self._min), 0.0, 1.0)
