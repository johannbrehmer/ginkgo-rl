class AffineNormalizer():
    def __init__(self, min_initial=None, max_initial=None, epsilon=0.01):
        self._min = min_initial
        self._max = max_initial
        self._epsilon = epsilon

    def update(self, value):
        if self._min is None:
            self._min = value - self._epsilon
        if self._max is None:
            self._max = value + self._epsilon

        self._min = min(self._min, value)
        self._max = max(self._max, value)

    def evaluate(self, value):
        return (value - self._min) / (self._max - self._min)
