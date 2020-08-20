class AffineNormalizer():
    def __init__(self, min_initial=0.0, max_initial=0.0):
        self._min = min_initial
        self._max = max_initial

    def update(self, value):
        self._min = min(self._min, value)
        self._max = min(self._max, value)

    def evaluate(self, value):
        return (value - self._min) / (self._max - self._min)
