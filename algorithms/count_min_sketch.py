import numpy as np
import mmh3  # MurmurHash for hashing
import random

class CountMinSketch:
    def __init__(self, width: int, depth: int):
        self.width = width
        self.depth = depth
        self.counters = np.zeros((depth, width), dtype=np.int32)
        self.seeds = [random.getrandbits(32) for _ in range(depth)]

    def _hash(self, item: str, seed: int) -> int:
        return mmh3.hash(item, seed) % self.width

    def update(self, item: str, count: int = 1) -> None:
        for i in range(self.depth):
            pos = self._hash(item, self.seeds[i])
            self.counters[i][pos] += count

    def estimate(self, item: str) -> int:
        return min(self.counters[i][self._hash(item, self.seeds[i])] 
                   for i in range(self.depth))
