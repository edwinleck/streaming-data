import random
import numpy as np

class NaiveCounter:
    def __init__(self):
        self.counts = {}
        
    def update(self, item: str) -> None:
        self.counts[item] = self.counts.get(item, 0) + 1
        
    def estimate(self, item: str) -> int:
        return self.counts.get(item, 0)

class ReservoirSampling:
    def __init__(self, size: int = 1000):
        self.size = size
        self.sample = []
        self.count = 0
        
    def update(self, item: float) -> None:
        self.count += 1
        if len(self.sample) < self.size:
            self.sample.append(item)
        else:
            j = random.randint(0, self.count - 1)
            if j < self.size:
                self.sample[j] = item
                
    def query(self, quantile: float) -> float:
        if not self.sample:
            return 0.0
        return np.quantile(self.sample, quantile)