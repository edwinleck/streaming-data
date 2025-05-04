from .count_min_sketch import CountMinSketch

class ConservativeCMS(CountMinSketch):
    def update(self, item: str, count: int = 1) -> None:
        current_min = self.estimate(item)
        for i in range(self.depth):
            pos = self._hash(item, self.seeds[i])
            if self.counters[i][pos] < current_min + count:
                self.counters[i][pos] = current_min + count

