import numpy as np
import random
import math

class KLLSketch:
    def __init__(self, k: int = 400):
        self.k = k
        self.compactors = [[]]  # Hierarchy of compactors
        self.size = 0
        
    def update(self, item):
        self.compactors[0].append(item)
        if len(self.compactors[0]) >= self.k:
            self._compress()  # Verify this creates new compactors when needed
        
    def _compress(self) -> None:
        for level in range(len(self.compactors)):
            if len(self.compactors[level]) >= self.k:
                # Sort and compact
                self.compactors[level].sort()
                compacted = [x for i,x in enumerate(sorted(self.compactors[level])) 
                        if i % 2 == 0 or random.random() < 0.5]
                
                # Add new level if needed
                if level + 1 == len(self.compactors):
                    self.compactors.append([])
                
                self.compactors[level + 1].extend(compacted)
                self.compactors[level] = []
    
    def query(self, quantile: float) -> float:
        if not 0 <= quantile <= 1:
            raise ValueError("Quantile must be between 0 and 1")
            
        # Collect all elements with weights
        elements = []
        for level in range(len(self.compactors)):
            weight = 2 ** level
            elements.extend([(x, weight) for x in self.compactors[level]])
        
        if not elements:
            return 0.0
            
        elements.sort()
        total_weight = sum(w for _, w in elements)
        target = quantile * total_weight
        
        cum_weight = 0
        for x, w in elements:
            cum_weight += w
            if cum_weight >= target:
                return x
        return elements[-1][0]