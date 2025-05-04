import mmh3

def hash_functions(width: int, depth: int):
    """Generate independent hash functions for sketches"""
    seeds = [i * 1000 for i in range(depth)]  # Fixed seeds for reproducibility
    return [lambda x, seed=s: mmh3.hash(str(x), seed) % width for s in seeds]