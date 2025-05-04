import numpy as np

def generate_stream(n: int, dist: str = "powerlaw") -> list:
    """Generate synthetic data streams."""
    if dist == "powerlaw":
        # Heavy-tailed distribution (common in real-world)
        return np.random.zipf(a=1.5, size=n).tolist()
    elif dist == "normal":
        return np.random.normal(loc=0, scale=1, size=n).tolist()
    else:
        raise ValueError(f"Unknown distribution: {dist}")