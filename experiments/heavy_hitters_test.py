from algorithms.count_min_sketch import CountMinSketch
from algorithms.conservative_cms import ConservativeCMS
from utils.metrics import precision_recall
from collections import Counter
import pandas as pd
import numpy as np

def test_heavy_hitters(file_path: str, threshold_ratio: float = None):
    df = pd.read_csv(file_path)
    
    # 1. Preprocess prices with slight noise to avoid exact duplicates
    df['PRICE'] = df['PRICE'].apply(
        lambda x: round(x + np.random.uniform(-0.05, 0.05), 2)
    )
    price_stream = df['PRICE'].astype(str).tolist()
    
    # 2. Calculate true distribution
    true_counts = Counter(price_stream)
    print("\nPrice Frequency Distribution:")
    print(f"Total unique prices: {len(true_counts)}")
    print(f"Top 5 prices: {true_counts.most_common(5)}")
    
    # 3. Dynamic threshold calculation (if ratio not specified)
    if threshold_ratio is None:
        counts = np.array(list(true_counts.values()))
        q3 = np.percentile(counts, 75)
        threshold = int(q3 * 1.5)  # 1.5x the 75th percentile
        print(f"\nAuto-calculated threshold: {threshold} occurrences")
    else:
        threshold = int(len(price_stream) * threshold_ratio)
    
    # 4. Initialize sketches
    basic_cms = CountMinSketch(width=1000, depth=7)
    cons_cms = ConservativeCMS(width=1000, depth=7)
    
    # 5. Process stream
    for price in price_stream:
        basic_cms.update(price)
        cons_cms.update(price)
    
    # 6. Evaluate
    basic_results = precision_recall(basic_cms, true_counts, threshold)
    cons_results = precision_recall(cons_cms, true_counts, threshold)
    
    print(f"\nHeavy Hitter Results (>{threshold} occurrences):")
    print(f"Basic CMS - Precision: {basic_results[0]:.2f}, Recall: {basic_results[1]:.2f}")
    print(f"Conservative CMS - Precision: {cons_results[0]:.2f}, Recall: {cons_results[1]:.2f}")
    
    # 7. Debug info
    print(f"\nSample problematic cases:")
    overestimated = [
        (p, true_counts[p], basic_cms.estimate(p)) 
        for p in true_counts 
        if basic_cms.estimate(p) > true_counts[p]]
    
    print(f"Top 5 overestimated prices: {sorted(overestimated, key=lambda x: x[2]-x[1], reverse=True)[:5]}")