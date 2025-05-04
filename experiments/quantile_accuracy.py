from algorithms.kll_sketch import KLLSketch
from algorithms.baseline_methods import ReservoirSampling
import pandas as pd
import numpy as np
import time
from scipy import stats

def run_quantile_experiment(file_path: str, kll_k: int = 200, reservoir_size: int = 1000):
    # 1. Data Loading and Preprocessing
    df = pd.read_csv(file_path)
    
    # Handle zero/negative amounts and add noise to prevent ties
    amount_stream = np.abs(df['AMOUNT']) + np.random.uniform(0, 0.001, len(df))
    
    print(f"\nData Overview:")
    print(f"- Total trades: {len(amount_stream):,}")
    print(f"- Amount range: [{np.min(amount_stream):.4f}, {np.max(amount_stream):.4f}]")
    print(f"- Median amount: {np.median(amount_stream):.4f}")

    # 2. Initialize Sketches
    kll = KLLSketch(k=kll_k)
    reservoir = ReservoirSampling(size=reservoir_size)
    
    # 3. Stream Processing with Timing
    start_time = time.time()
    for amount in amount_stream:
        kll.update(amount)
    kll_time = time.time() - start_time
    
    start_time = time.time()
    for amount in amount_stream:
        reservoir.update(amount)
    reservoir_time = time.time() - start_time
    
    # 4. Quantile Analysis
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    results = []
    
    for q in quantiles:
        true_val = np.quantile(amount_stream, q)
        kll_est = kll.query(q)
        res_est = reservoir.query(q)
        
        results.append({
            'Quantile': q,
            'True': true_val,
            'KLL': kll_est,
            'Reservoir': res_est,
            'KLL Error (%)': 100 * abs(kll_est - true_val) / true_val,
            'Reservoir Error (%)': 100 * abs(res_est - true_val) / true_val
        })

    # 5. Statistical Reporting
    print("\nPerformance Metrics:")
    print(f"- KLL Processing Rate: {len(amount_stream)/kll_time:,.0f} items/sec")
    print(f"- Reservoir Processing Rate: {len(amount_stream)/reservoir_time:,.0f} items/sec")
    
    # 6. Detailed Results
    results_df = pd.DataFrame(results)
    print("\nQuantile Estimation Results:")
    print(results_df.round(4))
    
    # 7. Error Distribution Analysis
    kll_errors = results_df['KLL Error (%)']
    res_errors = results_df['Reservoir Error (%)']
    
    print("\nError Statistics:")
    print(f"KLL - Mean Error: {np.mean(kll_errors):.2f}%, Max: {np.max(kll_errors):.2f}%")
    print(f"Reservoir - Mean Error: {np.mean(res_errors):.2f}%, Max: {np.max(res_errors):.2f}%")
    
    # 8. Significance Testing
    _, p_value = stats.ttest_rel(kll_errors, res_errors)
    print(f"\nPaired t-test p-value: {p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'})")
    
    return {
        'results': results_df,
        'timings': {'KLL': kll_time, 'Reservoir': reservoir_time},
        'errors': {'KLL': kll_errors, 'Reservoir': res_errors}
    }