import sys
import time
import numpy as np
import pandas as pd
from algorithms.count_min_sketch import CountMinSketch
from algorithms.kll_sketch import KLLSketch
from utils.visualizer import plot_memory_usage
from memory_profiler import memory_usage

def get_deep_memory_usage(obj):
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    # Add other type-specific measurements
    return sys.getsizeof(obj) + sum(get_deep_memory_usage(x) for x in getattr(obj, '__dict__', {}).values())

def run_memory_test(stream_sizes: list, trials: int = 3):
    """Run memory benchmarks with statistical significance"""
    results = {
        "CMS": {"mean": [], "std": []},
        "KLL": {"mean": [], "std": []},
        "update_time": {"CMS": [], "KLL": []}
    }
    
    for size in stream_sizes:
        print(f"\n=== Testing stream size: {size:,} ===")
        cms_mem = []
        kll_mem = []
        cms_times = []
        kll_times = []
        
        for trial in range(trials):
            # Initialize fresh sketches for each trial
            cms = CountMinSketch(width=1000, depth=5)
            kll = KLLSketch(k=200)
            
            # Generate test data (mixed numeric and string data)
            numeric_data = np.random.uniform(0, 100, size)
            string_data = [f"item_{x:.2f}" for x in numeric_data]
            
            # Test CMS
            start = time.time()
            for item in string_data:
                cms.update(item)
            cms_times.append(time.time() - start)
            
            # Test KLL
            start = time.time()
            for item in numeric_data:
                kll.update(item)
            kll_times.append(time.time() - start)
            
            # Measure memory using multiple methods
            cms_mem.append(get_deep_memory_usage(cms))
            kll_mem.append(get_deep_memory_usage(kll))
            
            print(f"Trial {trial+1}: CMS={cms_mem[-1]:,}B, KLL={kll_mem[-1]:,}B")
        
        # Store aggregated results
        results["CMS"]["mean"].append(np.mean(cms_mem))
        results["CMS"]["std"].append(np.std(cms_mem))
        results["KLL"]["mean"].append(np.mean(kll_mem))
        results["KLL"]["std"].append(np.std(kll_mem))
        results["update_time"]["CMS"].append(np.mean(cms_times))
        results["update_time"]["KLL"].append(np.mean(kll_times))
    
    # Generate comprehensive report
    print("\n=== Final Memory Results ===")
    df = pd.DataFrame({
        "Stream Size": stream_sizes,
        "CMS Mean (MB)": np.array(results["CMS"]["mean"]) / 1e6,
        "CMS Std": np.array(results["CMS"]["std"]) / 1e6,
        "KLL Mean (MB)": np.array(results["KLL"]["mean"]) / 1e6,
        "KLL Std": np.array(results["KLL"]["std"]) / 1e6,
        "CMS Update (ms/item)": 1000 * np.array(results["update_time"]["CMS"]) / np.array(stream_sizes),
        "KLL Update (ms/item)": 1000 * np.array(results["update_time"]["KLL"]) / np.array(stream_sizes)
    })
    print(df.round(4))
    
    # Visualize results
    plot_memory_usage(
        stream_sizes,
        {
            "CMS": results["CMS"]["mean"],
            "KLL": results["KLL"]["mean"]
        },
        error_bars={
            "CMS": results["CMS"]["std"],
            "KLL": results["KLL"]["std"]
        }
    )
    
    return results