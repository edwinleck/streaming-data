import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional

def plot_error_comparison(errors_dict: Dict[str, np.ndarray], 
                         title: str = "Quantile Estimation Error Comparison",
                         quantiles: Optional[list] = None):
    """Enhanced error comparison plot with quantile labels"""
    plt.figure(figsize=(12, 6))
    
    # Default quantile labels if not provided
    if quantiles is None:
        quantiles = [f"Q{int(q*100)}" for q in np.linspace(0.1, 0.9, 5)]
    
    x_pos = np.arange(len(quantiles))
    
    for label, errors in errors_dict.items():
        plt.plot(x_pos, errors, marker='o', label=label, linewidth=2)
    
    plt.xticks(x_pos, quantiles)
    plt.xlabel("Quantile Points")
    plt.ylabel("Absolute Error")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels
    for i in x_pos:
        for label in errors_dict:
            plt.text(i, errors_dict[label][i], 
                   f"{errors_dict[label][i]:.2f}",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

def plot_memory_usage(sizes: list, 
                     memory_usage: Dict[str, list],
                     error_bars: Optional[Dict[str, list]] = None,
                     title: str = "Memory Efficiency Comparison"):
    """Enhanced memory plot with error bar support"""
    plt.figure(figsize=(12, 6))
    
    # Convert sizes to human-readable format
    def human_readable(size):
        if size >= 1e6:
            return f"{size/1e6:.1f}M"
        if size >= 1e3:
            return f"{size/1e3:.0f}K"
        return str(size)
    
    x_labels = [human_readable(size) for size in sizes]
    x_pos = np.arange(len(sizes))
    
    # Plot with error bars if provided
    for label in memory_usage:
        if error_bars and label in error_bars:
            plt.errorbar(x_pos, memory_usage[label], 
                        yerr=error_bars[label],
                        fmt='-o', capsize=5, capthick=2,
                        label=label, linewidth=2)
        else:
            plt.plot(x_pos, memory_usage[label], '-o', label=label, linewidth=2)
        
        # Add value labels
        for i, size in enumerate(sizes):
            plt.text(x_pos[i], memory_usage[label][i], 
                    f"{memory_usage[label][i]/1e6:.1f}MB",
                    ha='center', va='bottom', fontsize=9)
    
    plt.xticks(x_pos, x_labels)
    plt.xlabel("Stream Size (number of items)")
    plt.ylabel("Memory Usage (MB)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Use log scale if memory range is large
    max_mem = max(max(vals) for vals in memory_usage.values())
    min_mem = min(min(vals) for vals in memory_usage.values())
    if max_mem/min_mem > 100:
        plt.yscale('log')
    
    plt.tight_layout()
    plt.show()