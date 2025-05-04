import yaml
from data.synthetic_generator import generate_stream
from experiments.heavy_hitters_test import test_heavy_hitters
from experiments.quantile_accuracy import run_quantile_experiment
from experiments.memory_benchmark import run_memory_test

def main():
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Generate synthetic data
    stream = generate_stream(int(1e5), dist="powerlaw")  # Smaller stream for testing
    
    # Run experiments
    print("Running Heavy Hitters Test...")
    test_heavy_hitters('data/processed_trades.csv')  # Top 1% prices
    
    print("\nRunning Quantile Accuracy Test...")
    run_quantile_experiment('data/processed_trades.csv')  # Trade size distribution
    
    print("\nRunning Memory Benchmark...")
    run_memory_test(config['experiments']['stream_sizes'])

if __name__ == "__main__":
    main()