"""
bitbrush_demo.py

Demonstrates and benchmarks the BitBrush utility using matplotlib.

This script evaluates the performance of various bit pattern operations
and visualizes their execution time using bar plots.
"""

import time
import matplotlib.pyplot as plt
from bitbrush import BitBrush

def benchmark_operation(name: str, func, *args) -> float:
    """
    Measures execution time for a given BitBrush operation.

    Args:
        name (str): Name of the operation.
        func (callable): The function to benchmark.
        *args: Arguments to pass to the function.

    Returns:
        float: Execution time in milliseconds.
    """
    start = time.perf_counter()
    for _ in func(*args):
        pass
    end = time.perf_counter()
    return (end - start) * 1000


def plot_benchmark(results: dict):
    """
    Plots the benchmark results using matplotlib.

    Args:
        results (dict): Mapping of operation names to execution times in ms.
    """
    operations = list(results.keys())
    times = list(results.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(operations, times, color='skyblue', edgecolor='black')
    plt.title("BitBrush Operation Benchmark")
    plt.ylabel("Execution Time (ms)")
    plt.xlabel("Operation")
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar, time_ms in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{time_ms:.2f} ms", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def main():
    brush = BitBrush(width=32)

    results = {
        "sweep_ones": benchmark_operation("sweep_ones", brush.sweep_ones),
        "sweep_zeros": benchmark_operation("sweep_zeros", brush.sweep_zeros),
        "toggle_sparse": benchmark_operation("toggle_sparse", brush.toggle_sparse, 3),
        "scan_patterns": benchmark_operation("scan_patterns", brush.scan_patterns),
        "mirror_mask (single)": benchmark_operation(
            "mirror_mask",
            lambda: (brush.mirror_mask(x) for x in range(1000))
        )
    }

    plot_benchmark(results)


if __name__ == "__main__":
    main()
