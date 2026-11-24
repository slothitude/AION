#!/usr/bin/env python
"""Script to run dreaming experiments and benchmarks."""

import sys
import argparse
sys.path.insert(0, "src")

def run_experiments():
    """Run dreaming experiments."""
    from nnfs.dreaming.experiments import main as exp_main
    exp_main()

def run_benchmarks():
    """Run dreaming benchmarks."""
    from nnfs.dreaming.benchmarks import main as bench_main
    bench_main()

def main():
    parser = argparse.ArgumentParser(description="Run NNFS dreaming experiments")
    parser.add_argument("--experiments", action="store_true", help="Run dreaming experiments")
    parser.add_argument("--benchmarks", action="store_true", help="Run dreaming benchmarks")
    parser.add_argument("--all", action="store_true", help="Run both experiments and benchmarks")
    
    args = parser.parse_args()
    
    if args.all or args.experiments:
        print("Running dreaming experiments...")
        run_experiments()
    
    if args.all or args.benchmarks:
        print("Running dreaming benchmarks...")
        run_benchmarks()
    
    if not (args.all or args.experiments or args.benchmarks):
        parser.print_help()

if __name__ == "__main__":
    main()