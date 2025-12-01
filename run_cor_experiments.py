#!/usr/bin/env python
"""
Run comprehensive C&OR experiments for MOIWOF paper.

This script runs the full experimental suite with proper settings for
publication-quality results.

Usage:
    python run_cor_experiments.py [--quick]
    
Options:
    --quick     Run with reduced settings for testing
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from wms_optlab.experiments.run_experiments_v2 import ComprehensiveExperimentRunner


def run_experiments(quick_mode: bool = False):
    """Run experiments with appropriate settings."""
    
    if quick_mode:
        # Quick test settings
        config = {
            'num_runs': 2,
            'max_generations': 20,
            'population_size': 30,
            'include_large': False,
            'algorithms': ['MOIWOF', 'MOEA/D', 'NSGA-II', 'Sequential', 'ABC', 'Random']
        }
        output_dir = 'experiments_output_quick'
    else:
        # Full publication settings
        config = {
            'num_runs': 10,  # Reduced from 20 for practical runtime
            'max_generations': 80,
            'population_size': 60,
            'include_large': True,
            'algorithms': [
                'MOIWOF', 'MOIWOF-NoADS', 'MOIWOF-NoCCL', 'MOIWOF-NoAdapt',
                'NSGA-II', 'MOEA/D', 'Sequential', 'ABC', 'Random'
            ]
        }
        output_dir = 'experiments_output_cor'
    
    print(f"Running experiments in {'quick' if quick_mode else 'full'} mode")
    print(f"Output directory: {output_dir}")
    
    runner = ComprehensiveExperimentRunner(output_dir)
    results = runner.run_comprehensive_experiments(**config)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}/")
    
    return results


if __name__ == "__main__":
    quick = '--quick' in sys.argv
    run_experiments(quick_mode=quick)
