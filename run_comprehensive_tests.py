#!/usr/bin/env python3
"""
Run the comprehensive LBM testing notebook and generate all results.

This script executes the comprehensive_lbm_testing.ipynb notebook,
which tests all LBM CFD components and generates visualization plots
and data files.

Usage:
    python run_comprehensive_tests.py

Output:
    - comprehensive_lbm_testing_executed.ipynb (executed notebook)
    - results/ directory with PNG plots and NPY data files
"""

import subprocess
import sys
import os


def main():
    """Run the comprehensive testing notebook."""
    print("=" * 70)
    print("LBM CFD Comprehensive Testing")
    print("=" * 70)
    print()
    
    # Check if notebook exists
    notebook_path = "comprehensive_lbm_testing.ipynb"
    if not os.path.exists(notebook_path):
        print(f"ERROR: {notebook_path} not found!")
        sys.exit(1)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    print("✓ Results directory created/verified")
    
    # Execute notebook
    print(f"\nExecuting {notebook_path}...")
    print("This will run all tests and may take a few minutes...")
    print()
    
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute", notebook_path,
        "--ExecutePreprocessor.timeout=600",
        "--output", "comprehensive_lbm_testing_executed.ipynb"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        print("\n" + "=" * 70)
        print("SUCCESS! All tests completed.")
        print("=" * 70)
        print()
        print("Generated files:")
        print("  - comprehensive_lbm_testing_executed.ipynb (executed notebook)")
        print()
        
        # List result files
        if os.path.exists("results"):
            result_files = sorted(os.listdir("results"))
            png_files = [f for f in result_files if f.endswith('.png')]
            npy_files = [f for f in result_files if f.endswith('.npy')]
            
            if png_files:
                print(f"  Visualizations ({len(png_files)} PNG files):")
                for f in png_files:
                    print(f"    - results/{f}")
            
            if npy_files:
                print(f"\n  Data arrays ({len(npy_files)} NPY files):")
                for f in npy_files:
                    print(f"    - results/{f}")
        
        print()
        print("To view results, open comprehensive_lbm_testing_executed.ipynb")
        print("in Jupyter Notebook or JupyterLab.")
        
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 70)
        print("ERROR: Notebook execution failed!")
        print("=" * 70)
        print()
        print("STDOUT:")
        print(e.stdout)
        print()
        print("STDERR:")
        print(e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("\nERROR: jupyter not found!")
        print("Please install jupyter: pip install jupyter nbconvert")
        sys.exit(1)


if __name__ == "__main__":
    main()
