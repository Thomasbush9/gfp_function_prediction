#!/usr/bin/env python3
"""
Wrapper script to run Boltz with proper multiprocessing start method.
Fixes CUDA multiprocessing error when using multiple GPUs.
"""
import sys
import multiprocessing
import subprocess
import os

def main():
    # Set multiprocessing start method to 'spawn' before any CUDA operations
    # This must be done before importing torch or any CUDA-dependent modules
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, which is fine
        pass
    
    # Get all arguments except script name
    args = sys.argv[1:]
    
    # Find boltz command in PATH
    boltz_path = None
    for path_dir in os.environ.get('PATH', '').split(':'):
        potential_path = os.path.join(path_dir, 'boltz')
        if os.path.isfile(potential_path) and os.access(potential_path, os.X_OK):
            boltz_path = potential_path
            break
    
    if not boltz_path:
        # Try to find it in the conda environment
        conda_env = os.environ.get('CONDA_PREFIX', '')
        if conda_env:
            potential_path = os.path.join(conda_env, 'bin', 'boltz')
            if os.path.isfile(potential_path):
                boltz_path = potential_path
    
    if not boltz_path:
        # Fallback to just 'boltz' and let the shell find it
        boltz_path = 'boltz'
    
    # Build command
    cmd = [boltz_path] + args
    
    # Run boltz with the same environment
    sys.exit(subprocess.call(cmd))

if __name__ == '__main__':
    main()

