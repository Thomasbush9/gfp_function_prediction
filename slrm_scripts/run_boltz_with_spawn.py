#!/usr/bin/env python3
"""
Wrapper script to run Boltz with proper multiprocessing start method.
Fixes CUDA multiprocessing error when using multiple GPUs.
"""
import sys
import os

# CRITICAL: Set environment variable BEFORE any imports
os.environ['PYTORCH_MULTIPROCESSING_START_METHOD'] = 'spawn'

# Set multiprocessing start method before any torch imports
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, which is fine
    pass

# Now try multiple methods to call boltz
# Method 1: Try importing boltz CLI directly
try:
    from boltz.main import cli
    sys.argv = ['boltz'] + sys.argv[1:]
    cli()
    sys.exit(0)
except ImportError:
    pass

# Method 2: Try using runpy to run boltz as a module
try:
    import runpy
    sys.argv = ['boltz'] + sys.argv[1:]
    runpy.run_module('boltz', run_name='__main__')
    sys.exit(0)
except (ImportError, ModuleNotFoundError):
    pass

# Method 3: Try using python -m boltz
import subprocess
import shutil

# Find python executable
python_exe = sys.executable

# Try running as module first
try:
    cmd = [python_exe, '-m', 'boltz'] + sys.argv[1:]
    env = os.environ.copy()
    env['PYTORCH_MULTIPROCESSING_START_METHOD'] = 'spawn'
    sys.exit(subprocess.call(cmd, env=env))
except Exception:
    pass

# Method 4: Fallback to boltz binary with environment variable
boltz_path = shutil.which('boltz')
if not boltz_path:
    # Try to find it in the conda environment
    conda_env = os.environ.get('CONDA_PREFIX', '')
    if conda_env:
        potential_path = os.path.join(conda_env, 'bin', 'boltz')
        if os.path.isfile(potential_path):
            boltz_path = potential_path

if not boltz_path:
    boltz_path = 'boltz'

# Run boltz with environment variable set
cmd = [boltz_path] + sys.argv[1:]
env = os.environ.copy()
env['PYTORCH_MULTIPROCESSING_START_METHOD'] = 'spawn'
sys.exit(subprocess.call(cmd, env=env))

