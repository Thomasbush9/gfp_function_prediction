#!/usr/bin/env python3
"""
Wrapper script to run Boltz with proper multiprocessing start method.
Fixes CUDA multiprocessing error when using multiple GPUs.
"""
import sys
import os

# Set environment variable for PyTorch multiprocessing BEFORE any imports
os.environ['PYTORCH_MULTIPROCESSING_START_METHOD'] = 'spawn'

# Set multiprocessing start method before any torch imports
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, which is fine
    pass

# Now try to import and call boltz CLI directly
cli_imported = False
try:
    # Try importing boltz's CLI module
    from boltz.main import cli
    cli_imported = True
except ImportError:
    try:
        from boltz.src.boltz.main import cli
        cli_imported = True
    except ImportError:
        pass

if cli_imported:
    # Call the CLI directly with arguments
    sys.argv = ['boltz'] + sys.argv[1:]
    cli()
else:
    # Fallback to subprocess if direct import fails
    import subprocess
    
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
        boltz_path = 'boltz'
    
    # Build command with environment variable set
    cmd = [boltz_path] + sys.argv[1:]
    
    # Run boltz with environment variable set
    env = os.environ.copy()
    env['PYTORCH_MULTIPROCESSING_START_METHOD'] = 'spawn'
    sys.exit(subprocess.call(cmd, env=env))

