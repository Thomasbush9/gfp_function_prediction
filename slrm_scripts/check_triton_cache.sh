#!/bin/bash
# Helper script to check and manage Triton cache versions

set -euo pipefail

# Use home directory for cache storage
CACHE_BASE="$HOME/.boltz_triton_caches"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [COMMAND]

Commands:
    list        List all available cache versions
    latest      Show the latest cache version
    clean       Remove old cache versions (keeps latest 3)
    info        Show detailed info about current environment
    check       Check if a valid cache exists for current environment

EOF
    exit 1
}

# Function to get current version string
get_current_version() {
    module load python/3.12.8-fasrc01 gcc/14.2.0-fasrc01 cuda/12.9.1-fasrc01 cudnn/9.10.2.21_cuda12-fasrc01 2>/dev/null || true
    
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    mamba activate /n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/boltz 2>/dev/null || true
    
    BOLTZ_VERSION=$(python -c "import boltz; print(boltz.__version__)" 2>/dev/null || echo "unknown")
    TRITON_VERSION=$(python -c "import triton; print(triton.__version__)" 2>/dev/null || echo "unknown")
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
    GPU_ARCH=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | tr ' ' '_' || echo "unknown")
    
    echo "${BOLTZ_VERSION}_triton${TRITON_VERSION}_pytorch${PYTORCH_VERSION}_cuda${CUDA_VERSION}_${GPU_ARCH}"
}

# Command: list
cmd_list() {
    echo "Available cache versions:"
    echo "========================="
    if [ -d "$CACHE_BASE" ]; then
        for cache_dir in "$CACHE_BASE"/*/ ; do
            if [ -d "$cache_dir" ] && [ "$(basename "$cache_dir")" != "latest" ]; then
                cache_name=$(basename "$cache_dir")
                cache_size=$(du -sh "$cache_dir" 2>/dev/null | cut -f1 || echo "unknown")
                cache_date=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$cache_dir" 2>/dev/null || stat -c "%y" "$cache_dir" 2>/dev/null | cut -d' ' -f1,2 || echo "unknown")
                echo "  $cache_name"
                echo "    Size: $cache_size"
                echo "    Created: $cache_date"
                if [ -f "$cache_dir/cache_info.txt" ]; then
                    echo "    Info:"
                    sed 's/^/      /' "$cache_dir/cache_info.txt"
                fi
                echo ""
            fi
        done
    else
        echo "No cache directory found at $CACHE_BASE"
    fi
}

# Command: latest
cmd_latest() {
    if [ -L "${CACHE_BASE}/latest" ]; then
        target=$(readlink "${CACHE_BASE}/latest")
        echo "Latest cache points to:"
        echo "  $(basename "$target")"
        if [ -f "$target/cache_info.txt" ]; then
            echo ""
            cat "$target/cache_info.txt"
        fi
    else
        echo "No 'latest' symlink found"
    fi
}

# Command: info
cmd_info() {
    echo "Current Environment:"
    echo "==================="
    CURRENT_VERSION=$(get_current_version)
    echo "Expected cache version: $CURRENT_VERSION"
    echo ""
    
    EXPECTED_CACHE="${CACHE_BASE}/${CURRENT_VERSION}"
    if [ -d "$EXPECTED_CACHE" ]; then
        echo "✓ Matching cache exists"
        echo "  Location: $EXPECTED_CACHE"
        echo "  Size: $(du -sh "$EXPECTED_CACHE" 2>/dev/null | cut -f1)"
    else
        echo "✗ No matching cache found"
        echo "  Run prewarm_triton_cache.slrm to create it"
    fi
}

# Command: check
cmd_check() {
    CURRENT_VERSION=$(get_current_version)
    EXPECTED_CACHE="${CACHE_BASE}/${CURRENT_VERSION}"
    
    if [ -d "$EXPECTED_CACHE" ]; then
        echo "$EXPECTED_CACHE"
        exit 0
    else
        echo "No matching cache found for version: $CURRENT_VERSION" >&2
        exit 1
    fi
}

# Command: clean
cmd_clean() {
    echo "Cleaning old cache versions (keeping latest 3)..."
    if [ -d "$CACHE_BASE" ]; then
        # List directories by modification time, skip latest 3
        find "$CACHE_BASE" -maxdepth 1 -type d ! -name "latest" ! -path "$CACHE_BASE" -printf "%T@ %p\n" | \
            sort -rn | \
            tail -n +4 | \
            cut -d' ' -f2- | \
            while read -r old_cache; do
                echo "Removing: $(basename "$old_cache")"
                rm -rf "$old_cache"
            done
        echo "Cleanup complete"
    else
        echo "No cache directory found"
    fi
}

# Main
COMMAND="${1:-}"

case "$COMMAND" in
    list)
        cmd_list
        ;;
    latest)
        cmd_latest
        ;;
    info)
        cmd_info
        ;;
    check)
        cmd_check
        ;;
    clean)
        cmd_clean
        ;;
    *)
        usage
        ;;
esac

