#!/usr/bin/env bash
# Prints version numbers for BirdFlowPy's Python dependencies as actually
# installed in the running environment (e.g. inside BirdFlowContainer on
# Unity). Intended to answer "what version of X does this container have"
# questions like https://github.com/birdflow-science/BirdFlowPy/issues/5 --
# run it inside the container (e.g. from an RStudio session's Python
# console via `reticulate::repl_python()`, or `apptainer exec
# BirdFlowContainer.sif bash dev/print_container_versions.sh`).
set -euo pipefail

echo "python: $(python --version 2>&1)"

python - <<'EOF'
packages = ["jax", "jaxlib", "haiku", "optax", "scipy", "h5py", "numpy"]
for name in packages:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "unknown (no __version__ attribute)")
        print(f"{name}: {version}")
    except ImportError:
        print(f"{name}: not installed")
EOF
