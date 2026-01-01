#! /bin/bash

if ! (return 0 2>/dev/null); then
    echo "This script must be sourced: run with"
    echo "    source $(basename $0) ..."
    exit 1
fi

if [[ $# -ne 2 ]]; then
    echo "Usage: $(basename $0) <env_name> <flashinfer_dir> "
    return 1
fi

cpu_arch=$(lscpu | grep "Architecture" | awk '{ print $2 }')
if [[ $cpu_arch == "aarch64" ]]; then
    # for some reason, system torch on aarch64 messes with venv torch
    pip3 uninstall -y torch
fi

env_name=$1
flashinfer_dir=$(realpath $2)

if [[ -z "$(which uv)" ]]; then
    # Install preferred package manager uv to ~/.local/bin (replaces pip and venv)
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv venv --python 3.12 --seed "$env_name"
. "$env_name/bin/activate"

uv pip install setuptools
uv pip install --prerelease=allow --no-build-isolation -e "$flashinfer_dir"
uv pip install nvidia-modelopt
uv pip install --force-reinstall nvidia-cudnn-cu12 nvidia-cudnn-frontend
uv pip install pandas datasets
