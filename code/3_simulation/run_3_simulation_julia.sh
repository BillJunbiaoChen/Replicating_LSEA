#!/bin/bash

set -e

echo "Running Julia"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$script_dir"
cd "$script_dir"
julia julia/full_run.jl