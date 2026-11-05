#!/bin/sh
# Compiles the shaders to SPIR-V and validates them. Only needed after editing a shader:
# the results in spv/ are committed, so building the app needs no shader compiler.
set -eu
cd "$(dirname "$0")"
mkdir -p spv
for name in trace temporal atrous resolve tonemap; do
    glslangValidator --quiet --target-env vulkan1.2 -o "spv/$name.spv" "$name.comp"
    spirv-val --target-env vulkan1.2 "spv/$name.spv"
done
echo "Compiled and validated all shaders"
