#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
firmware_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
dependency_dir="$firmware_dir/dependencies"
idf_dir="$dependency_dir/esp-idf"
idf_version="v5.5.5"

if [ ! -d "$idf_dir/.git" ]; then
    mkdir -p "$dependency_dir"
    git clone --branch "$idf_version" --depth 1 --recursive --shallow-submodules \
        https://github.com/espressif/esp-idf.git "$idf_dir"
fi

"$idf_dir/install.sh" esp32s3

printf '\nESP-IDF is ready. Run:\n  source %s/export.sh\n  idf.py build\n' "$idf_dir"
