#!/usr/bin/env bash
# Downloads the NASA C-MAPSS turbofan dataset (~12 MB) into data/raw/cmapss.
set -euo pipefail
URL="https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
DEST="$(dirname "$0")/../data/raw"
mkdir -p "$DEST"
cd "$DEST"
curl -fsSL -o cmapss_outer.zip "$URL"
unzip -oq cmapss_outer.zip
unzip -oq "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip" -d cmapss
rm -rf cmapss_outer.zip "6. Turbofan Engine Degradation Simulation Data Set"
ls cmapss
