#!/usr/bin/env bash
# Downloads and extracts CUB-200-2011 into /kaggle/working/ (change OUT for another location).
set -e
OUT=${OUT:-/kaggle/working}
wget "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1" -O "$OUT/CUB_200_2011.tgz"
tar -xzf "$OUT/CUB_200_2011.tgz" -C "$OUT"
echo "Dataset extracted to $OUT/CUB_200_2011"
