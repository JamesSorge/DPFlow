#!/bin/bash

DIR_KUBRIC_1K="datasets/kubric_nk/1k"
DIR_CONFIGS="datasets/kubric_nk/configs/1k"
DIR_FORWARD_FLOW="datasets/kubric_nk/forward_flow/1k"
DIR_RGBA="datasets/kubric_nk/rgba/1k"
if [ ! -d "$DIR_KUBRIC_1K" ]; then
    echo "Creating kubric_nk/1k directory"
    mkdir -p ./datasets/kubric_nk/1k
    echo "Done."
else
    echo "kubric_nk/1k directory exists."
fi

DIR_ITERABLE_0="0"
DIR_ITERABLE_00="00"
MAX=29
THRESHOLD=10

for i in $(seq 0 $MAX)
do
    echo "i var: $i"
    if [ "$i" -lt "$THRESHOLD" ]; then
        mkdir -p "${DIR_KUBRIC_1K}/${DIR_ITERABLE_00}${i}"
        # mv content from directories in Configs
        mv ./${DIR_CONFIGS}/${DIR_ITERABLE_00}${i}/*.json "${DIR_KUBRIC_1K}/${DIR_ITERABLE_00}${i}"
        # mv content from directories in RGBA
        mv ./${DIR_RGBA}/${DIR_ITERABLE_00}${i}/*.png "${DIR_KUBRIC_1K}/${DIR_ITERABLE_00}${i}"
        # mv content from directories in Forward Flow
        mv ./${DIR_FORWARD_FLOW}/${DIR_ITERABLE_00}${i}/*.png "${DIR_KUBRIC_1K}/${DIR_ITERABLE_00}${i}"
    else
        mkdir -p "${DIR_KUBRIC_1K}/${DIR_ITERABLE_0}${i}"
        # mv content from directories in Configs
        mv ./${DIR_CONFIGS}/${DIR_ITERABLE_0}${i}/*.json "${DIR_KUBRIC_1K}/${DIR_ITERABLE_0}${i}"
        # mv content from directories in RGBA
        mv ./${DIR_RGBA}/${DIR_ITERABLE_0}${i}/*.png "${DIR_KUBRIC_1K}/${DIR_ITERABLE_0}${i}"
        # mv content from directories in Forward Flow
        mv ./${DIR_FORWARD_FLOW}/${DIR_ITERABLE_0}${i}/*.png "${DIR_KUBRIC_1K}/${DIR_ITERABLE_0}${i}"
    fi
done