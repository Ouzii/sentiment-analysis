#!/usr/bin/env bash

inputpath=$(pwd)/input
if [ -z "$( ls -A $inputpath )" ]; then
    printf 'Error: No inputs found' >&2
    exit 1
fi
cd senti-classify
for FILE in "$inputpath"/*.csv; do
    python predict.py $1 $2 $FILE $3 $4;
done