#!bin/bash

inputpath=$(pwd)/input
cd senti-classify
for FILE in "$inputpath"/*.csv; do
    python predict.py $1 $2 $FILE $3 $4;
done