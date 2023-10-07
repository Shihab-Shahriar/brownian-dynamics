#!/bin/bash

# for codeocean capsule

script /results/out.txt

make
time ./openrand
time ./curand
time ./r123

echo "testing reproducibility"
./openrand > a.txt
./openrand > b.txt
cmp a.txt b.txt


exit