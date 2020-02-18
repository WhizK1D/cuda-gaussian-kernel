#!/bin/bash

n=1000000000
h=0.1

if [ $# -eq 3 ]; then
    n=$1
    h=$2
fi

echo -e "Cleaning old build...\n"
make clean

echo -e "Re-building latest version...\n"
make all

