#!/bin/bash

echo "Preparing the code for the run."

if [ -z "$1" ]; then
    echo "Specify method: n2v, sbm or gnn"
    exit 1;
fi

rm -r ./module/*
rm Dockerfile
rm requirements.txt

echo "Copying the common code"
cp ../../../common/* ./module/

if [ "n2v" == "$1" ]; then
    echo "Copying N2V code to module"
    cp ../../../node2vec/*.py ./module/
    cp ../main_n2v.py ./main.py
    cp ../../../node2vec/requirements.txt ./
    cp ../../../node2vec/Dockerfile ./
fi

if [ "gnn" == "$1" ]; then
    echo "Copying GNN code to module"
    cp ../../../gnn/*.py ./module/
    cp ../main_gnn.py ./main.py
    cp ../../../gnn/requirements.txt ./
    cp ../../../gnn/Dockerfile ./
fi

if [ "sbm" == "$1" ]; then
    echo "Copying N2V code to module"
    cp ../../../sbm/*.py ./module/
    cp ../main_sbm.py ./main.py
    cp ../../../sbm/requirements.txt ./
    cp ../../../sbm/Dockerfile ./
fi

