#!/bin/bash

echo "Preparing the code for the run."

if [ -z "$1" ]; then
    echo "Specify method: n2v, sbm or gnn"
    exit 1;
fi

rm -r ./jobs/module/*
rm ./jobs/Dockerfile
rm ./jobs/requirements.txt

echo "Copying the common code"
cp ../../common/* ./jobs/module/

if [ "n2v" == "$1" ]; then
    echo "Copying N2V code to module"
    cp ../../node2vec/*.py ./jobs/module/
    cp main_n2v.py ./jobs/main.py
    cp ../../node2vec/requirements.txt ./jobs/
    cp ../../node2vec/Dockerfile ./jobs/
fi

if [ "gnn" == "$1" ]; then
    echo "Copying GNN code to module"
    cp ../../gnn/*.py ./jobs/module/
    cp main_gnn.py ./jobs/main.py
    cp ../../gnn/requirements.txt ./jobs/
    cp ../../gnn/Dockerfile ./jobs/
fi

if [ "sbm" == "$1" ]; then
    echo "Copying SBM code to module"
    cp ../../sbm/*.py ./jobs/module/
    cp main_sbm.py ./jobs/main.py
    cp ../../sbm/requirements.txt ./jobs/
    cp ../../sbm/Dockerfile ./jobs/
fi

