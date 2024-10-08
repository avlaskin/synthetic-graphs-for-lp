#!/bin/bash

if [ -z "${PROJECT_ID}" ]; then
    echo "Set PROJECT_ID"
    exit 1;
fi

if [ -z "$1" ]; then
    
    echo "Specify method: n2v, sbm or gnn"
    exit 1;
fi

./prepare.sh "${1}"

export CLOUD_PROJECT=${PROJECT_ID}

cd ./jobs

rm *~
rm -r __pycache__

echo "Setting up environment."

export CONFIG_NAME="${1}-job"


gcloud config set project ${CLOUD_PROJECT}
gcloud config set run/region us-central1

echo "Building " ${CLOUD_PROJECT} "/" ${CONFIG_NAME}

sudo docker build --tag ${CONFIG_NAME}:python .
gcloud builds submit --tag gcr.io/${CLOUD_PROJECT}/${CONFIG_NAME}
echo "Deploying"



