#!/bin/bash

if [ -z "${PROJECT_ID}" ]; then
    echo "Set PROJECT_ID"
    exit 1;
fi

if [ -z "$1" ]; then
    echo "Specify method: n2v, sbm or gnn"
    exit 1;
fi

gcloud beta run jobs deploy "job-${1}" --image "gcr.io/${PROJECT_ID}/${1}-job" --region us-east1 --cpu 2 --memory 4G --timeout=3000 --set-env-vars GRAPH_INDEX=1 --set-env-vars PROJECT_ID="${PROJECT_ID}"

