#!/bin/bash

export CLOUD_PROJECT=alexey-api
export CONFIG_NAME="${1}-job"


if [ -z "${PROJECT_ID}" ]; then
    echo "Set PROJECT_ID"
    exit 1;
fi

if [ -z "$1" ]; then
    echo "Specify method: n2v, sbm or gnn"
    exit 1;
fi

if [ -z "$2" ]; then
    
    echo "Specify graph index, typically - [0; 10]"
    exit 1;
fi

#epochs="16"
#if [ -z "$3" ]; then
#    epochs="${3}"
#fi

# Cheap regions
timestamp=$(date +%s)
regions=(northamerica-northeast1 northamerica-northeast2 us-west1 us-central1 us-east5 me-west1 us-south1 me-central1 asia-south2)
echo "Creating a job" ${CLOUD_PROJECT} "/" ${CONFIG_NAME} " - " $CONFIG_NAME-${2}-${timestamp}
echo "Epochs " ${3}

if [ "gnn" == "$1" ]; then
    gcloud beta run jobs create $CONFIG_NAME-${2}-${timestamp} --image gcr.io/${CLOUD_PROJECT}/${CONFIG_NAME} --set-env-vars GRAPH_INDEX=${2} --set-env-vars PROJECT_ID="${PROJECT_ID}" --set-env-vars EPOCHS=${3} --set-env-vars TASK_PREFIX=gaab --cpu 8 --memory 5G --execute-now --tasks 10 --task-timeout 14400 --region "${regions[$2]}" --max-retries 1

else
    gcloud beta run jobs create $CONFIG_NAME-${2}-${timestamp} --image gcr.io/${CLOUD_PROJECT}/${CONFIG_NAME} --set-env-vars GRAPH_INDEX=${2} --set-env-vars PROJECT_ID="${PROJECT_ID}" --set-env-vars EPOCHS=${3} --set-env-vars TASK_PREFIX=gaab --cpu 8 --memory 5G --execute-now --tasks 2 --task-timeout 14400 --region "${regions[$2]}"
    
fi


