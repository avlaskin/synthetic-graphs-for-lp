# Running inference on link prediction

## This is how to run this on Google Cloud

1. Setup bucket with name "graphs-data" in GCP project.
(GCP Storage UI)[https://console.cloud.google.com/storage/browser]

2. Setup creadentials json file in the Secret manager. This should be able to read and write to Firebase DB.

3. Build image.

2. Run create job(s).


## Build image

```
build.sh [n2v|sbm|gnn]
```

Parameter indicates the method to be used for these graphs.

## Run image

```
create.sh [n2v|sbm|gnn] [0-9] [faaa|faab|gaaa|gaab|haaa|haab|<your_graph_prefix_here>]
```

First Parameter indicates the method to be used for these graphs.
Second is the index of the graph if you have a set of graphs.
Third is the prefix name for the graph to be used from the bucket.

## This is how to run locally

1. Run prepare.sh script.

2. Run local_run.sh and specify which set of graphs are you running for.

