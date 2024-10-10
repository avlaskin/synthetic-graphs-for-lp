# Running inference on link prediction

## This is how to run this on Google Cloud

1. Setup bucket with name "graphs-data" in GCP project.
[GCP Storage UI](https://console.cloud.google.com/storage/browser)

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

Local run is supported for all algorithms. Although SBM requires a docker container
to be run locally for that setup. This is because graphtool require C++ dependencies and
much more easier to use it inside the container than building it from scratch.

Python3.8 environment is triky, so here is the full instruction for GNN env:

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.8
sudo apt-get install python3.8-distutils
```

0. Create a virtual environment with specific verison of Python.

  - N2V can work with any python version > 3.9/
  - GNN requires python3.8.
  - SBM needs a docker container to run locally.

```
mkdir env3X
cd env3X
virtualenv -p python3.X .
source ./bin/activate
cd ..
```

1. Run prepare.sh script.

```
./prepare.sh [n2v|gnn]
```

3. Install dependencies for you environment:

```
pip install -r ./jobs/requirements.txt
```

4. Run local_run.sh and specify which set of graphs are you running for.

```
local_run.sh
```

## Troubleshooting

```
Error occurred when finalizing GeneratorDataset iterator: FAILED_PRECONDITION: Python interpreter state is not initialized. The process may be terminated.
```

GPU maybe too small for the GNN, set:
```
export CUDA_VISIBLE_DEVICES=""
```