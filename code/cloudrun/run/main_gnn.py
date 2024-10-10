'''This app reads and writes to the cloud bucket.'''
import asyncio
import numpy as np
import logging
import os
import pickle
import sys


from datetime import datetime
from module import experiments
from module.mstorage import read_bucket_to_file, write_file_to_bucket, read_data
from module.experiments import GraphParams
sys.modules['experiments'] = experiments
from module.msecrets import access_secret_version, get_json_key
from module.stellar import *
from module.mwriter import *

BUCKET_NAME = "graphs-data"
URL = "https://storage.googleapis.com/"
TASK_INDEX = os.getenv("CLOUD_RUN_TASK_INDEX", 0)
GRAPH_SIMBOL = os.getenv("GRAPH_INDEX", 0)
ATTEMPT_INDEX = os.getenv("ATTEMPT_INDEX", 0)
TASK_PREFIX = os.getenv("TASK_PREFIX", "")
PROJECT_ID = os.getenv("PROJECT_ID", "")
EPOCHS = os.getenv("EPOCHS", 16)
LOCAL_DATA = os.getenv("LOCAL_DATA", "../../../../data/")

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))


def do_work(local_name: str,
            prefix: str, i: int, j: int) -> None:
    """Does the work for GNN method prediction eval."""
    local_run = detect_local_run(PROJECT_ID)
    res_fname = "results/result_%s_%d_%d.txt" % (prefix, i, j)
    res_tmp = "/tmp/result_%s_%d_%d.txt" % (prefix, i, j)
    if not local_run:
        json_key = get_json_key()
        calculation_done = read_bucket_to_file(json_key,
                                               BUCKET_NAME,
                                               res_fname,
                                               res_tmp)
        if calculation_done:
            print("This calculation has been done. ", datetime.now())
            return
    
    data = get_graph_data(local_name)   
    s, t = train_graphsage_with_data(
        data,
        epochs=int(EPOCHS),
        batch=128,
        num_samples=[10, 5],
        workers=8,
        walks=20)
    result = "%f,%f" % (s, t)
    print('Results: %s' % result)
    asyncio.run(write_result(prefix, i, j , json_key=json_key, data=result, method='gnn', local_run=local_run))

    
if __name__ == "__main__":
    task_prefix = str(TASK_PREFIX)
    if len(task_prefix) == 0:
        prefix = "faab"
    else:
        prefix = task_prefix
    index = int(TASK_INDEX) + int(ATTEMPT_INDEX)
    graph_set = int(GRAPH_SIMBOL)
    i = graph_set
    j = index % 10
    n = None
    if prefix in {'faaa', 'faab'}:
        n = 3200
    local_run = detect_local_run(PROJECT_ID)
    if n:
        fname = "graph_%s%d_%d_%d.pkl" % (prefix, i, n, j)
    else:
        fname = "graph_%s%d_%d.pkl" % (prefix, i,  j) # - new, No N

    print("Starting the task file %s - %d %d %s" % (fname, graph_set, index, datetime.now()))
    local_name = "/tmp/%s" % fname
    
    logging.info("Starting reading the file %s" % fname)
    if not local_run:
        json_key = get_json_key()
        res = read_data(fname, local_name, bucket_name=BUCKET_NAME, json_key=json_key)
        print('Result of the remote read: ', res)
    else:
        res = read_local_data(fname, local_name, LOCAL_DATA)
        print('Result of the local read: ', res)
    logging.info('Result of the read: ', res)
    if res:
        do_work(local_name, prefix, i, j)
        print('Work is done!')
        sys.exit(0)

