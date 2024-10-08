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
N = os.getenv("N", None)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))


def do_work(local_name: str,
            prefix: str, i: int, j: int) -> None:
    json_key = get_json_key()
    res_fname = "results/result_%s_%d_%d.txt" % (prefix, i, j)
    res_tmp = "/tmp/result_%s_%d_%d.txt" % (prefix, i, j)
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
    asyncio.run(write_result(prefix, i, j , json_key=json_key, data=result))
    print('Results: %s' % result)

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
    N = None
    if prefix in {'faaa', 'faab'}:
        N = 3200
    k = get_json_key()
    if N:
        fname = "graph_%s%d_%d_%d.pkl" % (prefix, i, n, j)
    else:
        fname = "graph_%s%d_%d.pkl" % (prefix, i,  j) # - new, No N

    print("Starting the task file %s - %d %d %s" % (fname, graph_set, index, datetime.now()))

    local_name = "/tmp/%s" % fname
    json_key = get_json_key()
    logging.info("Starting reading the file %s" % fname)
    res = read_data(fname, local_name, bucket_name=BUCKET_NAME, json_key=json_key)
    logging.info('Result of the read: ', res)
    if res:
        do_work(local_name, prefix, i, j)
        print('Work is done!')
        sys.exit(0)

