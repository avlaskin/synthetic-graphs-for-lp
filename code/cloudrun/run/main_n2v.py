'''This app reads and writes to the cloud bucket.'''
import asyncio
import numpy as np
import logging
import os
import pickle
import sys

from datetime import datetime
from module import experiments
from module.mstorage import read_bucket_to_file, write_file_to_bucket
from module.experiments import GraphParams, get_graph_data
sys.modules['experiments'] = experiments
from module.mynode2vec import n2v_work
from module.msecrets import access_secret_version, get_json_key
from module.mdb import init_firebase, post_data
from module.mwriter import write_result, detect_local_run, read_local_data


BUCKET_NAME = "graphs-data"
URL = "https://storage.googleapis.com/"
TASK_INDEX = os.getenv("CLOUD_RUN_TASK_INDEX", 0)
GRAPH_SIMBOL = os.getenv("GRAPH_INDEX", 0)
ATTEMPT_INDEX = os.getenv("ATTEMPT_INDEX", 0)
TASK_PREFIX = os.getenv("TASK_PREFIX", "")
PROJECT_ID = os.getenv("PROJECT_ID", "")
LOCAL_DATA = os.getenv("LOCAL_DATA", "../../../../data/")

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))


def read_data(fname: str, local_name: str) -> str:
    json_key = get_json_key()
    name = read_bucket_to_file(json_key,
                               BUCKET_NAME,
                               "data/" + fname,
                               local_name)
    if name:
        ### Data has been read
        logging.info("File is read all good")
        return local_name
    return None


def do_work(local_name: str,
            prefix: str,
            i: int,
            j: int, local_run: bool = False) -> None:
    """Does the core work for method evaluation on the graph."""
    local_run = detect_local_run(PROJECT_ID)
    json_key = None
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
    s,t, m = n2v_work(data, params={})
    string = "%f,%f" % (s, t)
    asyncio.run(write_result(prefix, i, j, data=string, method='n2v', json_key=json_key, local_run=local_run))


if __name__ == "__main__":
    task_prefix = str(TASK_PREFIX)
    if len(task_prefix) == 0:
        prefix = "faab"
    else:
        prefix = task_prefix
    local_run = detect_local_run(PROJECT_ID)
    index = int(TASK_INDEX) + int(ATTEMPT_INDEX)
    graph_set = int(GRAPH_SIMBOL)
    i = graph_set
    j = index % 10
    N = 3200
    k = get_json_key()
    if N:
        fname = "graph_%s%d_%d_%d.pkl" % (prefix, i, N, j)
    else:
        fname = "graph_%s%d_%d.pkl" % (prefix, i,  j) # - new, No N

    print("Starting the task file %s - %d %d %s" % (fname, graph_set, index, datetime.now()))
    local_name = "/tmp/%s" % fname
    logging.info("Starting reading the file %s" % fname)
    res = True # for local run
    if not local_run:
        res = read_data(fname, local_name)
        print('Result of the remote read: ', res)
    else:
        res = read_local_data(fname, local_name, LOCAL_DATA)
        print('Result of the local read: ', res)
    if res:
        do_work(local_name, prefix, i, j, local_run)
        print('Work is done!')
        sys.exit(0)

