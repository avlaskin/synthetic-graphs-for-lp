import os

from module.mdb import init_firebase, post_data
from module.msecrets import access_secret_version, get_json_key
from module.mstorage import read_bucket_to_file, write_file_to_bucket


def read_local_data(fname: str, local_name: str, folder: str) -> str:
    filename = os.path.join(folder, fname)
    with open(filename, 'rb') as fr:
        with open(local_name, 'wb') as fw:
            fw.write(fr.read())
    return local_name


def detect_local_run(project_id: str):
    if not project_id:
        print("Local run detected.")
        return True
    print("Remote run detected.")
    return False


async def write_result(prefix:str,
                       i: int,
                       j: int,
                       data: str,
                       method: str,
                       json_key: str,
                       local_run: bool = False,
                       write_to_bucket=False):
    """Writes results for the graph prefix, i, j."""
    if local_run:
        path = "/tmp/results"
        if not os.path.exists(path):
            os.mkdir(path)
        fname = os.path.join(path, "result_%s_%d_%d.txt" % (prefix, i, j))
        with open(fname, "w") as fw:
            fw.write(data)
        print("Result ", data, " written to ", fname)
        return

    if not json_key:
        logging.error('Project id is not set! Can not save the result')
        return
    if write_to_bucket:
        fname = "results/result_%s_%d_%d.txt" % (prefix, i, j)
        _ = write_file_to_bucket(json_key,
                                 BUCKET_NAME,
                                 data.encode(),
                                 content_type="text",
                                 remote_file_name=fname)
    else:
        fdb = await init_firebase(json_key)
        doc_id = '%s_%s_%d_%d' % (method, prefix, i, j)
        print('Writing result: ', doc_id)
        _ = await post_data(
            doc_id,
            data=data,
            collection='graph-results',
            db=fdb
        )
