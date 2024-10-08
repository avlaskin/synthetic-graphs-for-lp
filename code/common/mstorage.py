import logging
import time
import os
import json


from google.cloud import storage
from google.oauth2 import service_account   
from google.cloud.storage.blob import Blob


log = logging.getLogger("storage")


def read_bucket_to_file(key_str: str,
                        bucket_name: str,
                        remote_file_name: str,
                        local_file_name: str) -> str:
    ''' Reads remote file and saves it locally. '''
    json_data = json.loads(key_str)
    creds = service_account.Credentials.from_service_account_info(json_data)
    storage_client = storage.Client(credentials=creds)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.get_blob(remote_file_name)
    fname = local_file_name
    try:
        with open(fname, 'wb') as stream:
            blob.download_to_file(stream)
            stream.close()
    except Exception as e:
        #log.error("Error reading storage: ", e)
        return None
    return fname


def read_data(fname: str,
              local_name: str,
              bucket_name: str,
              json_key: str) -> str:
    """Reads data from the bucket."""
    full_name = "data/" + fname
    print('Reading %s from %s' % (full_name, bucket_name))
    name = read_bucket_to_file(json_key,
                               bucket_name,
                               full_name,
                               local_name)
    if name:
        print('File is read all good')
        return local_name
    print('File was not found.')
    return None


def write_file_to_bucket(key_str: str,
                         bucket_name: str,
                         image_as_bytes: bytes,
                         content_type: str,
                         remote_file_name: str) -> bool:
    if len(image_as_bytes) < 1:
        log.warning("Error: string image is  not found.")
        return False
    json_data = json.loads(key_str)
    creds = service_account.Credentials.from_service_account_info(json_data)
    storage_client = storage.Client(credentials=creds)
    #storage_client = storage.Client.from_service_account_json(key_path)
    bucket = storage_client.get_bucket(bucket_name)
    # Create a remote blob first
    try:
        blob = bucket.blob(remote_file_name)
        blob.upload_from_string(image_as_bytes, content_type=content_type)
        log.info("Writing remote file: %s" % remote_file_name)
    except Exception as e:
        log.warning("Error writing to storage:", e)
        return False
    return True
