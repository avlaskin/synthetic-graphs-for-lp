from datetime import datetime
import json

from google.cloud import firestore
from google.oauth2 import service_account   

_QUERY_COLLECTION = 'results'
JSON_SECRET_KEY = 'DB_CREDS'
_DB_NAME = 'graphs'


async def init_firebase(json_secret: str):
    """Inits firebase async client."""
    creds = None
    json_data = None
    if json_secret:
        json_data = json.loads(json_secret)
        creds = service_account.Credentials.from_service_account_info(json_data)
        print('Found creds for project: ', json_data['project_id'])
    if creds:
        print('Getting DB client with Credentials.') 
        db = firestore.AsyncClient(
            database=_DB_NAME,
            project=json_data['project_id'],
            credentials=creds,
        )
    else:
        print('Attmpt to retrieve clint with no credentials.')
        db = firestore.AsyncClient()
    return db


async def post_data(doc_id: str,
                    data: str,
                    collection: str,
                    db: firestore.AsyncClient):
    """Posts survey data by appending it to the list."""
    doc_ref = db.collection(collection).document(doc_id)
    print('Add new record: %s' % doc_id)
    now = datetime.now()
    ts = datetime.timestamp(now)
    res = await doc_ref.set({
        'doc_id': doc_id,
        'auc': data,
        'time': ts
    })
    return res
