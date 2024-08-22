"""Here we define all secrets."""

import os
from google.cloud import secretmanager


PROJECT_ID = os.environ['PROJECT_ID']
JSON_SECRET_KEY = 'DB_CREDS'


def access_secret_version(secret_id, version_id='latest'):
    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret version.
    name = f'projects/{PROJECT_ID}/secrets/{secret_id}/versions/{version_id}'

    # Access the secret version.
    response = client.access_secret_version(name=name)

    # Return the decoded payload. Don't print this.
    return response.payload.data.decode('UTF-8')


def get_json_key():
    if not PROJECT_ID:
        print('Project id is not set! Can not save the resutl')
        return None
    json_key = access_secret_version(JSON_SECRET_KEY)
    return json_key



