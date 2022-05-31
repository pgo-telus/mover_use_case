# Global import 
import os
import re
import google.oauth2.credentials
from google.cloud import bigquery
from google.cloud import storage


def connect_bq_services(project_name: str) -> bigquery.Client:
    # Get credentials
    token = os.popen('gcloud auth print-access-token').read()
    token = re.sub(f'\n$', '', token)
    credentials = google.oauth2.credentials.Credentials(token)

    # Create big query client
    bq_client = bigquery.Client(credentials=credentials, project=project_name)

    return bq_client


def connect_storage_services(project_name: str) -> storage.Client:
    # Get credentials
    token = os.popen('gcloud auth print-access-token').read()
    token = re.sub(f'\n$', '', token)

    # Create big query client
    storage_client = storage.Client(credentials=google.oauth2.credentials.Credentials(token))

    return storage_client


