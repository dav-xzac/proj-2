from airflow.models.dag import DAG
from airflow.providers.http.operators.http import HttpOperator
import json

from datetime import datetime

with DAG(
    dag_id='model_training',
    catchup=False # Prevents backfilling missed runs
) as dag:
    def trigger_model_training():
        training_task = HttpOperator(
            task_id='trigger_model_training',
            http_conn_id='GITHUB_API',  
            method='POST',
            endpoint='repos/dav-xzac/proj-2/actions/workflows/train.yaml/dispatches',  
            headers={
            "Authorization": "Bearer {{ conn.GITHUB_API.password }}",
            "Accept": "application/vnd.github+json",
            },
            
            data=json.dumps({
            "ref": "develop",
            "inputs": {}
            }),  
            response_check=lambda response: response.status_code == 204
            )
        return training_task
    
    training = trigger_model_training()

    
    training
    
