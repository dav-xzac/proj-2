from airflow.models.dag import DAG
from airflow.providers.http.operators.http import HttpOperator
import json

from datetime import datetime

with DAG(
    dag_id='app_testing',
    catchup=False # Prevents backfilling missed runs
) as dag:
    def trigger_app_testing():
        app_run_task = HttpOperator(
            task_id='trigger_app_testing',
            http_conn_id='GITHUB_API',  
            method='POST',
            endpoint='repos/dav-xzac/proj-2/actions/workflows/test.yaml/dispatches',  
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
        return app_run_task
    
    app_run = trigger_app_testing()

    
    app_run
    
