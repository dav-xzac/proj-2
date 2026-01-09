from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.providers.http.operators.http import HttpOperator


from datetime import datetime

with DAG(
    dag_id='model_training',
    start_date=datetime(2024, 1, 1),
    catchup=False # Prevents backfilling missed runs
) as dag:
    def trigger_model_training():
        training_task = HttpOperator(
            task_id='trigger_model_training',
            http_conn_id='GITHUB_API',  
            method='POST',
            endpoint='.github/workflows/train.yaml/dispatches',  
            headers={
            "Authorization": "Bearer {{ conn.GITHUB_API.password }}",
            "Accept": "application/vnd.github+json",
            },
            data='{}'  # Payload if needed
        )
        return training_task
    
    training = trigger_model_training()

    
    training
    
