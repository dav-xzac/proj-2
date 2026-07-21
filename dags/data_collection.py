from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="data_collection",
    schedule="12 7 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:
    fetch = BashOperator(
        task_id="fetch_data_from_api",
        bash_command="python3 /opt/airflow/data_collection/mastodon.py",
        env={
            "HF_TOKEN": "{{ var.value.HF_TOKEN }}",
            "SPACE_URL": "{{ var.value.SPACE_URL }}",
            "HF_USER": "{{ var.value.HF_USER }}",
            "COMPANY": "{{ var.value.COMPANY }}",
        },
    )
