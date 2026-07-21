from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="deploy_grafana",
    schedule=None,
    start_date=datetime(2026,1,1),
    catchup=False,
) as dag:
    deploy_grafana = BashOperator(
        task_id = "push_dashboard_to_cloud",
        # BashOperator default to tmp working directory
        # set the working folder with cd to allow relative path to resolve correctly
        bash_command="cd /opt/airflow && python scripts/deploy_grafana.py",
        env={
            "GRAFANA_TOKEN": "{{ var.value.GRAFANA_TOKEN}}",
            "GRAFANA_INSTANCE": "{{ var.value.GRAFANA_INSTANCE}}",
            "SPACE_URL": "{{ var.value.SPACE_URL}}",
        },
    )
