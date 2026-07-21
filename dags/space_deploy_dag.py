from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="deploy_space",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:
    
    unit_test = BashOperator(
    task_id="unit_test",
    bash_command="cd /opt/airflow && PYTHONPATH=/opt/airflow/app /opt/airflow/app-test-venv/bin/pytest tests/test_unit.py",
    )


    deployment = BashOperator(
        task_id="complete_app_deployment",
        # BashOperator default to tmp working directory
        # set the working folder with cd to allow relative path to resolve correctly
        bash_command="cd /opt/airflow && python scripts/deploy_space.py",
        env={
            "HF_TOKEN": "{{ var.value.HF_TOKEN }}",
            "HF_USER": "{{ var.value.HF_USER }}",
            "SPACE_NAME": "{{ var.value.SPACE_NAME }}",
            "MODEL_REPO": "{{ var.value.MODEL_REPO }}",
            "GRAFANA_URL": "{{ var.value.GRAFANA_URL }}",
            "KAGGLE_USERNAME": "{{ var.value.KAGGLE_USERNAME }}",
            "COMPANY": "{{ var.value.COMPANY }}",
            "SYNTH_DATA_REPO": "{{ var.value.SYNTH_DATA_REPO }}",
            "POSTS_REPO": "{{ var.value.POSTS_REPO}}"
        },
    )

    unit_test >> deployment

