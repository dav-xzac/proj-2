from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="synth_data_generation",
    schedule='22 7 * * 0',
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:
    update_secrets = BashOperator(
        task_id="update_kaggle_secrets",
        bash_command="export PATH=$PATH:/home/airflow/.local/bin && cd /opt/airflow && python scripts/update_kaggle_secrets.py",
        env={
            "KAGGLE_USERNAME": "{{ var.value.KAGGLE_USERNAME}}",
            "KAGGLE_API_TOKEN": "{{ var.value.KAGGLE_TOKEN}}",
            "HF_TOKEN": "{{ var.value.HF_TOKEN}}",
            "HF_USER": "{{ var.value.HF_USER}}",
            "SPACE_NAME": "{{ var.value.SPACE_NAME}}",
            "MODEL_REPO": "{{ var.value.MODEL_REPO}}",
            "METRICS_REPO": "{{ var.value.METRICS_REPO}}",
            "SYNTH_DATA_REPO": "{{ var.value.SYNTH_DATA_REPO}}",
            "SPACE_URL": "{{ var.value.SPACE_URL}}",
            "COMPANY": "{{ var.value.COMPANY}}",
            "COMPANY_DESC": "{{ var.value.COMPANY_DESC}}",
        }
    )

    push_notebook = BashOperator(
        task_id="push_training_notebook",
        bash_command="export PATH=$PATH:/home/airflow/.local/bin && cd /opt/airflow && python scripts/push_kaggle_kernel.py synth_generation synth-generation",
        env={
            "KAGGLE_USERNAME": "{{ var.value.KAGGLE_USERNAME}}",
            "KAGGLE_API_TOKEN": "{{ var.value.KAGGLE_TOKEN}}",
        }
    )

    update_secrets >> push_notebook
