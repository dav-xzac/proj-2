from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime
import requests, random, time

APP_URL = "https://divde-sentiment-proj-5.hf.space/predict"

def _send_posts():
    from huggingface_hub import InferenceClient

    client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        token=Variable.get("HF_TOKEN")
    )
    sentiments = ["positive", "negative", "neutral"]
    n_posts = random.randint(10, 50)

    for i in range(n_posts):
        sentiment = random.choice(sentiments)
        text = client.text_generation(
            f"Write one short {sentiment} social media post about Machine Innovators ltd, "
            f"a leader in scalable ML applications. Output only the post text.",
            max_new_tokens=60,
        ).strip()
        requests.post(APP_URL, json={"text": text}, timeout=15)
        print(f"[{i+1}/{n_posts}] {sentiment}: {text[:60]}...")
        if i % 10 == 9:
            time.sleep(2)

with DAG(
    dag_id="traffic_simulator",
    schedule="0 10 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1},
) as dag:

    PythonOperator(
        task_id="send_posts",
        python_callable=_send_posts,
    )
