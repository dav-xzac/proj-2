from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.sensors.python import PythonSensor
from airflow.models import Variable
from datetime import datetime, timedelta
import requests, json, os

LOGS_URL            = "https://divde-sentiment-proj-5.hf.space/logs"
METRICS_REPO        = "divde/sentiment-training-metrics"
KERNEL_SLUG         = "dav-xzac/sentiment-retraining"
CONFIDENCE_THRESHOLD = 0.7
NEUTRAL_THRESHOLD   = 0.5
MIN_PREDICTIONS     = 50

def _check_drift():
    if Variable.get("retraining_improved", default_var="unknown") == "false":
        print("Last retraining did not improve — manual reset required.")
        return False
    if Variable.get("retraining_in_progress", default_var="false") == "true":
        print("Retraining already in progress.")
        return False

    logs = requests.get(LOGS_URL, timeout=10).json()
    total = logs.get("total_predictions", 0)
    if total < MIN_PREDICTIONS:
        print(f"Not enough predictions ({total}/{MIN_PREDICTIONS}).")
        return False

    confidence = logs.get("avg_confidence", 1.0)
    neutral    = logs.get("label_ratios", {}).get("neutral", 0)
    drift      = confidence < CONFIDENCE_THRESHOLD or neutral > NEUTRAL_THRESHOLD
    print(f"Drift={'YES' if drift else 'NO'} | confidence={confidence:.3f} | neutral={neutral:.3f}")
    return drift

def _trigger_kaggle():
    import subprocess
    os.environ["KAGGLE_TOKEN"] = Variable.get("KAGGLE_TOKEN")

    Variable.set("retraining_in_progress", "true")
    subprocess.run(
        ["kaggle", "kernels", "push", "-p", "/opt/airflow/model/"],
        capture_output=True, text=True, check=True
    )

def _check_kaggle_status():
    import subprocess
    os.environ["KAGGLE_TOKEN"] = Variable.get("KAGGLE_TOKEN")
 
 
    result = subprocess.run(
        ["kaggle", "kernels", "status", KERNEL_SLUG],
        capture_output=True, text=True
    )
    output = result.stdout.lower()
    print(result.stdout)
    if "complete" in output:
        return True
    if "error" in output or "failed" in output:
        raise Exception(f"Kaggle kernel failed: {result.stdout}")
    return False

def _evaluate_improvement():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=METRICS_REPO, filename="metrics.json",
        repo_type="dataset", token=Variable.get("HF_TOKEN")
    )
    metrics  = json.load(open(path))
    improved = metrics.get("improved", False)
    Variable.set("retraining_improved",    str(improved).lower())
    Variable.set("retraining_in_progress", "false")
    print(f"{'IMPROVED' if improved else 'NO IMPROVEMENT'}: "
          f"{metrics.get('baseline_f1')} → {metrics.get('new_f1')}")

with DAG(
    dag_id="retraining_pipeline",
    schedule="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
) as dag:

    check_drift = ShortCircuitOperator(
        task_id="check_drift",
        python_callable=_check_drift,
    )
    trigger_kaggle = PythonOperator(
        task_id="trigger_kaggle",
        python_callable=_trigger_kaggle,
    )
    wait_kaggle = PythonSensor(
        task_id="wait_kaggle_completion",
        python_callable=_check_kaggle_status,
        poke_interval=300,
        timeout=7200,
        mode="reschedule",
    )
    evaluate = PythonOperator(
        task_id="evaluate_improvement",
        python_callable=_evaluate_improvement,
    )

    check_drift >> trigger_kaggle >> wait_kaggle >> evaluate
