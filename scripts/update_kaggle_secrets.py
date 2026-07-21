import json
import os
import subprocess

KAGGLE_USERNAME = os.environ["KAGGLE_USERNAME"]

os.makedirs("/tmp/kaggle-secrets", exist_ok=True)

# Load template and write secrets and variables to new file to be pushed as kaggle input for notebooks
with open("secrets/dataset-metadata_template.json") as f:
    dataset_metadata = json.load(f)
dataset_metadata["id"] = f"{KAGGLE_USERNAME}/secrets"
with open("/tmp/kaggle-secrets/dataset-metadata.json", "w") as f:
    json.dump(dataset_metadata, f)

with open("secrets/secrets_template.json") as f:
    secrets = json.load(f)
secrets.update({
    "HF_TOKEN": os.environ["HF_TOKEN"],
    "HF_USER": os.environ["HF_USER"],
    "SPACE_NAME": os.environ["SPACE_NAME"],
    "MODEL_REPO": os.environ["MODEL_REPO"],
    "METRICS_REPO": os.environ["METRICS_REPO"],
    "SYNTH_DATA_REPO": os.environ["SYNTH_DATA_REPO"],
    "SPACE_URL": os.environ["SPACE_URL"],
    "COMPANY": os.environ["COMPANY"],
    "COMPANY_DESC": os.environ["COMPANY_DESC"],
})
with open("/tmp/kaggle-secrets/secrets.json", "w") as f:
    json.dump(secrets, f)

# check dataset existance to update or create
exists = subprocess.run(
    ["kaggle", "datasets", "metadata", "-d", f"{KAGGLE_USERNAME}/secrets"],
    capture_output=True,
).returncode == 0

if exists:
    subprocess.run(["kaggle", "datasets", "version", "-p", "/tmp/kaggle-secrets", "-m", "update"], check=True)
else:
    subprocess.run(["kaggle", "datasets", "create", "-p", "/tmp/kaggle-secrets"], check=True)
