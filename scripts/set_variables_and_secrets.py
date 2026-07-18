import os
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USER = os.environ["HF_USER"]
SPACE_NAME = os.environ["SPACE_NAME"]

api = HfApi(token=HF_TOKEN)
repo_id = f"{HF_USER}/{SPACE_NAME}"

# api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

SECRETS = {"HF_TOKEN": HF_TOKEN}
VARIABLES = {
    "HF_USER": HF_USER,
    "MODEL_REPO": os.environ["MODEL_REPO"],
    "GRAFANA_URL": os.environ["GRAFANA_URL"],
    "KAGGLE_USERNAME": os.environ["KAGGLE_USERNAME"],
    "COMPANY": os.environ["COMPANY"],
    "SYNTH_DATA_REPO": os.environ["SYNTH_DATA_REPO"],
}

existing_secrets = api.get_space_secrets(repo_id=repo_id)
for key, value in SECRETS.items():
    if key not in existing_secrets:
        print(f"Creating secret {key}")
        api.add_space_secret(repo_id=repo_id, key=key, value=value)
    else:
        print(f"Secret {key} already exists")

existing_vars = api.get_space_variables(repo_id=repo_id)
for key, value in VARIABLES.items():
    if key not in existing_vars:
        print(f"Imposto variabile {key}")
        api.add_space_variable(repo_id=repo_id, key=key, value=value)
    else:
        print(f"Variable {key} already exists")
