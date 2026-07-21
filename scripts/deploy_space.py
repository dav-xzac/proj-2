import os
from huggingface_hub import HfApi, Volume, create_bucket
from datetime import datetime
HF_TOKEN = os.environ["HF_TOKEN"]
HF_USER = os.environ["HF_USER"]
SPACE_NAME = os.environ["SPACE_NAME"]

api = HfApi(token=HF_TOKEN)
repo_id = f"{HF_USER}/{SPACE_NAME}"
BUCKET_ID = f"{HF_USER}/{SPACE_NAME}-storage"
MOUNT_PATH = "/data"

# api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

# Persistent storage within free tier for predictions.db and mlflow.db mounted on space
create_bucket(BUCKET_ID, exist_ok=True, token=HF_TOKEN)
api.set_space_volumes(
    repo_id=repo_id,
    volumes=[Volume(type="bucket", source=BUCKET_ID, mount_path=MOUNT_PATH)],
)

# Secrets and Variables update in the space settings
SECRETS = {"HF_TOKEN": HF_TOKEN}
VARIABLES = {
    "HF_USER": HF_USER,
    "MODEL_REPO": os.environ["MODEL_REPO"],
    "GRAFANA_URL": os.environ["GRAFANA_URL"],
    "KAGGLE_USERNAME": os.environ["KAGGLE_USERNAME"],
    "COMPANY": os.environ["COMPANY"],
    "SYNTH_DATA_REPO": os.environ["SYNTH_DATA_REPO"],
    "POSTS_REPO":os.environ["POSTS_REPO"]
}


for key, value in SECRETS.items():
    print(f"Set secret {key}")
    api.add_space_secret(repo_id=repo_id, key=key, value=value)


for key, value in VARIABLES.items():
    print(f"Set variable {key}")
    api.add_space_variable(repo_id=repo_id, key=key, value=value)

#Push of the updated folder to the space and restarting
api.upload_folder(
    repo_id=repo_id,
    repo_type="space",
    folder_path="app",
    commit_message=f"deploy update {datetime.now().strftime('%Y-%m-%d %H:%M')}",
)

api.restart_space(repo_id=repo_id)
