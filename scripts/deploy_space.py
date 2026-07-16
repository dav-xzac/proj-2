import os
from huggingface_hub import HfApi
from datetime import datetime
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USER = os.getenv("HF_USER")
SPACE_NAME = os.getenv("SPACE_NAME")

api = HfApi(token=HF_TOKEN)
repo_id = f"{HF_USER}/{SPACE_NAME}"
# api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="space",
    folder_path="app",
    commit_message=f"deploy update {datetime.now().strftime('%Y-%m-%d %H:%M')}",
)
