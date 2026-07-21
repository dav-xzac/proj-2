import json
import os
import subprocess
import sys

# python scripts/push_kaggle_kernel.py(argv[0]) folder=model(argv[1]) kernel_slug=sentiment-retraining(argv[2])
folder, kernel_slug = sys.argv[1], sys.argv[2]
KAGGLE_USERNAME = os.environ["KAGGLE_USERNAME"]

# Replace placeholder in kernel metadata and write to the file to be pushed to kaggle
with open(f"{folder}/kernel-metadata_template.json") as f:
    metadata = json.load(f)
metadata["id"] = f"{KAGGLE_USERNAME}/{kernel_slug}"
metadata["dataset_sources"] = [f"{KAGGLE_USERNAME}/secrets"]
with open(f"{folder}/kernel-metadata.json", "w") as f:
    json.dump(metadata, f)

subprocess.run(["kaggle", "kernels", "push", "-p", folder, "--accelerator", "NvidiaTeslaT4"], check=True)
