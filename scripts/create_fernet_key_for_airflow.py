# pip install cryptography required
# Use this script to generate a key 
# to encrypt secrets values for grafana, kaggle and hf 
# also while deploying Airflow locally, to avoid any token exposed on DB
# Copy the value in an env key --> AIRFLOW_FERNET_KEY=<printed_key>

from cryptography.fernet import Fernet

print(Fernet.generate_key().decode())
