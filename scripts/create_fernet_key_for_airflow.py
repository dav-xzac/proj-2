# pip install cryptography required
# Use this script for generating a key 
# to encrypt secrets values for grafana, kaggle and hf 
# also while deploying Airflow locally, to avoid any token exposed on DB
# Copy the value in an env key --> FERNET_KEY=<printed_key>

from cryptography.fernet import Fernet

print(Fernet.generate_key().decode())
