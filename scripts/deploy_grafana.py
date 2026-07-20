import json
import os
import requests

GRAFANA_TOKEN = os.environ["GRAFANA_TOKEN"]
GRAFANA_INSTANCE = os.environ["GRAFANA_INSTANCE"]
SPACE_URL = os.environ["SPACE_URL"]
BASE_URL = f"https://{GRAFANA_INSTANCE}.grafana.net"
HEADERS = {"Authorization": f"Bearer {GRAFANA_TOKEN}"}

check = requests.get(f"{BASE_URL}/api/datasources/uid/infinity-datasource", headers=HEADERS)
if check.status_code == 200:
    print("Datasource already exists, skipping")
else:
    requests.post(
        f"{BASE_URL}/api/datasources",
        headers=HEADERS,
        json={"name": "Infinity", "type": "yesoreyeram-infinity-datasource", "uid": "infinity-datasource", "access": "proxy", "isDefault": True},
    )

with open("monitoring/grafana/dashboards/sentiment.json") as f:
    dashboard = json.load(f)

def replace_placeholder(obj):
    if isinstance(obj, dict):
        return {k: replace_placeholder(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_placeholder(v) for v in obj]
    if isinstance(obj, str):
        return obj.replace("SPACE_URL", SPACE_URL)
    return obj

dashboard = replace_placeholder(dashboard)
dashboard["id"] = None

payload = {"dashboard": dashboard, "overwrite": True, "folderId": 0}
requests.post(f"{BASE_URL}/api/dashboards/db", headers=HEADERS, json=payload)
