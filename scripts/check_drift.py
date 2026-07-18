import os
import sys
import requests

SPACE_URL = os.environ["SPACE_URL"]

stats = requests.get(f"{SPACE_URL}/weekly_stats").json()
total = stats["total_predictions_7d"]
avg = stats["avg_confidence_7d"]
print(f"Week Total Predictions: {total} | Avg confidence: {avg}", file=sys.stderr)

if total < 100:
    print("Insufficient data collected, skip retraining", file=sys.stderr)
    print("false")
elif avg < 0.80:
    print("true")
else:
    print("false")
