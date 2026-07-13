from fastapi import FastAPI, Query, Request, Response
from pydantic import BaseModel
from contextlib import asynccontextmanager
import gradio as gr
from huggingface_hub import InferenceClient, repo_exists
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import HfApi
import torch
import os
import subprocess
import httpx
from pathlib import Path
import time
from db_setup import init_db,get_conn,log_prediction,export_to_excel
torch.set_grad_enabled(False)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USER = os.getenv("HF_USER")
MODEL_REPO = os.getenv("MODEL_REPO")
GRAFANA_URL = os.getenv("GRAFANA_URL")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_NOTEBOOK_URL = f"https://www.kaggle.com/code/{KAGGLE_USERNAME}/sentiment-retraining" if KAGGLE_USERNAME else None
GITHUB_REPO_URL = os.getenv("GITHUB_REPO_URL", "https://github.com/dav-xzac/proj-2")
SYNTH_DATA_REPO = os.getenv("SYNTH_DATA_REPO", "")
SYNTH_DATA_URL = f"https://huggingface.co/datasets/{HF_USER}/{SYNTH_DATA_REPO}" if SYNTH_DATA_REPO else None
ASPECT = os.getenv("COMPANY", "anthropic")
MLFLOW_INTERNAL = "http://127.0.0.1:5000"
MLFLOW_DIR = Path("/data" if Path("/data").exists() else "/tmp")


if HF_TOKEN == None:
    print("Warning: HF_TOKEN secret is not defined")
if HF_USER == None:
    print("Warning: HF_USER var is not defined")

MODEL_EXISTS = HF_USER and MODEL_REPO and repo_exists(HF_USER + "/" + MODEL_REPO)

if not MODEL_EXISTS:
    print("Warning: model doesn't exist yet on huggingface")
    MODEL_REPO = "twitter-roberta-base-sentiment-latest"
    HF_USER = "cardiffnlp"

MODEL_PATH = f"{HF_USER}/{MODEL_REPO}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model

    init_db()
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}", token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(f"{MODEL_PATH}", token=HF_TOKEN)
    model.eval()
    
    mlflow_server = subprocess.Popen([
        "mlflow", "server",
        "--backend-store-uri", f"sqlite:////{MLFLOW_DIR}/mlflow.db",
        "--host", "127.0.0.1",
        "--port", "5000",
        "--static-prefix", "/mlflow",
    ])

    mlflow_ready = False
    for _ in range(60):
        try:
            if httpx.get("http://127.0.0.1:5000/mlflow/", timeout=2).status_code == 200:
                mlflow_ready = True
                break
        except httpx.RequestError:
            pass
        time.sleep(3)
    
    if not mlflow_ready:
        print("WARNING: Mlflow not yet ready")
   
    yield

    mlflow_server.terminate()
    del model
    torch.cuda.empty_cache()



app = FastAPI(lifespan = lifespan)




def analyze_sentiment(text: str) -> str:
    inputs = tokenizer(text,ASPECT, return_tensors="pt", max_length=512, truncation=True, padding=True)
    with torch.inference_mode():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()
    label = model.config.id2label[predicted_class]
    log_prediction(text, label, confidence)
    return label,confidence


def get_latest_model_version(model_path):
    try:
        refs = HfApi().list_repo_refs(model_path)
        tags = sorted(t.name for t in refs.tags)
        return tags[-1] if tags else "no_version"
    except Exception:
        return "no_version"
    
MODEL_VERSION = get_latest_model_version(MODEL_PATH)
 
class SentimentRequest(BaseModel):
    text: str


@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict_sentiment(request: SentimentRequest):
    text = request.text
    sentiment, confidence = analyze_sentiment(text)
    return {"text": text, "sentiment": sentiment, "confidence": confidence}

@app.get("/daily_stats")
def daily_stats():
    conn = get_conn()
    rows = conn.execute("""
            SELECT
                date(ts) as day,
                        label,
                        COUNT(*) as count
            FROM predictions
            GROUP BY day, label
            ORDER BY day DESC
            LIMIT 90
            """).fetchall()

    conf_rows = conn.execute("""
            SELECT
                date(ts) as day,
                AVG(confidence) as avg_confidence
            FROM predictions
            GROUP BY day
            ORDER BY day DESC
            LIMIT 90
            """).fetchall()
    conn.close()

    result = {}
    for day, label, count, in rows:
        if day not in result:
            result[day] = {"date":day, "positive": 0, "neutral":0, "negative":0}
        result[day][label] = count

    for day, avg_conf in conf_rows:
        if day in result:
            result[day]["avg_confidence"] = round(avg_conf, 4)


    return list(result.values())

@app.get("/weekly_stats")
def weekly_stats():
    with get_conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                AVG(confidence) as avg_confidence
            FROM predictions
            WHERE ts >= datetime('now', '-7 days')
        """).fetchone()

    return {
        "total_predictions_7d": row["total"],
        "avg_confidence_7d": round(row["avg_confidence"] or 0, 4),
    }


@app.get("/logs")
async def get_logs(last_n:int = Query(default = 2000, ge=1, le=5000)):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT ts, label, confidence, text_len FROM predictions ORDER BY id DESC LIMIT ?",
            (last_n,)
        ).fetchall()
        counts = conn.execute(
            "SELECT label, COUNT(*) as count FROM predictions GROUP BY label"
        ).fetchall()
        avg_conf = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0]
        avg_len = conn.execute("SELECT AVG(text_len)    FROM predictions").fetchone()[0]
        total = conn.execute("SELECT COUNT(*)           FROM predictions").fetchone()[0]

    label_counts = {r["label"]: r["count"] for r in counts}
    label_ratios = {k: round(v / total, 4) for k, v in label_counts.items()} if total else {}

    return {
        "total_predictions":    total,
        "label_counts":         label_counts,
        "label_ratios":         label_ratios,
        "avg_confidence":       round(avg_conf or 0, 4),
        "avg_text_length":      round(avg_len or 0, 1),
        "recent_predictions":   [dict(r) for r in rows]
    }

@app.api_route("/mlflow/{path:path}", methods = ["GET","POST"])
async def mlflow_route(path: str, request: Request):
    url = f"{MLFLOW_INTERNAL}/mlflow/{path}"
    async with httpx.AsyncClient() as client:
        routed = await client.request(
            request.method, url,
            params=request.query_params,
            headers={k: v for k,v in request.headers.items() if k.lower() != "host"},
            content=await request.body(),
        )
    return Response(content=routed.content, status_code= routed.status_code, headers =dict(routed.headers))


with gr.Blocks(title="Sentiment Analysis") as io:
    gr.Textbox(value=f"{MODEL_PATH}({MODEL_VERSION})", label="Serving model", interactive=False)
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                f"**Public Links**\n\n[MLflow](/mlflow/) &nbsp;·&nbsp; [GitHub Repo]({GITHUB_REPO_URL}) &nbsp;·&nbsp; [Synthetic Data]({SYNTH_DATA_URL})"
            )
        with gr.Column():
            gr.Markdown(
                f"**Admin Links**\n\n[GRAFANA]({GRAFANA_URL}) &nbsp;·&nbsp; [Training Notebook]({KAGGLE_NOTEBOOK_URL})"
                )
    with gr.Tab("Analyze"):
        text_input = gr.Textbox(label="Text")
        with gr.Row():
            sentiment_output = gr.Label(label = "Sentiment")
            confidence_output = gr.Number(label = "Confidence")
        gr.Button("Submit").click(
            analyze_sentiment,
            inputs = text_input,
            outputs = [sentiment_output, confidence_output]
        )
    
    with gr.Tab("Export"):
        with gr.Row():
            from_input = gr.Textbox(label="From (YYYY-MM-DD)", placeholder = "2026-06-01")
            to_input = gr.Textbox(label="To (YYYY-MM-DD)", placeholder = "2026-06-20")
        label_input = gr.Dropdown(
            choices = ["all", "positive", "neutral", "negative"],
            value = "all",
            label = "Sentiment filter"
        )
        file_output = gr.Files(label="Download Excel")
        gr.Button("Export").click(
            export_to_excel,
            inputs=[from_input,to_input,label_input],
            outputs = file_output
        )


app = gr.mount_gradio_app(app, io, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
