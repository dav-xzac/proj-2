from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse 
from pydantic import BaseModel
from contextlib import asynccontextmanager
import gradio as gr
from huggingface_hub import InferenceClient
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch
import sqlite3
import os
from pathlib import Path
import tempfile
import openpyxl
# torch.set_grad_enabled(False)

HF_TOKEN = os.getenv("HF")
MODEL = os.getenv("MODEL")
client = InferenceClient(model = MODEL, token = HF_TOKEN)

DB_DIR = Path("/data" if Path("/data").exists() else "/tmp")
DB_PATH = DB_DIR / "predictions.db"

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
                     id         INTEGER PRIMARY KEY AUTOINCREMENT,
                     ts         TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                     text       TEXT    NOT NULL,
                     label      TEXT    NOT NULL,
                     confidence REAL    NOT NULL,
                     text_len   INTEGER NOT NULL
                     )
                """)
        conn.commit()

def log_prediction(text, label, confidence):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO predictions (text, label, confidence, text_len) VALUES (?,?,?,?)",
            (text, label,round(confidence, 4), len(text))
        )
        conn.commit()

def export_to_excel(from_date,to_date,label_filter):
    query = "SELECT ts, text, label, confidence FROM predictions WHERE 1=1"
    params = []
    if label_filter and label_filter != "all":
        query += " AND label = ?"
        params.append(label_filter)
    if from_date:
        query += " AND date(ts) >= ?"
        params.append(from_date)
    if to_date:
        query += " AND date(ts) <= ?"
        params.append(to_date)
    query += " ORDER BY ts ASC"

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()

    if not rows:
        return None
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sentiment Report "
    if from_date:
        ws.title += f"from {from_date}"
    if to_date:
        ws.title += f"from {to_date}"
    ws.append(["Date", "Time", "Text", "Label", "Confidence"])
    for r in rows:
        ws.append([r["ts"][:10], r["ts"][11:19], r["text"], r["label"], r["confidence"]])
        ws.column_dimensions["A"].width = 12
        ws.column_dimensions["B"].width = 10
        ws.column_dimensions["C"].width = 60
        ws.column_dimensions["D"].width = 12
        ws.column_dimensions["E"].width = 12

    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete = False)
    wb.save(tmp.name)
    return tmp.name

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global tokenizer, model

#     init_db()
#     tokenizer = AutoTokenizer.from_pretrained("divde/sentiment_analysis_classifier", token=HF_TOKEN)
#     model = AutoModelForSequenceClassification.from_pretrained("divde/sentiment_analysis_classifier", token=HF_TOKEN)
#     model.eval()
#     yield
#     del model
#     torch.cuda.empty_cache()

@asynccontextmanager
async def lifespan(app: FastAPI):

    init_db()
    yield


app = FastAPI(lifespan = lifespan)




# def analyze_sentiment(text: str) -> str:
#     inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
#     with torch.inference_mode():
#         outputs = model(**inputs)
#     probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     predicted_class = torch.argmax(probabilities, dim=1).item()
#     confidence = probabilities[0][predicted_class].item()
#     label = model.config.id2label[predicted_class]
#     log_prediction(text, label, confidence)
#     return label,confidence

# def analyze_sentiment(text: str):
#     results = client.text_classification(text)
#     top = max(results, key=lambda x: x.score)
#     log_prediction(text, top.label, top.score)
#     return top.label, top.score

import requests

def analyze_sentiment(text: str):
    response = requests.post(
        f"https://api-inference.huggingface.co/models/divde/sentiment_analysis_classifier",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": text}
    )
    results = response.json()
    top = max(results[0], key=lambda x: x["score"])
    log_prediction(text, top["label"], top["score"])
    return top["label"], top["score"]

    
class SentimentRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <div align="center">
            <title>Sentiment Analysis classifier</title>
    </head>
    <body> 
        <h1>Welcome to the Sentiment Analysis Classifier API</h1>
        <p>Use the /predict endpoint to analyze sentiment.</p>
        <a href="/gradio/">Procedi all' applicazione gradio</a>
    </body>
    </html>
    """
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
    conn.close()

    result = {}
    for day, label, count, in rows:
        if day not in result:
            result[day] = {"date":day, "positive": 0, "neutral":0, "negative":0}
        result[day][label] = count

    return list(result.values())

@app.get("/logs")
async def get_logs(last_n:int = Query(default = 200, ge=1, le=5000)):
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

with gr.Blocks(title="Sentiment Analysis") as io:
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


# io = gr.Interface(
#     fn = analyze_sentiment,
#     inputs=gr.Textbox(label="text"), 
#     outputs=[gr.Label(label="sentiment"), gr.Number(label="confidence")])
app = gr.mount_gradio_app(app, io, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
