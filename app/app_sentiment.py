from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse 
from pydantic import BaseModel
from contextlib import asynccontextmanager
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import sqlite3
import os
from pathlib import Path
torch.set_grad_enabled(False)



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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model

    init_db()
    tokenizer = AutoTokenizer.from_pretrained("divde/sentiment_analysis_classifier")
    model = AutoModelForSequenceClassification.from_pretrained("divde/sentiment_analysis_classifier")  
    model.eval()
    yield
    del model
    torch.cuda.empty_cache()


app = FastAPI(lifespan = lifespan)


def analyze_sentiment(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    with torch.inference_mode():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()
    label = model.config.id2label[predicted_class]
    log_prediction(text, label, confidence)
    return label,confidence

    
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
                date(timestamp) as day,
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
async def get_logs(last_n = Query(default = 200, ge=1, le=5000)):
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

io = gr.Interface(
    fn = analyze_sentiment,
    inputs=gr.Textbox(label="text"), 
    outputs=[gr.Label(label="sentiment"), gr.Number(label="confidence")])
app = gr.mount_gradio_app(app, io, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
