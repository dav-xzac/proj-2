from fastapi import FastAPI
from fastapi.responses import HTMLResponse 
from pydantic import BaseModel
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pathlib import Path
torch.set_grad_enabled(False)


def analyze_sentiment(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    with torch.inference_mode():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()
    return model.config.id2label[predicted_class],confidence


app = FastAPI()

@app.on_event("startup")
def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained("divde/sentiment_analysis_classifier")
    model = AutoModelForSequenceClassification.from_pretrained("divde/sentiment_analysis_classifier")  
    model.eval()

    
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
@app.post("/predict")
async def predict_sentiment(request: SentimentRequest):
    text = request.text
    sentiment, confidence = analyze_sentiment(text)
    return {"text": text, "sentiment": sentiment, "confidence": confidence}

io = gr.Interface(fn = analyze_sentiment,inputs=gr.Textbox(label="text"), outputs=[gr.Label(label="sentiment"), gr.Number(label="confidence")])
app = gr.mount_gradio_app(app, io, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)