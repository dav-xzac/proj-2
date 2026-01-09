from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
from evaluate import load
import mlflow
from huggingface_hub import login
import os   
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest",add_prefix_space=True)
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest", num_labels=3)
""" tokenizer.save_pretrained("./base_model")
model.save_pretrained("./base_model") """


def tokenize(examples):
    return tokenizer(examples["text"], max_length=512, truncation=True)
    
    

dataset = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset")
dataset.remove_columns(["id","sentiment"])
small_train_dataset = dataset["train"].shuffle(seed=42).select([i for i in list(range(400))])
small_test_dataset = dataset["test"].shuffle(seed=42).select([i for i in list(range(40))])

small_train_dataset = small_train_dataset.map(tokenize, batched=True,batch_size = 5)
small_test_dataset = small_test_dataset.map(tokenize, batched = True, batch_size = 2)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(eval_pred):
   load_accuracy = load("accuracy")
   load_f1 = load("f1")
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels, average= "macro")["f1"]
   return {"accuracy": accuracy, "f1": f1}


training_args = TrainingArguments(
    push_to_hub=True,
    hub_model_id= "divde/sentiment_analysis_classifier",
    hub_strategy="end",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_test_dataset,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
    data_collator=data_collator,
)

mlflow.set_experiment("Sentiment Analysis Classifier")
with mlflow.start_run():
    trainer.train()
    metrics=trainer.evaluate()
    mlflow.log_metrics(metrics)

    if metrics["eval_f1"] > 0.8:
        trainer.push_to_hub("divde/sentiment_analysis_classifier")
        mlflow.log_param("model_repo", "divde/sentiment_analysis_classifier")