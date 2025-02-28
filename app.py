import os
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from fastapi import FastAPI
from pydantic import BaseModel

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()

model_path='bert_mrpc_fine_tuned'


model = AutoModelForSequenceClassification.from_pretrained(model_path)



tokenizer = AutoTokenizer.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

class InputData(BaseModel):
    sentence1: str
    sentence2: str



@app.post("/predict")
def predict(data: InputData):
    inputs = tokenizer(data.sentence1, data.sentence2, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        pred_label = torch.argmax(logits, dim=1).item()

    return {"paraphrase": bool(pred_label), "label": pred_label}
