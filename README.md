# Paraphrase Detection API using FastAPI & BERT

## 🚀 Overview
This FastAPI application serves a **BERT-based model** (`bert-base-uncased`) fine-tuned for **sentence pair classification**, determining whether two sentences are paraphrases.

## 📌 Features
- **FastAPI for serving the model**
- **Tokenization using Hugging Face Transformers**
- **Inference using PyTorch**


---

## 🛠 Installation

### 1️⃣ **Clone the repository**
```bash
git clone https://github.com/JisnaP/Fine_tuning_pretrained_models.git
cd Fine_tuning_pretrained_models
```

### 2️⃣ **Create a virtual environment **
```bash
python -m venv venv
source venv/bin/activate  
```

### 3️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```
📌 Setup Instructions
1️⃣ Run Colab Notebook to Fine-tune and Save Model
Before running the API, you must first run the Colab notebook to train and save the model parameters.

Open and run the Google Colab notebook: fine_tuning_pretrained_models_using_trainer_api_for_paraphrase_sentences.ipynb
Save the trained model and download it. (You will have use these as weights in the model in app.py)
Move model.pth to the models/ directory in your local repo.
---

## 🎯 Running the API

### **1️⃣ Start the FastAPI Server**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
- The API will now be available at: `http://127.0.0.1:8000`
- You can test the API using the interactive **Swagger UI** at: `http://127.0.0.1:8000/docs`

### **2️⃣ Example API Request**
Send a `POST` request to `/predict` with a JSON payload containing two sentences:
```json
{
    "sentence1": "The weather is nice today.",
    "sentence2": "It's a beautiful day outside."
}
```



### **3️⃣ Example API Response**
```json
{
    "paraphrase": true,
    "confidence": 0.89
}
```
- `paraphrase: true` means the sentences are paraphrases
- `confidence: 0.89` indicates an 89% confidence level

---

## 🔹 Model Used
This API uses the **BERT base uncased (`bert-base-uncased`)** model fine-tuned on a paraphrase detection dataset glue/mrpc.

If the model is not downloaded, it will be automatically fetched from Hugging Face.

---




## 📜 License
This project is open-source and available under the **MIT License**.

---

## 🔗 References
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)



