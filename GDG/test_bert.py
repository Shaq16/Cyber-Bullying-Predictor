import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DistilBertForSequenceClassification

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the saved model and tokenizer
model_path = "cyberbullying_model"  # Ensure this folder exists
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()  # Set model to evaluation mode

# Function to predict if a text is cyberbullying or not
def predict(texts):
    if isinstance(texts, str):  # Convert single string input to a list
        texts = [texts]

    if not texts:
        return ["Invalid Input"]

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
    predictions = ["Cyberbullying" if p[1] > p[0] else "Not Cyberbullying" for p in probs]
    confidence = [round(float(max(p)) * 100, 2) for p in probs]  # Confidence percentage

    return [f"{pred} (Confidence: {conf}%)" for pred, conf in zip(predictions, confidence)]

# Example usage
texts = [
    "You are amazing!",
    "I hate you!",
    "Nice work, keep it up!",
    "You're the worst person ever."
]

results = predict(texts)
for text, result in zip(texts, results):
    print(f"Text: {text} => Prediction: {result}")
