import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch.nn.functional as F

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
file_path = "cyberbullying_combined.csv"  # Change this to the actual filename
df = pd.read_csv(file_path)

# Drop missing values
df = df.dropna(subset=["Message", "label"])

# Convert labels to integer type (ensure consistency)
df["label"] = df["label"].astype(int)

# Split into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["Message"].astype(str).tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize_texts(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

train_encodings = tokenize_texts(train_texts)
test_encodings = tokenize_texts(test_texts)

# Dataset class for PyTorch
class CyberbullyingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Create datasets
train_dataset = CyberbullyingDataset(train_encodings, train_labels)
test_dataset = CyberbullyingDataset(test_encodings, test_labels)

# Load pre-trained DistilBERT model for binary classification
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,  # Adjust batch size based on GPU memory
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=200,
    fp16=torch.cuda.is_available(),  # Use mixed precision if on GPU
    save_strategy="epoch",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train model
trainer.train()

# Save trained model and tokenizer
model.save_pretrained("cyberbullying_model")
tokenizer.save_pretrained("cyberbullying_model")

# Function to predict if a text is cyberbullying or not
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    pred_label = torch.argmax(probs, dim=-1).item()
    return "Cyberbullying" if pred_label == 1 else "Not Cyberbullying"

# Example usage
new_text = "You are terrible!"
print(f"Prediction: {predict(new_text)}")
