# Cyberbullying Detection using DistilBERT

This project aims to detect cyberbullying in text messages using a fine-tuned **DistilBERT** model. The model classifies text into two categories: **Cyberbullying** and **Not Cyberbullying**.

## Table of Contents
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Training the Model](#training-the-model)
- [Testing and Prediction](#testing-and-prediction)
- [Exporting and Sharing Environment](#exporting-and-sharing-environment)
- [Acknowledgments](#acknowledgments)

## Project Structure
```
|-- hackathon
    |-- bert_env/          # Virtual environment
    |-- cyberbullying_model/  # Trained model files
    |-- results/          # Training results and logs
    |-- datasets/         # CSV files for training
    |-- scripts/
        |-- train_bert.py    # Training script
        |-- test_bert.py     # Testing script
        |-- combine.py       # Data preprocessing
```

## Setup
1. Clone the repository and navigate to the project directory:
   ```sh
   git clone <repository_link>
   cd hackathon
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv bert_env
   # Windows
   bert_env\Scripts\activate
   # Linux/macOS
   source bert_env/bin/activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Training the Model
To train the model using the dataset:
```sh
python scripts/train_bert.py
```
The trained model will be saved inside `cyberbullying_model/`.

## Testing and Prediction
To test the model with sample text:
```sh
python scripts/test_bert.py
```
You can modify the `test_bert.py` file to input new text and evaluate predictions.

## Exporting and Sharing Environment
To share your virtual environment:
1. Export the dependencies:
   ```sh
   pip freeze > requirements.txt
   ```
2. Send `requirements.txt` to others. They can install dependencies using:
   ```sh
   pip install -r requirements.txt
   ```
3. Share the `cyberbullying_model/` folder for model weights.

## Acknowledgments
This project uses **DistilBERT** from the Hugging Face Transformers library. Special thanks to all contributors involved in this hackathon project.

